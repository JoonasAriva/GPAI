import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import Tensor as T
from torchvision.models import resnet
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


# from experiments.MIL_with_encoder.models import Attention


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class AttentionHeadV3(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionHeadV3, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, K),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)


class ResNetAttentionV3(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=False, resnet_type="18"):
        super().__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm

        print("Using neighbour attention with a range of ", self.neighbour_range)
        print("# of attention heads: ", self.num_attention_heads)
        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

    def disable_dropout(self):
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x, return_unnorm_attention=True, scorecam_wrt_classifier_score=False, full_pass=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        if self.neighbour_range != 0:
            combinedH = H.view(-1)
            combinedH = F.pad(combinedH, (self.L * self.neighbour_range, self.L * self.neighbour_range), "constant",
                              0)  # TODO: mirror padding?
            combinedH = combinedH.unfold(0, self.L * (self.neighbour_range * 2 + 1), self.L)

            H = 0.25 * combinedH[:, :self.L] + 0.5 * combinedH[:, self.L:2 * self.L] + 0.25 * combinedH[:, 2 * self.L:]

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = torch.mean(attention_maps, dim=1)
        unnorm_A = unnorm_A.view(1, -1)

        A = F.softmax(unnorm_A, dim=1)
        # A = unnorm_A / (torch.sum(unnorm_A) + 0.01)

        if scorecam_wrt_classifier_score:
            Y_probs = self.classifier(H)
            return None, None, Y_probs

        M = torch.mm(A, H)
        # print("M", M.shape)
        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        return Y_prob, Y_hat, unnorm_A


class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, attention_masking: bool = False, keep_neighbours: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (self.embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.keep_neighbours = keep_neighbours
        self.attention_masking = attention_masking

    def create_mask(self, matrix_size: int, unmasked_neighbours: int):
        a = torch.zeros(matrix_size, matrix_size)
        a = torch.diagonal_scatter(a, torch.ones(matrix_size), 0)
        for i in range(unmasked_neighbours):
            a = torch.diagonal_scatter(a, torch.ones(matrix_size - i - 1), 1 + i)
            a = torch.diagonal_scatter(a, torch.ones(matrix_size - i - 1), -1 - i)
        return a

    def head_partition(self, x: T):
        return einops.rearrange(x, 's (h d) -> h s d', h=self.num_heads)

    def head_merging(self, x: T):
        return einops.rearrange(x, 'h s d -> s (h d)')

    def forward(self, x: T):

        q, k, v = self.project_qkv(x).chunk(3, dim=-1)

        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = q @ k.transpose(-1, -2) * self.scale
        if self.attention_masking:
            MASKING_VALUE = -1e+10 if attn_scores.dtype == torch.float32 else -1e+4
            mask = self.create_mask(attn_scores.size()[-1], unmasked_neighbours=self.keep_neighbours)
            mask = torch.unsqueeze(mask, dim=0)
            attn_scores[~mask.bool()] = MASKING_VALUE
        attn_weights = self.softmax(attn_scores)
        out = attn_weights @ v
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out, attn_scores


class ResNetSelfAttention(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=True, resnet_type="18"):
        super().__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.self_attention = SelfAttention(embed_dim=512, num_heads=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 512))
        torch.nn.init.trunc_normal_(self.cls_token)

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H = torch.concat((self.cls_token, H), dim=0)

        H, weights = self.self_attention(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(CLS)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat


class FFN(nn.Sequential):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )


class TransformerEncoderBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        super().__init__()

        self.attn = SelfAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: T):
        res = x

        out, weights = self.attn(x)

        out = self.norm1(res + out)

        res = out
        out = self.ffn(out)
        out = self.norm2(res + out)

        return out, weights


class ResNetTransformer(ResNetSelfAttention):

    def __init__(self, nr_of_blocks=2):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(nr_of_blocks):
            self.blocks.append(TransformerEncoderBlock(embed_dim=512, num_heads=1, hidden_dim=512))

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        for i, block in enumerate(self.blocks):
            if i + 1 == len(self.blocks):
                H = torch.concat((self.cls_token, H), dim=0)
            H, weights = block(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(CLS)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(max_len, embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: T) -> T:
        return x + self.pos_embed[:x.shape[0]]


class ResNetTransformerPosEmbed(ResNetSelfAttention):

    def __init__(self, instnorm):
        super().__init__(instnorm)
        self.pos_embed = PositionalEmbedding(300, embed_dim=512)

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H = self.pos_embed(H)
        H = torch.concat((self.cls_token, H), dim=0)

        H, weights = self.self_attention(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(CLS)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat


class ResNetTransformerPosEnc(ResNetSelfAttention):

    def __init__(self, instnorm):
        super().__init__(instnorm)
        self.pos_enc = PositionalEncoding1D(512)

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H = torch.squeeze(self.pos_enc(torch.unsqueeze(H, dim=2)))
        H = torch.concat((self.cls_token, H), dim=0)

        H, weights = self.self_attention(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(CLS)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat


class ResNetGrouping(ResNetSelfAttention):

    def __init__(self, instnorm):
        super().__init__(instnorm)

        self.attention = AttentionHeadV3(512, 128, 1)

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        groups = H.unfold(dimension=0, size=10, step=10).permute(0, 2, 1)
        group_average_features = groups.mean(axis=1)
        group_attentions = self.attention(group_average_features)

        A = F.softmax(group_attentions, dim=1).T
        M = torch.mm(A, group_average_features)

        # H = torch.concat((self.cls_token, H), dim=0)
        #
        # H, weights = self.self_attention(H)
        #
        # CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, group_attentions, M, group_average_features
        else:
            return Y_prob, Y_hat


class SelfSelectionNet(ResNetSelfAttention):

    def __init__(self, instnorm):
        super().__init__(instnorm)

        self.attention = AttentionHeadV3(512, 128, 1)
        self.relu = nn.LeakyReLU()
        self.window_size = 10

    def forward(self, x, include_weights=False):

        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        groups = H.unfold(dimension=0, size=self.window_size, step=self.window_size).permute(0, 2, 1)
        group_average_features = groups.mean(axis=1)  # mean or max or highest abs value?
        group_attentions = self.attention(group_average_features)

        thresholded_attentions = self.relu(group_attentions)
        attentions = einops.repeat(thresholded_attentions, 'i j-> (i copy) j', copy=self.window_size)
        A = F.softmax(attentions, dim=1).T
        M = torch.mm(A, H)

        # H = torch.concat((self.cls_token, H), dim=0)
        #
        # H, weights = self.self_attention(H)
        #
        # CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if include_weights:
            return Y_prob, Y_hat, group_attentions, thresholded_attentions, attentions, A
        else:
            return Y_prob, Y_hat


class TwoStageNet(nn.Module):

    def __init__(self, instnorm=False):
        super().__init__()

        self.instnorm = instnorm

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.attention_head = AttentionHeadV3(self.L, self.D, self.K)
        self.important_head = AttentionHeadV3(self.L, self.D, self.K)
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 3))

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
            sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            model.load_state_dict(sd, strict=False)
        else:

            model = resnet18(weights=ResNet18_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=False)

    def disable_dropout(self):
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x):

        H = self.backbone(x)

        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        rois = self.attention_head(H)
        rois = rois.view(1, -1)

        rois_important = self.relu(rois)
        rois_non_important = self.relu(-1 * rois)

        MASKING_VALUE = -1e+10 if rois.dtype == torch.float32 else -1e+4

        mask_important = rois_important == 0
        mask_non_important = rois_non_important == 0

        rois_important = torch.where(mask_important, torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                     rois_important)
        rois_non_important = torch.where(mask_non_important,
                                         torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                         rois_non_important)

        rois_important = F.softmax(rois_important, dim=1)
        rois_non_important = F.softmax(rois_non_important, dim=1)

        M_important = torch.mm(rois_important, H)
        M_non_important = torch.mm(rois_non_important, H)

        important_probs = self.classifier(M_important)
        non_important_probs = self.classifier(M_non_important)

        return important_probs, non_important_probs, rois, H


class TwoStageNetTwoHeads(TwoStageNet):
    def __init__(self, instnorm=False):
        super().__init__()
        self.tumor_classifier = nn.Linear(self.L * self.K, 1)
        self.relevancy_classifier = nn.Linear(self.L * self.K, 1)

    def forward(self, x):
        H = self.backbone(x)

        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        rois = self.attention_head(H)
        rois = rois.view(1, -1)

        rois_important = self.relu(rois)
        rois_non_important = self.relu(-1 * rois)

        MASKING_VALUE = -1e+10 if rois.dtype == torch.float32 else -1e+4

        mask_important = rois_important == 0
        mask_non_important = rois_non_important == 0

        rois_important = torch.where(mask_important, torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                     rois_important)
        rois_non_important = torch.where(mask_non_important,
                                         torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                         rois_non_important)

        rois_important = F.softmax(rois_important, dim=1)
        rois_non_important = F.softmax(rois_non_important, dim=1)

        M_important = torch.mm(rois_important, H)
        M_non_important = torch.mm(rois_non_important, H)

        important_tumor_probs = self.tumor_classifier(M_important)

        non_important_relevancy_probs = self.relevancy_classifier(M_non_important)
        important_relevancy_probs = self.relevancy_classifier(M_important)

        return important_tumor_probs, non_important_relevancy_probs, important_relevancy_probs, rois


class TwoStageNetTwoHeadsV2(TwoStageNet):
    def __init__(self, instnorm=False):
        super().__init__()
        self.tumor_classifier = nn.Linear(self.L * self.K, 1)
        self.relevancy_classifier = nn.Linear(self.L * self.K, 1)

        self.attention_head = AttentionHeadV3(self.L, self.D, self.K)
        self.roi_head = AttentionHeadV3(self.L, self.D, self.K)

    def forward(self, x):
        H = self.backbone(x)

        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        rois = self.roi_head(H)
        rois = rois.view(1, -1)

        rois_important = self.relu(rois)
        rois_non_important = self.relu(-1 * rois)

        MASKING_VALUE = -1e+10 if rois.dtype == torch.float32 else -1e+4

        mask_important = rois_important == 0
        mask_non_important = rois_non_important == 0

        rois_non_important = torch.where(mask_non_important,
                                         torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                         rois_non_important)

        rois_important = torch.where(mask_important,
                                     torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                     rois_non_important)

        rois_non_important = F.softmax(rois_non_important, dim=1)

        attention = self.attention_head(H).view(1, -1)
        attention = torch.where(mask_important, torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                attention)

        attention = F.softmax(attention, dim=1)

        M_important = torch.mm(attention, H)
        M_non_important_rel = torch.mm(rois_non_important, H)

        important_tumor_probs = self.tumor_classifier(M_important)

        rois_important = F.softmax(rois_important, dim=1)
        M_important_rel = torch.mm(rois_important, H)

        non_important_relevancy_probs = self.relevancy_classifier(M_non_important_rel)
        important_relevancy_probs = self.relevancy_classifier(M_important_rel)

        return important_tumor_probs, non_important_relevancy_probs, important_relevancy_probs, rois, attention


class TwoStageNetMaskedAttention(TwoStageNet):
    def __init__(self, instnorm=False):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim=512, num_heads=1, attention_masking=True, keep_neighbours=3)

    def forward(self, x):
        H = self.backbone(x)

        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H, attn_weights = self.self_attention(H)
        rois = self.attention_head(H)
        rois = rois.view(1, -1)

        rois_important = F.softmax(rois, dim=1)
        rois_non_important = F.softmax(-1 * rois, dim=1)

        M_important = torch.mm(rois_important, H)
        M_non_important = torch.mm(rois_non_important, H)

        important_probs = self.classifier(M_important)
        non_important_probs = self.classifier(M_non_important)

        return important_probs, non_important_probs, rois


class MultiHeadTwoStageNet(TwoStageNet):
    def __init__(self, instnorm=False):
        super().__init__()

        self.all_slices_head = AttentionHeadV3(self.L, self.D, self.K)
        self.all_slices_classifier = nn.Linear(self.L * self.K, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.backbone(x)

        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        rois = self.attention_head(H)
        rois = rois.view(1, -1)

        rois_important = self.relu(rois)
        rois_non_important = self.relu(-1 * rois)

        MASKING_VALUE = -1e+10 if rois.dtype == torch.float32 else -1e+4

        mask_important = rois_important == 0
        mask_non_important = rois_non_important == 0

        rois_important = torch.where(mask_important, torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                     rois_important)
        rois_non_important = torch.where(mask_non_important,
                                         torch.tensor(MASKING_VALUE, device=rois.device, dtype=rois.dtype),
                                         rois_non_important)

        rois_important = F.softmax(rois_important, dim=1)
        rois_non_important = F.softmax(rois_non_important, dim=1)

        M_important = torch.mm(rois_important, H)
        M_non_important = torch.mm(rois_non_important, H)

        important_probs = self.classifier(M_important)
        non_important_probs = self.classifier(M_non_important)

        attention = self.all_slices_head(H)
        A = F.softmax(attention, dim=1).T
        M = torch.mm(A, H)
        logits = self.all_slices_classifier(M)
        Y_hat = self.sigmoid(logits)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        return important_probs, non_important_probs, rois, logits, Y_hat, attention


class TwoStageNetSimple(TwoStageNet):

    def __init__(self, instnorm=False):
        super().__init__()

    def forward(self, x, df):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)  # feature vectors from resnet18

        rois = self.attention_head(H)  # get attention scores for each vector
        rois = rois.view(1, -1)

        # find important slices from pre-calculated dataframe

        df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
        df["important_all"] = df["important_all"].fillna(0)
        df.reset_index(inplace=True)
        # categorize slice vectors by the dataframe
        H_important = H[df["important_all"] == 1]
        H_non_important = H[df["important_all"] == 0]

        # categorize attention scores by the dataframe
        rois_important = rois[:, df["important_all"] == 1]
        rois_non_important = rois[:, df["important_all"] == 0]

        # normalize attention scores
        rois_important = rois_important / rois_important.sum()
        rois_non_important = rois_non_important / rois_non_important.sum()

        # aggregate together feature vectors, use attention scores as weights
        M_important = torch.mm(rois_important, H_important)
        M_non_important = torch.mm(rois_non_important, H_non_important)

        # final classification
        important_probs = self.classifier(M_important)
        non_important_probs = self.classifier(M_non_important)

        return important_probs, non_important_probs, rois


class TransMIL(nn.Module):

    def __init__(self):
        super().__init__()

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Sequential(nn.Linear(512, 1))
        self.sig = nn.Sigmoid()

        model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
        sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        model.load_state_dict(sd, strict=False)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.self_attention1 = SelfAttention(embed_dim=512, num_heads=8)
        self.self_attention2 = SelfAttention(embed_dim=512, num_heads=8)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.cls_token = nn.Parameter(torch.zeros(1, 512))
        torch.nn.init.trunc_normal_(self.cls_token)

    def forward(self, x):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H = torch.concat((self.cls_token, H), dim=0)

        H = H + self.self_attention1(self.norm1(H))[0]
        H_out, weights = self.self_attention2(self.norm2(H))
        H = H_out + H
        CLS = torch.unsqueeze(self.norm3(H)[0, :], dim=0)

        Y_prob = self.classifier(CLS)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        return Y_prob, Y_hat, H_out, weights


class ResNetDepth(nn.Module):

    def __init__(self, instnorm=False, resnet_type="18"):
        super().__init__()
        self.instnorm = instnorm

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        depth_scores = self.classifier(H)

        return depth_scores, H


class TransDepth(nn.Module):

    def __init__(self, instnorm=False, resnet_type="18"):
        super().__init__()
        self.instnorm = instnorm
        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Sequential(nn.Linear(512, 1))
        self.sig = nn.Sigmoid()

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        self.self_attention1 = SelfAttention(embed_dim=512, num_heads=8)
        self.self_attention2 = SelfAttention(embed_dim=512, num_heads=8)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        H = H + self.self_attention1(self.norm1(H))[0]
        depth_scores = self.classifier(self.norm2(H))

        return depth_scores


class CompassModel(ResNetDepth):
    def __init__(self, instnorm=False, resnet_type="18"):
        super().__init__(instnorm, resnet_type)

        self.attention = AttentionHeadV3(512, 128, 1)
        self.tumor_classifier = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scan_end):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        H = H[:scan_end]

        depth_scores = self.classifier(H)
        attention_scores = self.attention(H)
        softmaxed_attention_scores = F.softmax(attention_scores, dim=1)

        aggregated_vec = torch.mm(softmaxed_attention_scores.T, H)
        tumor_score = self.tumor_classifier(aggregated_vec)
        prediction = self.sigmoid(tumor_score)
        Y_hat = torch.ge(prediction, 0.5).float()

        individual_preds = self.tumor_classifier(H)

        return depth_scores, tumor_score, Y_hat, attention_scores, individual_preds


class CompassModelV2(ResNetDepth):
    def __init__(self, instnorm=False, resnet_type="18"):
        super().__init__(instnorm, resnet_type)

        self.depth_attention = nn.Sequential(nn.Linear(1, 4),
                                             nn.ReLU(),
                                             nn.Linear(4, 1))
        self.tumor_classifier = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scan_end):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        H = H[:scan_end]
        depth_scores = self.classifier(H)
        attention_scores = self.depth_attention(depth_scores)
        softmaxed_attention_scores = F.softmax(attention_scores, dim=1)

        aggregated_vec = torch.mm(softmaxed_attention_scores.T, H)
        tumor_score = self.tumor_classifier(aggregated_vec)
        prediction = self.sigmoid(tumor_score)
        Y_hat = torch.ge(prediction, 0.5).float()

        return depth_scores, tumor_score, Y_hat, attention_scores


class TwoStageCompass(ResNetDepth):
    def __init__(self, instnorm=False, resnet_type="18"):
        super().__init__(instnorm, resnet_type)

        self.tumor_classifier = nn.Linear(512, 1)
        self.relevancy_classifier = nn.Linear(512, 1)

        self.depth_range = nn.Parameter(torch.Tensor([-0.5, 0.5]))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scan_end):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        H = H[:scan_end]

        depth_scores = self.classifier(H)

        relevancy_scores = self.relevancy_classifier(H)
        rel_prediction = self.sigmoid(relevancy_scores)
        Y_hat_rel = torch.ge(rel_prediction, 0.5).float()

        MASKING_VALUE = -1e+10 if depth_scores.dtype == torch.float32 else -1e+4
        important_mask = torch.where(
            (self.depth_range[0].item() < depth_scores) & (depth_scores < self.depth_range[1].item()), 1,
            MASKING_VALUE)

        important_mask = F.softmax(important_mask.T, dim=1)
        important_vector = torch.mm(important_mask, H)
        tumor_score = self.tumor_classifier(important_vector)
        prediction = self.sigmoid(tumor_score)
        Y_hat = torch.ge(prediction, 0.5).float()

        return depth_scores, tumor_score, Y_hat, relevancy_scores, Y_hat_rel
