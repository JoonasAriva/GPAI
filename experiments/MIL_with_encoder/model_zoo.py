import einops
import numpy as np
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
        num_groups = max(1, num_channels // 8)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class GhostBatchNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.nano_bs = 20
        # self.register_buffer("nano_bs", torch.Tensor([self.nano_bs]))
        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        chunks = X.chunk(int(np.ceil(X.shape[0] / self.nano_bs)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


class AttentionHeadV3(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionHeadV3, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(0.2),  ## test this out
            nn.Linear(D, K),
        )

    def forward(self, x):
        return self.head(x)


class ResNetAttentionV3(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=False, ghostnorm=False, resnet_type="18", frozen_backbone: bool = False):
        super().__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm
        self.frozen_backbone = frozen_backbone

        print("Using neighbour attention with a range of ", self.neighbour_range)
        print("# of attention heads: ", self.num_attention_heads)

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type

        # self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        self.classifier = nn.Sequential(nn.Linear(512, 1))
        self.sig = nn.Sigmoid()
        if ghostnorm:
            if resnet_type == "18":
                self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=GhostBatchNorm2d)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=GhostBatchNorm2d)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            self.model.load_state_dict(sd, strict=False)

        elif instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            self.model.load_state_dict(sd, strict=False)
            self.model.fc = nn.Identity()
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # modules = list(model.children())[:-2]
        # self.backbone = nn.Sequential(*modules)

        self.cls_head = nn.Linear(512, 3)  # for depth predicitng
        if self.frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                print("Backbone frozen")
    def forward(self, x, scan_end, cam=False, **kwargs):
        out = dict()
        H = self.model(x[:scan_end])
        # H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = attention_maps.view(self.num_attention_heads, -1)

        A = F.softmax(unnorm_A, dim=1)

        all_agg_vectors = torch.einsum('tb,bf->tf', A, H)  # (num_heads, feature_dim)
        M = all_agg_vectors.reshape(1, -1)

        # A = torch.mean(A, dim=0).view(1, -1)
        # M = torch.mm(A, H)

        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = Y_prob
        out['attention_weights'] = A
        out['unnorm_A'] = unnorm_A

        preds = self.cls_head(H)
        out['depth_scores'] = preds

        out["all_attention"] = F.softmax(attention_maps, dim=0)
        return out  # Y_prob, Y_hat, unnorm_A


class OrganModel(nn.Module):

    def __init__(self, num_attention_heads=1, instnorm=False):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm

        print("# of attention heads: ", self.num_attention_heads)

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1

        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        # self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()

        if instnorm:
            # load the resnet with instance norm instead of batch norm

            self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
            sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            self.model.load_state_dict(sd, strict=False)
            self.model.fc = nn.Identity()
        else:
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def forward(self, x, scan_end, **kwargs):
        out = dict()
        H = self.model(x[:scan_end])
        H = H.view(-1, 512 * 1 * 1)

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = attention_maps.view(self.num_attention_heads, -1)

        A = F.softmax(unnorm_A, dim=1)

        out['attention_weights'] = A
        out['unnorm_A'] = unnorm_A

        if self.num_attention_heads > 1:
            out["all_attention"] = F.softmax(attention_maps, dim=0)
        return out


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

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=True, resnet_type="34"):
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
        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])
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
        # self.cls_token = nn.Parameter(torch.zeros(1, 512))
        # torch.nn.init.trunc_normal_(self.cls_token)

    def forward(self, x, scan_end, include_weights=False, **kwargs):
        out = dict()
        H = self.backbone(x[:scan_end])
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        # H = torch.concat((self.cls_token, H), dim=0)

        H2, weights = self.self_attention(H)

        # CLS = torch.unsqueeze(H[0, :], dim=0)

        attention_maps = [head(H2) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = torch.mean(attention_maps, dim=1)
        unnorm_A = unnorm_A.view(1, -1)
        A = F.softmax(unnorm_A, dim=1)
        M = torch.mm(A, H)
        # print("M", M.shape)
        Y_prob = self.classifier(M)

        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = Y_prob
        out['attention_weights'] = A
        out['unnorm_A'] = unnorm_A

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return out


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

    def __init__(self, instnorm=False, ghostnorm: bool = False, resnet_type="34"):
        super().__init__()
        print("Resnet type: {}".format(resnet_type))
        self.instnorm = instnorm

        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))

        if ghostnorm:
            print("Using GhostNorm!")
            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=GhostBatchNorm2d)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=GhostBatchNorm2d)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)

        elif self.instnorm:
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

    def forward(self, x, scan_end):
        # print(f"Forward pass on {x.device}")
        # print("x shape: ", x.shape)
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H[:scan_end]
        H = H.view(-1, 512 * 1 * 1)

        depth_scores = self.classifier(H)

        return depth_scores


class DepthTumor(ResNetDepth):
    def __init__(self, instnorm=False, ghostnorm=False, resnet_type="18"):
        super().__init__(instnorm=instnorm, ghostnorm=ghostnorm, resnet_type=resnet_type)

        self.tmr_classifier = nn.Linear(512, 1)
        self.attention_head = AttentionHeadV3(512, 128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, scan_end, cam=False, **kwargs):
        out = dict()

        H = self.backbone(x[:scan_end])
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        unnorm_A = self.attention_head(H)
        A = F.softmax(unnorm_A, dim=1)
        M = torch.mm(A.T, H)
        depth_scores = self.classifier(H)
        tmr_scores = self.tmr_classifier(M)

        Y_hat = self.sig(tmr_scores)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = tmr_scores
        out['attention_weights'] = unnorm_A
        out['depth_scores'] = depth_scores
        return out


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

        normalized_attention_scores = attention_scores / (attention_scores.abs().sum() + 0.001)
        # softmaxed_attention_scores = F.softmax(attention_scores, dim=1)

        aggregated_vec = torch.mm(normalized_attention_scores.T, H)
        tumor_score = self.tumor_classifier(aggregated_vec)
        prediction = self.sigmoid(tumor_score)
        Y_hat = torch.ge(prediction, 0.5).float()

        return depth_scores, tumor_score, Y_hat, attention_scores
