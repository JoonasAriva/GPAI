import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch import Tensor as T
from torchvision.models import resnet
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


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
            nn.Linear(D, K)
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

        # self.attention_heads.apply(init_weights)

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

    def forward(self, x, return_unnorm_attention=False, scorecam_wrt_classifier_score=False, full_pass=False):

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

        if full_pass:
            Y_probs = self.classifier(H)
            # Y_probs = self.sig(Y_probs)
            return Y_prob, Y_hat, unnorm_A, Y_probs  #
        if return_unnorm_attention:
            return Y_prob, Y_hat, A
        else:  # i think this part is not used currently
            return Y_prob, Y_hat, unnorm_A, H, attention_maps


class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (self.embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def head_partition(self, x: T):
        return einops.rearrange(x, 's (h d) -> h s d', h=self.num_heads)

    def head_merging(self, x: T):
        return einops.rearrange(x, 'h s d -> s (h d)')

    def forward(self, x: T):
        q, k, v = self.project_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = q @ k.transpose(-1, -2) * self.scale
        attn_weights = self.softmax(attn_scores)

        out = attn_weights @ v
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out, attn_scores


class ResNetBackbone(nn.Module):
    def __init__(self, instnorm=True):
        super().__init__()

        if instnorm:
            # load the resnet with instance norm instead of batch norm
            model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
            sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            model.load_state_dict(sd, strict=False)
        else:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)

        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        H = self.backbone(x)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        return H


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        logits = self.classifier(x)
        probabilities = torch.sigmoid(logits)
        predictions = torch.ge(probabilities, 0.5).float()
        return probabilities, predictions


class ResNetSelfAttention2(nn.Module):

    def __init__(self, instnorm=True):
        super().__init__()

        self.resnet_backbone = ResNetBackbone(instnorm=instnorm)
        self.cls_head = ClassificationHead()

        self.self_attention = SelfAttention(embed_dim=512, num_heads=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 512))
        torch.nn.init.trunc_normal_(self.cls_token)

    def forward(self, x, include_weights=False):

        H = self.resnet_backbone(x)
        H = torch.concat((self.cls_token, H), dim=0)
        H, weights = self.self_attention(H)
        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob, Y_hat = self.cls_head(CLS)

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


class ResNetTransformer(ResNetSelfAttention2):

    def __init__(self, nr_of_blocks=2):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(nr_of_blocks):
            self.blocks.append(TransformerEncoderBlock(embed_dim=512, num_heads=1, hidden_dim=512))

    def forward(self, x, include_weights=False):

        H = self.resnet_backbone(x)

        for i, block in enumerate(self.blocks):
            if i + 1 == len(self.blocks):
                H = torch.concat((self.cls_token, H), dim=0)
            H, weights = block(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob, Y_hat = self.cls_head(CLS)

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


class ResNetTransformerPosEmbed(ResNetSelfAttention2):

    def __init__(self):
        super().__init__()
        self.pos_embed = PositionalEmbedding(300, embed_dim=512)

    def forward(self, x, include_weights=False):

        H = self.resnet_backbone(x)
        H = self.pos_embed(H)
        H = torch.concat((self.cls_token, H), dim=0)

        H, weights = self.self_attention(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob, Y_hat = self.cls_head(CLS)

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat


class ResNetTransformerPosEnc(ResNetSelfAttention2):

    def __init__(self):
        super().__init__()
        self.pos_enc = PositionalEncoding1D(512)

    def forward(self, x, include_weights=False):

        H = self.resnet_backbone(x)
        H = torch.squeeze(self.pos_enc(torch.unsqueeze(H, dim=2)))
        H = torch.concat((self.cls_token, H), dim=0)

        H, weights = self.self_attention(H)

        CLS = torch.unsqueeze(H[0, :], dim=0)

        Y_prob, Y_hat = self.cls_head(CLS)

        if include_weights:
            return Y_prob, Y_hat, weights
        else:
            return Y_prob, Y_hat
