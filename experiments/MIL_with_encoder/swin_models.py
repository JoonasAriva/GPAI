import einops
import torch
import torch.nn as nn
#from monai.networks.nets.swin_unetr import SwinTransformer
from modded_swin_model import SwinTransformer
from monai.utils import ensure_tuple_rep
from torch import Tensor as T


class SWINCompass(nn.Module):
    def __init__(self):
        super().__init__()

        spatial_dims = 3
        window_size = ensure_tuple_rep(7, spatial_dims)  # 7
        self.backbone = SwinTransformer(in_chans=1,
                                        embed_dim=48,
                                        window_size=window_size,
                                        patch_size=(4, 4, 4),  # 7,7,7
                                        depths=(2, 2, 2, 2),
                                        num_heads=(3, 6, 12, 24),
                                        mlp_ratio=4.0,
                                        qkv_bias=True,
                                        drop_rate=0.0,
                                        attn_drop_rate=0.0,
                                        drop_path_rate=0.0,
                                        norm_layer=nn.LayerNorm,
                                        use_checkpoint=False,
                                        spatial_dims=3,
                                        downsample='merging',
                                        use_v2=False)
        self.classifier = nn.Linear(768, 3)

    def forward(self, x):
        x = self.backbone(x)[-1]
        b, c, h, w, d = x.shape
        #print("before einops in model: ", x.shape)
        x = einops.rearrange(x, 'b c h w d -> b (h w d) c')
        #print("in model after einops: ", x.shape)
        scores = self.classifier(x)
        return scores, (h, w, d)


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

    def head_partition(self, x: T):
        return einops.rearrange(x, 'b s (h d) -> b h s d', h=self.num_heads)

    def head_merging(self, x: T):
        return einops.rearrange(x, 'b h s d -> b s (h d)')

    def forward(self, x: T):
        q, k, v = self.project_qkv(x).chunk(3, dim=-1)

        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = q @ k.transpose(-1, -2) * self.scale

        attn_weights = self.softmax(attn_scores)
        out = attn_weights @ v
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out, attn_scores


class SWINClassifier(SWINCompass):
    def __init__(self):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 768))
        torch.nn.init.trunc_normal_(self.cls_token)

        self.self_attention = SelfAttention(embed_dim=768, num_heads=1)
        self.tumor_classifier = nn.Linear(768, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)[-1]
        b, c, h, w, d = x.shape

        x = einops.rearrange(x, 'b c h w d -> b (h w d) c')

        CLS = einops.repeat(self.cls_token, 'm n -> k m n', k=b)

        x = torch.concat((CLS, x), dim=1)
        x, weights = self.self_attention(x)

        CLS = x[:, 0, :]

        scores = self.tumor_classifier(CLS)
        prediction = self.sig(scores)
        prediction = torch.ge(prediction, 0.5).float()

        out = dict()
        out['predictions'] = prediction
        out['scores'] = scores
        return out, weights, (h, w, d)
