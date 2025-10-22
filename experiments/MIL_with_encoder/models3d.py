import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet
from monai.networks.nets import resnet18
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet

from model_zoo import MyGroupNorm, AttentionHeadV3


class ResNet3DDepth(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)

        # --- 2. Strip off the final classifier to get feature vectors ---
        # The final layer is model.fc
        num_features = self.backbone.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.backbone.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.backbone(x)
        preds = self.cls_head(feats)
        return preds


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=1.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch, N_patches, dim]
        h = self.norm1(x)
        h2, _ = self.attn(h, h, h)
        x = x + h2
        x = x + self.mlp(self.norm2(x))
        return x


class ResNetDepth2dPatches(nn.Module):
    def __init__(self, use_transformer_layers=False):
        super().__init__()
        self.use_transformer_layers = use_transformer_layers
        print("Using transformer layers: ", self.use_transformer_layers)

        if self.use_transformer_layers:
            self.transformer = TransformerBlock(dim=512, n_heads=8, mlp_ratio=1)

        self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
        sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()
        self.model.load_state_dict(sd, strict=False)

        num_features = self.model.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.model.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.model(x)
        attn_output, attn_weights = self.transformer(feats)
        preds = self.cls_head(feats)
        return preds, attn_weights


class TransAttention(nn.Module):
    def __init__(self, num_attention_heads=2, instnorm=False, resnet_type="18"):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm

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

        if instnorm:
            # load the resnet with instance norm instead of batch norm
            if resnet_type == "18":
                self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            self.model.load_state_dict(sd, strict=False)

        else:
            if resnet_type == "18":
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        #modules = list(model.children())[:-2]
        #self.backbone = nn.Sequential(*modules)
        self.trans_block = TransformerBlock(dim=512, n_heads=8, mlp_ratio=4)
        # for keeping the pretraining task active

        #self.cls_head = nn.Linear(512, 3)

    def forward(self, x, scan_end, **kwargs):
        out = dict()
        H = self.model(x[:scan_end])
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        mixed_H = torch.squeeze(self.trans_block(H.unsqueeze(0)))

        attention_maps = [head(F.layer_norm(mixed_H, mixed_H.shape[1:])) for head in self.attention_heads]

        #attention_maps = [head(mixed_H) for head in self.attention_heads]

        attention_maps = torch.cat(attention_maps, dim=1)
        print("attention maps: ",attention_maps.shape)
        unnorm_A = attention_maps.view(self.num_attention_heads, -1)
        print("unnorm_A: ", unnorm_A.shape)
        A = F.softmax(unnorm_A, dim=1)
        print("A: ", A.shape)
        A = torch.mean(A, dim=0).view(1, -1)
        print("A: ", A.shape)
        M = torch.mm(A, H)
        print("M: ", M.shape)
        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = Y_prob
        out['attention_weights'] = A
        out['unnorm_A'] = unnorm_A
        if self.num_attention_heads > 1:
            out["all_attention"] = F.softmax(attention_maps, dim=0)
        return out
