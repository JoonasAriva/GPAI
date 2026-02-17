import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet

from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet, resnet50, \
    ResNet50_Weights
from monai.networks.nets import resnet18, resnet34, resnet50
from gradient_reversal import GradientReversal
from model_zoo import MyGroupNorm, AttentionHeadV3


class ResNet3DDepth(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)
        print("Using Resnet50")

        # The final layer is model.fc
        # Find final feature vector dimensions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # now forward pass outputs feature vectors
        # Add own classifier head
        self.cls_head = nn.Linear(num_features, 3)

    def forward(self, x):
        feats = self.backbone(x)
        preds = self.cls_head(feats)
        return preds


class ResNet3D(nn.Module):

    def __init__(self, num_attention_heads=1, resnet_type="50",
                 frozen_backbone: bool = False, GRL: bool = False):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.frozen_backbone = frozen_backbone
        self.grl = GRL

        print("# of attention heads: ", self.num_attention_heads)

        self.resnet_type = resnet_type

        # self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        if resnet_type == "18":
            self.backbone = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)
        elif resnet_type == "34":
            self.backbone = resnet34(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)
        elif resnet_type == "50":
            self.backbone = resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)

        self.num_features = self.backbone.fc.in_features
        self.classifier = nn.Sequential(nn.Linear(self.num_features, 1))
        self.sig = nn.Sigmoid()
        self.backbone.fc = nn.Identity()

        self.cls_head = nn.Linear(self.num_features, 3)  # for depth predicitng

        self.L = self.num_features
        self.D = 128
        self.K = 1

        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        if self.grl:
            self.grad_reversal = GradientReversal(1)
            self.grl_classifier = nn.Sequential(nn.Linear(512, 1))

    def forward(self, x, scan_end, cam=False, **kwargs):
        out = dict()
        H = self.backbone(x[:scan_end])
        H = H.view(-1, self.num_features * 1 * 1)

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = attention_maps.view(self.num_attention_heads, -1)

        A = F.softmax(unnorm_A, dim=1)
        opposite_A = F.softmax(-unnorm_A, dim=1)

        all_agg_vectors = torch.einsum('tb,bf->tf', A, H)  # (num_heads, feature_dim)

        # M = all_agg_vectors.reshape(1, -1)

        # A = torch.mean(A, dim=0).view(1, -1)
        # M = torch.mm(A, H)

        Y_logits = self.classifier(all_agg_vectors)
        individual_predictions = self.classifier(H)
        Y_probs = self.sig(Y_logits)

        if self.grl:
            opposite_agg_vectors = torch.einsum('tb,bf->tf', opposite_A, H)
            opposite_Y_logits = self.grl_classifier(self.grad_reversal(opposite_agg_vectors))
            out["opposite_Y_logits"] = opposite_Y_logits

        # scores = Y_probs.chunk(2, dim=1)
        # print(scores)
        # left_score = scores[:,0]
        # right_score = scores[:,1]
        # scan_prob = 1 - (1 - left_score) * (1 - right_score)
        Y_hat = torch.ge(Y_probs, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = Y_logits
        out['attention_weights'] = A
        out['unnorm_A'] = unnorm_A

        preds = self.cls_head(H)
        out['depth_scores'] = preds
        out['individual_predictions'] = individual_predictions
        out["all_attention"] = F.softmax(attention_maps, dim=0)
        return out  # Y_prob, Y_hat, unnorm_A


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

        self.model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], norm_layer=MyGroupNorm)
        sd = resnet50(weights=ResNet50_Weights.DEFAULT).state_dict()
        self.model.load_state_dict(sd, strict=False)
        print("using resnet50!")
        num_features = self.model.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.model.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.model(x)
        # attn_output, attn_weights = self.transformer(feats)
        preds = self.cls_head(feats)
        return preds


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
        # modules = list(model.children())[:-2]
        # self.backbone = nn.Sequential(*modules)
        self.trans_block = TransformerBlock(dim=512, n_heads=4, mlp_ratio=4)
        # for keeping the pretraining task active

        # self.cls_head = nn.Linear(512, 3)

    def forward(self, x, scan_end, **kwargs):
        out = dict()
        H = self.model(x[:scan_end])
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        mixed_H = torch.squeeze(self.trans_block(H.unsqueeze(0)))

        attention_maps = [head(F.layer_norm(mixed_H, mixed_H.shape[1:])) for head in self.attention_heads]

        # attention_maps = [head(mixed_H) for head in self.attention_heads]

        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = attention_maps.view(self.num_attention_heads, -1)

        A = F.softmax(unnorm_A, dim=1)

        A = torch.mean(A, dim=0).view(1, -1)

        M = torch.mm(A, H)

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
