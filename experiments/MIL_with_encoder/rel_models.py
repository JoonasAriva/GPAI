import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet

from model_zoo import AttentionHeadV3, GhostBatchNorm2d, MyGroupNorm


class MuNormalTracker:
    def __init__(self, feature_dim, momentum=0.99, device='cpu'):
        self.mu = torch.zeros(feature_dim, device=device)
        self.momentum = momentum
        self.initialized = False
        print("Using normal vector MU tracker!")

    def update(self, features, labels):
        # Select only normal samples (label == 0)
        mask = torch.squeeze(labels) == 0
        if mask.any():
            normal_features = features[mask, :].mean(dim=0)

            if not self.initialized:
                self.mu = normal_features.detach()
                self.initialized = True
            else:

                self.mu = self.momentum * self.mu + (1 - self.momentum) * normal_features.detach()

    def get_mu(self):
        return self.mu


class MuDataSourceTracker:
    def __init__(self, feature_dim, momentum=0.99, device='cpu'):
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = device
        self.mu_dict = {}  # {source_name: mu_vector}
        print("Using data source MU tracker!")

    def update(self, features, src):

        f = features.detach().mean(dim=0)

        if src not in self.mu_dict:
            self.mu_dict[src] = f.clone()
        else:
            self.mu_dict[src] = (
                    self.momentum * self.mu_dict[src] +
                    (1 - self.momentum) * f
            )

    def get_mu(self, source):
        source = str(source)
        if source in self.mu_dict:
            return self.mu_dict[source]
        else:
            return torch.zeros(self.feature_dim, device=self.device)

    def to(self, device):
        self.device = device
        for k in self.mu_dict:
            self.mu_dict[k] = self.mu_dict[k].to(device)


class ResNetRel(nn.Module):

    def __init__(self, num_attention_heads=1, instnorm=False, ghostnorm=False, resnet_type="18", p=0.5, **kwargs):
        super().__init__()

        self.instnorm = instnorm
        self.num_attention_heads = num_attention_heads
        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type
        self.p = p
        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()
        if ghostnorm:
            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=GhostBatchNorm2d)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=GhostBatchNorm2d)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)

        elif instnorm:
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
        # self.mu_tracker = MuNormalTracker(feature_dim=512)
        self.mu_tracker = MuDataSourceTracker(feature_dim=512)

    def disable_dropout(self):
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x, scan_end, label, source_label, cam=False):
        out = dict()
        H = self.backbone(x[:scan_end])
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = torch.mean(attention_maps, dim=1)
        unnorm_A = unnorm_A.view(1, -1)

        A = F.softmax(unnorm_A, dim=1)

        if cam:
            Y_probs = self.classifier(H)
            out['scores'] = Y_probs
            out['attention_weights'] = unnorm_A
            return out

        M = torch.mm(A, H)
        if self.training and torch.rand(1).item() < self.p:
            M_relative = M - self.mu_tracker.get_mu(source_label).cuda()
        else:
            M_relative = M
        # M_relative = M - self.mu_tracker.get_mu(source_label).cuda()
        Y_prob = self.classifier(M_relative)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        out['predictions'] = Y_hat
        out['scores'] = Y_prob
        out['attention_weights'] = unnorm_A

        self.mu_tracker.update(M, source_label)
        return out  # Y_prob, Y_hat, unnorm_A
