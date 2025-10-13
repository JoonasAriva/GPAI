import torch.nn as nn
from monai.networks.nets import resnet
from monai.networks.nets import resnet18
from torchvision.models import resnet34, ResNet34_Weights, resnet

from model_zoo import MyGroupNorm


class ResNet3DDepth(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=3)

        # --- 2. Strip off the final classifier to get feature vectors ---
        # The final layer is model.fc
        num_features = self.model.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.model.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.model(x)
        preds = self.cls_head(feats)
        return preds


class ResNetDepth2dPatches(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
        sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()
        self.model.load_state_dict(sd, strict=False)
        # --- 2. Strip off the final classifier to get feature vectors ---
        # The final layer is model.fc
        num_features = self.model.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.model.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.model(x)
        preds = self.cls_head(feats)
        return preds
