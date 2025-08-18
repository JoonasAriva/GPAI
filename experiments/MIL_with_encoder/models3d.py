import torch.nn as nn
from monai.networks.nets import resnet


class ResNet3DDepth(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet.resnet18(
            spatial_dims=3,  # 3D convolutions
            n_input_channels=1,  # CT scan has 1 channel
            num_classes=2  # temporarily set for classification
        )

        # --- 2. Strip off the final classifier to get feature vectors ---
        # The final layer is model.fc
        num_features = self.model.fc.in_features

        self.cls_head = nn.Linear(num_features, 3)
        self.model.fc = nn.Identity()  # now forward pass outputs feature vectors

    def forward(self, x):
        feats = self.model(x)
        preds = self.cls_head(feats)
        return preds
