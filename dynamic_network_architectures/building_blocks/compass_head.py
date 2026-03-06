import torch
from torch import nn
from typing import Type

from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import get_matching_pool_op


class COMPASSCoordinateHead(nn.Module):
    """
    COMPASS coordinate prediction head.

    Takes the bottleneck feature map from an encoder and predicts 3D (D, H, W)
    coordinates of the patch center via global average pooling + linear layers.

    This head does NOT use skip connections — it only receives the last encoder
    stage output.

    Input: (B, C, *spatial) from the last encoder stage
    Output: (B, coord_dim) predicted coordinates (default coord_dim=3 for D, H, W)
    """

    def __init__(self,
                 in_channels: int,
                 conv_op: Type[_ConvNd],
                 hidden_dim: int = 256,
                 coord_dim: int = 3):
        """
        Args:
            in_channels: Number of channels from the encoder bottleneck.
            conv_op: Convolution op class (Conv2d or Conv3d) — used to select
                the matching adaptive average pooling op.
            hidden_dim: Hidden dimension of the MLP.
            coord_dim: Number of output coordinates (default 3 for D, H, W).
        """
        super().__init__()
        pool_op = get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type='avg')
        self.pool = pool_op(1)
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, coord_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder bottleneck features of shape (B, C, *spatial).

        Returns:
            Predicted coordinates of shape (B, coord_dim).
        """
        x = self.pool(x)       # (B, C, 1, 1) or (B, C, 1, 1, 1)
        x = x.flatten(1)       # (B, C)
        return self.head(x)    # (B, coord_dim)
