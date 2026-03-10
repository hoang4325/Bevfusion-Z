# Added by UMONS-Numediart, 2026.

import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn

from mmdet.models import BACKBONES

__all__ = ["RadarBEVBackbone"]


@BACKBONES.register_module()
class RadarBEVBackbone(BaseModule):
    """Lightweight 2D convolutional backbone applied to the radar BEV
    pseudo-image **after** PointPillarsScatter.

    Motivation
    ----------
    Raw radar scatter produces an extremely sparse BEV map (>99 % zero for
    typical NuScenes frames with ~200 radar points on a 180×180 grid).
    A few conv layers with small stride allow the network to:

    * **Densify** the feature map — information propagates from occupied
      pillars to their neighbours via the receptive field.
    * **Extract spatial patterns** — e.g. clusters of moving objects,
      guardrails, static clutter — before the features are fused with
      the much denser camera and LiDAR BEV maps.

    The architecture mirrors :class:`SECOND` but is intentionally kept
    shallow (2 stages, stride-1) so that the spatial resolution is
    preserved and matches the camera / LiDAR BEV grids exactly.

    Args:
        in_channels (int): Number of input channels from the scatter
            (= RadarFeatureNet output channels, typically 64).
        mid_channels (int): Hidden feature dimension.
        out_channels (int): Output channels fed to the fuser.
        num_blocks (int): Number of residual-style conv blocks per stage.
        norm_cfg (dict): Normalization config.
        conv_cfg (dict): Convolution config.
    """

    def __init__(
        self,
        in_channels: int = 64,
        mid_channels: int = 128,
        out_channels: int = 64,
        num_blocks: int = 3,
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        # Stage 1: expand channels, stride=1 (preserve spatial res)
        layers_1 = [
            build_conv_layer(conv_cfg, in_channels, mid_channels, 3, padding=1),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers_1 += [
                build_conv_layer(conv_cfg, mid_channels, mid_channels, 3, padding=1),
                build_norm_layer(norm_cfg, mid_channels)[1],
                nn.ReLU(inplace=True),
            ]
        self.stage1 = nn.Sequential(*layers_1)

        # Stage 2: project back to out_channels, stride=1
        layers_2 = [
            build_conv_layer(conv_cfg, mid_channels, out_channels, 3, padding=1),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers_2 += [
                build_conv_layer(conv_cfg, out_channels, out_channels, 3, padding=1),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
            ]
        self.stage2 = nn.Sequential(*layers_2)

        # Learnable residual gate: controls how much the processed features
        # add on top of the raw scatter.  Initialized near zero so training
        # starts from the raw-scatter baseline and gradually incorporates
        # the refined features.
        self.gate = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        if init_cfg is None:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) — sparse radar BEV from scatter.

        Returns:
            (B, out_channels, H, W) — densified radar BEV features.
        """
        identity = x
        x = self.stage1(x)
        x = self.stage2(x)

        # Gated residual: if in_channels == out_channels, add skip
        if identity.shape[1] == x.shape[1]:
            x = identity + self.gate * x

        return x
