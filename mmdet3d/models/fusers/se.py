from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["SEFuser"]


class SensorSEBlock(nn.Module):
    """Squeeze-and-Excitation block that learns per-channel attention weights
    for a single sensor's BEV features.

    Given input (B, C, H, W), it produces channel-wise attention weights via:
        GlobalAvgPool → FC → ReLU → FC → Sigmoid → scale input
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class SpatialAttentionGate(nn.Module):
    """Learns a per-pixel importance mask from the concatenated multi-sensor
    features, allowing the network to spatially highlight regions where a
    specific sensor is most informative (e.g. radar velocity in dynamic areas,
    camera semantics at distant objects).

    Input:  (B, C_total, H, W)   — concatenated features from all sensors
    Output: (B, N_sensors, H, W) — per-sensor spatial attention maps
    """

    def __init__(self, in_channels: int, num_sensors: int) -> None:
        super().__init__()
        mid = max(in_channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, num_sensors, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N_sensors, H, W) — softmax across sensors so they compete
        return torch.softmax(self.gate(x), dim=1)


@FUSERS.register_module()
class SEFuser(nn.Module):
    """Squeeze-Excitation Fusion module for multi-sensor BEV features.

    Instead of naive channel concatenation (ConvFuser), this module:

    1. Projects each sensor's BEV features to a common channel dimension
       via a learned 1×1 conv — this equalizes representation power so
       sensors with fewer channels (radar=64) aren't drowned out by
       sensors with more channels (lidar=256).

    2. Applies per-sensor Squeeze-and-Excitation (SE) channel attention
       to let each branch recalibrate its own feature channels.

    3. Computes a spatial attention gate on the concatenated projected
       features to produce per-sensor, per-pixel importance weights.
       This allows the network to learn WHERE each sensor matters most
       (e.g. radar is important in dynamic regions, camera at distance).

    4. Fuses via weighted sum + residual refinement conv.

    Args:
        in_channels (list[int]): Input channels for each sensor
            e.g. [80, 256, 64] for [camera, lidar, radar].
        out_channels (int): Output channels after fusion.
        se_reduction (int): SE block channel reduction ratio.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_sensors = len(in_channels)

        # Step 1: Per-sensor projection to common channel dim
        self.projections = nn.ModuleList()
        for ch in in_channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

        # Step 2: Per-sensor channel attention (SE)
        self.se_blocks = nn.ModuleList(
            [SensorSEBlock(out_channels, se_reduction) for _ in range(num_sensors)]
        )

        # Step 3: Spatial attention gate
        self.spatial_gate = SpatialAttentionGate(
            in_channels=out_channels * num_sensors,
            num_sensors=num_sensors,
        )

        # Step 4: Residual refinement after weighted sum
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.in_channels)

        # Project + SE attention per sensor
        projected = []
        for i, (proj, se) in enumerate(zip(self.projections, self.se_blocks)):
            feat = proj(inputs[i])   # (B, out_ch, H, W)
            feat = se(feat)          # channel attention
            projected.append(feat)

        # Spatial attention gate from concatenated features
        concat = torch.cat(projected, dim=1)           # (B, out_ch*N, H, W)
        attn = self.spatial_gate(concat)                # (B, N, H, W)

        # Weighted sum across sensors
        fused = torch.zeros_like(projected[0])
        for i, feat in enumerate(projected):
            fused = fused + attn[:, i : i + 1, :, :] * feat

        # Residual refinement
        out = self.relu(fused + self.refine(fused))

        return out
