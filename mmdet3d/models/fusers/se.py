from typing import Dict, List, Optional, Union

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
    specific sensor is most informative.

    Input:  (B, C_total, H, W)   — concatenated features from all sensors
    Output: (B, N_sensors, H, W) — per-sensor spatial attention maps

    Uses kernel_size=3 (safe for BEV maps ≥ 3×3).  A runtime assertion fires
    loudly if the map is smaller, rather than producing a silent wrong result.
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
        # Task 8: runtime shape guard — fail loudly on tiny BEV maps
        assert x.shape[2] >= 3 and x.shape[3] >= 3, (
            f"SpatialAttentionGate requires BEV map ≥ 3×3, "
            f"but got {x.shape[2]}×{x.shape[3]}. "
            "Reduce conv stride or increase BEV resolution."
        )
        # (B, N_sensors, H, W) — softmax across sensors so they compete
        return torch.softmax(self.gate(x), dim=1)


@FUSERS.register_module()
class SEFuser(nn.Module):
    """Squeeze-Excitation Fusion module for multi-sensor BEV features.

    Task 7 — Dict-based sensor routing
    -----------------------------------
    ``in_channels`` can be either:

    * ``List[int]``  — backward-compatible; interpreted in ``sensor_order``
      (old configs that pass ``[80, 256, 64]`` continue to work).
    * ``Dict[str, int]``  — preferred; explicitly maps sensor name to channel
      count, e.g. ``{"camera": 80, "lidar": 256, "radar": 64}``.

    When ``forward()`` receives a plain ``List[Tensor]`` it expects the
    caller to also supply ``sensor_order`` so the mapping is unambiguous.
    When it receives a ``Dict[str, Tensor]`` the sensor names are used
    directly.

    Runtime channel assertions guard against silent mis-routing.

    Args:
        in_channels: Per-sensor input channels (List or Dict).
        out_channels: Output channels after fusion.
        se_reduction: SE block channel reduction ratio.
        sensor_order: Required when ``in_channels`` is a List and ``forward``
            receives a List — specifies the name of each list position.
    """

    def __init__(
        self,
        in_channels: Union[List[int], Dict[str, int]],
        out_channels: int,
        se_reduction: int = 4,
        sensor_order: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        # --- Normalise in_channels to an ordered dict ---
        if isinstance(in_channels, (list, tuple)):
            if sensor_order is not None:
                assert len(sensor_order) == len(in_channels), (
                    f"sensor_order length {len(sensor_order)} must match "
                    f"in_channels length {len(in_channels)}"
                )
                names = list(sensor_order)
            else:
                # Fallback: auto-assign generic names so forward() still works
                # when called with a List.  Explicit sensor names are preferred.
                names = [f"sensor_{i}" for i in range(len(in_channels))]
            self.in_channels: Dict[str, int] = dict(zip(names, in_channels))
        else:
            self.in_channels = dict(in_channels)

        self.sensor_names: List[str] = list(self.in_channels.keys())
        num_sensors = len(self.sensor_names)

        # Step 1: Per-sensor projection to common channel dim (nn.ModuleDict
        #         keyed by sensor name — wrong routing is now impossible)
        self.projections = nn.ModuleDict()
        for name, ch in self.in_channels.items():
            self.projections[name] = nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

        # Step 2: Per-sensor channel attention (SE)
        self.se_blocks = nn.ModuleDict({
            name: SensorSEBlock(out_channels, se_reduction)
            for name in self.sensor_names
        })

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
        self._last_sensor_attention_mean: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_sensors(self) -> int:
        return len(self.sensor_names)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
        sensor_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Fuse multi-sensor BEV features.

        Args:
            inputs: Either a ``List[Tensor]`` (in the order matching
                ``sensor_order`` / ``self.sensor_names``) or a
                ``Dict[str, Tensor]`` keyed by sensor name.
            sensor_order: When ``inputs`` is a List, use this to look up
                sensor names.  Falls back to ``self.sensor_names``.

        Returns:
            Fused BEV feature map of shape (B, out_channels, H, W).
        """
        # --- Normalise inputs to dict ---
        if isinstance(inputs, (list, tuple)):
            names = sensor_order if sensor_order is not None else self.sensor_names
            assert len(inputs) == len(names), (
                f"SEFuser received {len(inputs)} feature tensors but "
                f"sensor_names has {len(names)} entries: {names}"
            )
            feat_dict: Dict[str, torch.Tensor] = dict(zip(names, inputs))
        else:
            feat_dict = dict(inputs)

        # --- Runtime channel assertions (Task 7) ---
        for name in self.sensor_names:
            assert name in feat_dict, (
                f"SEFuser expected sensor '{name}' but it was not in the "
                f"input dict.  Available keys: {list(feat_dict.keys())}"
            )
            actual_ch = feat_dict[name].shape[1]
            expected_ch = self.in_channels[name]
            assert actual_ch == expected_ch, (
                f"SEFuser sensor '{name}': expected {expected_ch} channels, "
                f"got {actual_ch}.  Check encoder out_channels vs fuser in_channels."
            )

        # --- Project + SE attention per sensor ---
        projected = []
        for name in self.sensor_names:
            feat = self.projections[name](feat_dict[name])   # (B, out_ch, H, W)
            feat = self.se_blocks[name](feat)                 # channel attention
            projected.append(feat)

        # --- Spatial attention gate from concatenated features ---
        concat = torch.cat(projected, dim=1)            # (B, out_ch*N, H, W)
        attn = self.spatial_gate(concat)                 # (B, N, H, W)
        self._last_sensor_attention_mean = {
            name: attn[:, i : i + 1, :, :].detach().float().mean()
            for i, name in enumerate(self.sensor_names)
        }

        # --- Weighted sum across sensors ---
        fused = torch.zeros_like(projected[0])
        for i, feat in enumerate(projected):
            fused = fused + attn[:, i : i + 1, :, :] * feat

        # --- Residual refinement ---
        out = self.relu(fused + self.refine(fused))

        return out
