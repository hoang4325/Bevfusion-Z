

from typing import Any, Dict, List, Optional

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        instrumentation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        self.voxelize_reduce = {}
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce["lidar"] = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce["radar"] = encoders["radar"].get("voxelize_reduce", True)

        self.sensor_order = self._resolve_sensor_order(kwargs.pop("sensor_order", None))
        if fuser is not None:
            fuser_cfg = fuser
            if (
                getattr(fuser, "get", None) is not None
                and fuser.get("type") == "SEFuser"
                and "sensor_order" not in fuser
            ):
                fuser_cfg = dict(fuser)
                fuser_cfg["sensor_order"] = list(self.sensor_order)
            self.fuser = build_fuser(fuser_cfg)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']
        self.instrumentation = instrumentation or {}
        self.instrumentation_enabled = bool(self.instrumentation.get("enabled", False))
        self.instrumentation_interval = max(
            int(self.instrumentation.get("interval", 200)),
            1,
        )
        self._instrument_step = 0
        self._instrument_should_log = False
        self._instrument_voxel_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._instrument_grad_stats: Dict[str, Dict[str, float]] = {}


        self.init_weights()

    def _instrument_active(self) -> bool:
        return self.instrumentation_enabled and self.training

    def _instrument_active_this_step(self) -> bool:
        return self._instrument_active() and self._instrument_should_log

    @staticmethod
    def _feature_stat_tensors(x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.detach().float()
        return {
            "mean_abs": x.abs().mean(),
            "l2": torch.linalg.vector_norm(x),
        }

    def _loggable_feature_stats(
        self, prefix: str, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        stats = {}
        for name, value in self._feature_stat_tensors(x).items():
            stats[f"stats/instrument/{prefix}/{name}"] = value
        return stats

    def _register_grad_probe(self, sensor: str, feature: torch.Tensor) -> None:
        if not self._instrument_active() or not feature.requires_grad:
            return

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.detach().float()
            self._instrument_grad_stats[sensor] = {
                "prev_step_grad_mean_abs": float(grad.abs().mean().item()),
                "prev_step_grad_l2": float(torch.linalg.vector_norm(grad).item()),
            }
            return grad

        feature.register_hook(_hook)

    def _collect_conv_fuser_stats(
        self,
        ordered_sensors: List[str],
        feature_by_sensor: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.fuser is None or self.fuser.__class__.__name__ != "ConvFuser":
            return {}
        if "radar" not in feature_by_sensor or "radar" not in ordered_sensors:
            return {}

        if not isinstance(self.fuser, nn.Sequential) or len(self.fuser) == 0:
            return {}
        conv = self.fuser[0]
        if (
            not isinstance(conv, nn.Conv2d)
            or getattr(conv, "weight", None) is None
            or conv.weight.ndim != 4
        ):
            return {}
        channel_counts = [int(feature_by_sensor[sensor].shape[1]) for sensor in ordered_sensors]
        radar_index = ordered_sensors.index("radar")
        start = sum(channel_counts[:radar_index])
        end = start + channel_counts[radar_index]
        if end > conv.weight.shape[1]:
            return {}
        radar_slice = conv.weight[:, start:end, :, :].detach().float()
        return {
            "stats/instrument/fuser/radar_slice_weight_mean_abs": radar_slice.abs().mean(),
            "stats/instrument/fuser/radar_slice_weight_l2": torch.linalg.vector_norm(radar_slice),
        }

    def _collect_se_fuser_stats(self) -> Dict[str, torch.Tensor]:
        if self.fuser is None or self.fuser.__class__.__name__ != "SEFuser":
            return {}

        stats = {}
        last_attn = getattr(self.fuser, "_last_sensor_attention_mean", None)
        if isinstance(last_attn, dict):
            radar_attn = last_attn.get("radar")
            if radar_attn is not None:
                stats["stats/instrument/fuser/radar_gate_mean"] = radar_attn
        return stats

    def _resolve_sensor_order(self, sensor_order: Optional[List[str]]) -> List[str]:
        available_sensors = list(self.encoders.keys())
        if sensor_order is None:
            return available_sensors
        if not isinstance(sensor_order, (list, tuple)):
            raise TypeError(
                f"sensor_order must be a list/tuple of sensor names, got {type(sensor_order)}"
            )

        sensor_order = list(sensor_order)
        if len(sensor_order) != len(set(sensor_order)):
            raise ValueError(f"sensor_order contains duplicates: {sensor_order}")

        if set(sensor_order) != set(available_sensors):
            raise ValueError(
                "sensor_order must exactly match configured encoders. "
                f"sensor_order={sensor_order}, encoders={available_sensors}"
            )

        return sensor_order

    def _assert_fuser_compatibility(
        self, features: List[torch.Tensor], sensor_order: Optional[List[str]] = None
    ) -> None:
        if self.fuser is None:
            return

        spatial_shapes = {(int(feat.shape[2]), int(feat.shape[3])) for feat in features}
        if len(spatial_shapes) != 1:
            raise ValueError(
                "All encoder features must share the same spatial shape before fusion, "
                f"but got {sorted(list(spatial_shapes))} in sensor order {self.sensor_order}."
            )

        expected_channels = getattr(self.fuser, "in_channels", None)
        if expected_channels is None:
            return
        actual_channels = [int(feat.shape[1]) for feat in features]
        feature_order = list(sensor_order) if sensor_order is not None else list(self.sensor_order)

        if isinstance(expected_channels, dict):
            missing_sensors = [
                sensor for sensor in feature_order if sensor not in expected_channels
            ]
            if missing_sensors:
                raise ValueError(
                    "Fuser in_channels mapping is missing sensors required by the current "
                    f"feature order. missing={missing_sensors}, "
                    f"available={list(expected_channels.keys())}, feature_order={feature_order}"
                )
            expected_channels = [int(expected_channels[sensor]) for sensor in feature_order]
        else:
            if isinstance(expected_channels, int):
                expected_channels = [expected_channels]
            expected_channels = [int(x) for x in expected_channels]

        if len(expected_channels) != len(actual_channels):
            raise ValueError(
                f"Fuser expects {len(expected_channels)} inputs ({expected_channels}), "
                f"but got {len(actual_channels)} features ({actual_channels}) "
                f"in feature_order={feature_order}."
            )

        if expected_channels != actual_channels:
            raise ValueError(
                f"Fuser in_channels mismatch. expected={expected_channels}, actual={actual_channels}, "
                f"feature_order={feature_order}"
            )

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce.get(sensor, True):
                # clamp(min=1) prevents division-by-zero when radar voxels
                # are empty (all points filtered out).  Empty voxels produce
                # a zero sum, so the result is 0.0 — semantically correct.
                safe_sizes = sizes.clamp(min=1).type_as(feats).view(-1, 1)
                feats = feats.sum(dim=1, keepdim=False) / safe_sizes
                feats = feats.contiguous()

        if self._instrument_active_this_step():
            non_empty_voxels = feats.new_tensor(float(coords.shape[0]))
            sensor_stats = {
                "non_empty_voxels": non_empty_voxels,
            }
            middle_encoder = getattr(self.encoders[sensor]["backbone"], "pts_middle_encoder", None)
            output_shape = getattr(middle_encoder, "output_shape", None)
            if output_shape is not None and len(output_shape) >= 2:
                total_cells = max(int(len(points) * output_shape[0] * output_shape[1]), 1)
                sensor_stats["scatter_density"] = non_empty_voxels / total_cells
            self._instrument_voxel_stats[sensor] = sensor_stats

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        feature_by_sensor = {}
        ordered_sensors = []
        auxiliary_losses = {}
        instrumentation_stats = {}
        self._instrument_voxel_stats = {}
        sensor_extract_order = self.sensor_order if self.training else list(reversed(self.sensor_order))
        for sensor in sensor_extract_order:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                if points is None:
                    raise ValueError("LiDAR encoder is enabled but 'points' input is None.")
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                if radar is None:
                    raise ValueError("Radar encoder is enabled but 'radar' input is None.")
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)
            ordered_sensors.append(sensor)
            feature_by_sensor[sensor] = feature
            if sensor in {"radar", "lidar"}:
                self._register_grad_probe(sensor, feature)
            if self._instrument_active_this_step() and sensor in {"radar", "lidar"}:
                instrumentation_stats.update(
                    self._loggable_feature_stats(f"prefusion/{sensor}", feature)
                )
                sensor_voxel_stats = self._instrument_voxel_stats.get(sensor, {})
                for name, value in sensor_voxel_stats.items():
                    instrumentation_stats[f"stats/instrument/{sensor}/{name}"] = value

        if not self.training:
            # avoid OOM
            features = features[::-1]
            ordered_sensors = ordered_sensors[::-1]

        if self.fuser is not None:
            self._assert_fuser_compatibility(features, ordered_sensors)
            if self.fuser.__class__.__name__ == "SEFuser":
                x = self.fuser(features, sensor_order=ordered_sensors)
            else:
                x = self.fuser(features)
            if self._instrument_active_this_step():
                instrumentation_stats.update(
                    self._collect_conv_fuser_stats(ordered_sensors, feature_by_sensor)
                )
                instrumentation_stats.update(self._collect_se_fuser_stats())
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            if self._instrument_active_this_step():
                outputs.update(instrumentation_stats)
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def train_step(self, data, optimizer):
        self._instrument_step += 1
        self._instrument_should_log = (
            self.instrumentation_enabled
            and self._instrument_step % self.instrumentation_interval == 0
        )

        losses = self(**data)
        if self._instrument_should_log and self._instrument_grad_stats:
            reference = next(
                value for value in losses.values() if isinstance(value, torch.Tensor)
            )
            for sensor in ("radar", "lidar"):
                sensor_stats = self._instrument_grad_stats.get(sensor, {})
                for name, value in sensor_stats.items():
                    losses[f"stats/instrument/{sensor}/{name}"] = reference.new_tensor(value)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["metas"]))
        return outputs

