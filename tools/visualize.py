# Modified by UMONS-Numediart, Ratha SIV in 2026.

import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDistributedDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack import distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def extract_points_array(data, key):
    if key not in data:
        return None

    value = data[key]
    if isinstance(value, DataContainer):
        value = value.data[0]
        if isinstance(value, list):
            value = value[0]

    if hasattr(value, "tensor"):
        value = value.tensor

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif value is not None:
        value = np.asarray(value)

    if value is None:
        return None

    if value.ndim == 1:
        value = value.reshape(1, -1)
    elif value.ndim == 2 and value.shape[1] < 2 and value.shape[0] >= 2:
        value = value.T

    if value.ndim != 2 or value.shape[1] < 2:
        return None

    return value


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--draw-heading", dest="draw_heading", action="store_true")
    parser.add_argument("--no-draw-heading", dest="draw_heading", action="store_false")
    parser.add_argument("--draw-velocity", dest="draw_velocity", action="store_true")
    parser.add_argument("--no-draw-velocity", dest="draw_velocity", action="store_false")
    parser.add_argument("--vel-scale", type=float, default=2.0)
    parser.add_argument("--min-speed", type=float, default=0.2)
    parser.set_defaults(draw_heading=True, draw_velocity=True)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])
        scores = None

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool_)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
                draw_heading=args.draw_heading,
                draw_velocity=args.draw_velocity,
                vel_scale=args.vel_scale,
                min_speed=args.min_speed,
            )

        radar = extract_points_array(data, "radar")
        if radar is not None:
            visualize_lidar(
                os.path.join(args.out_dir, "radar", f"{name}.png"),
                radar,
                bboxes=bboxes,
                labels=labels,
                scores=scores,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
                draw_heading=args.draw_heading,
                draw_velocity=args.draw_velocity,
                vel_scale=args.vel_scale,
                min_speed=args.min_speed,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
