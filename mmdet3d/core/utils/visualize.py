# Modified by UMONS-Numediart, Ratha SIV in 2026.

import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

HEADING_ARROW_COLOR = (0, 229, 255)
VELOCITY_ARROW_COLOR = (255, 99, 71)
BEV_TEXT_BOX = {
    "facecolor": (0.0, 0.0, 0.0, 0.65),
    "edgecolor": "none",
    "boxstyle": "round,pad=0.25",
}


def _mpl_color(color: Tuple[int, int, int]) -> np.ndarray:
    return np.asarray(color, dtype=np.float32) / 255.0


def _get_box_name(
    labels: Optional[np.ndarray],
    classes: Optional[List[str]],
    index: int,
) -> str:
    if labels is None:
        return "box"
    label = int(labels[index])
    if classes is None or label < 0 or label >= len(classes):
        return str(label)
    return classes[label]


def _get_box_velocity_xy(
    bboxes: LiDARInstance3DBoxes,
) -> Optional[np.ndarray]:
    tensor = bboxes.tensor.detach().cpu().numpy()
    if tensor.shape[1] < 9:
        return None
    return tensor[:, 7:9]


def _draw_bev_arrow(
    ax,
    start: np.ndarray,
    vector: np.ndarray,
    *,
    color: Tuple[int, int, int],
    linewidth: float,
    mutation_scale: float,
    zorder: int,
) -> None:
    if np.linalg.norm(vector) <= 1e-6:
        return

    end = start + vector
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=_mpl_color(color),
            lw=linewidth,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0,
            alpha=0.95,
        ),
        zorder=zorder,
    )


def _draw_bev_box_overlays(
    ax,
    *,
    bboxes: LiDARInstance3DBoxes,
    labels: Optional[np.ndarray],
    classes: Optional[List[str]],
    scores: Optional[np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    draw_heading: bool,
    draw_velocity: bool,
    vel_scale: float,
    min_speed: float,
) -> None:
    if len(bboxes) == 0:
        return

    centers = bboxes.center[:, :2].detach().cpu().numpy()
    corners = bboxes.corners.detach().cpu().numpy()
    # Heading uses the box yaw: front face center is derived from rotated corners.
    front_centers = corners[:, [4, 7], :2].mean(axis=1)
    # Velocity uses the decoded box motion fields (vx, vy) in LiDAR/BEV coordinates.
    velocities = _get_box_velocity_xy(bboxes)

    scene_span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    heading_linewidth = max(2.5, scene_span * 0.04)
    velocity_linewidth = max(2.5, scene_span * 0.04)
    arrow_head = max(18.0, scene_span * 0.3)
    text_offset = max(0.6, scene_span * 0.012)
    heading_min_len = max(1.0, scene_span * 0.012)
    heading_max_len = max(3.0, scene_span * 0.06)
    velocity_min_len = max(1.0, scene_span * 0.015)
    velocity_max_len = max(5.0, scene_span * 0.08)
    font_size = float(np.clip(scene_span * 0.16, 9, 16))

    for index in range(len(bboxes)):
        name = _get_box_name(labels, classes, index)
        center = centers[index]
        front_center = front_centers[index]
        heading_vec = front_center - center
        heading_norm = np.linalg.norm(heading_vec)
        speed = None
        is_static = False

        if draw_heading and heading_norm > 1e-6:
            heading_len = np.clip(heading_norm * 1.15, heading_min_len, heading_max_len)
            _draw_bev_arrow(
                ax,
                center,
                heading_vec / heading_norm * heading_len,
                color=HEADING_ARROW_COLOR,
                linewidth=heading_linewidth,
                mutation_scale=arrow_head,
                zorder=7,
            )

        if velocities is not None:
            velocity_vec = velocities[index]
            speed = float(np.linalg.norm(velocity_vec))
            is_static = speed < min_speed
            if draw_velocity and speed >= min_speed and speed > 1e-6:
                velocity_len = np.clip(speed * vel_scale, velocity_min_len, velocity_max_len)
                _draw_bev_arrow(
                    ax,
                    center,
                    velocity_vec / speed * velocity_len,
                    color=VELOCITY_ARROW_COLOR,
                    linewidth=velocity_linewidth,
                    mutation_scale=arrow_head,
                    zorder=8,
                )

        score_text = ""
        if scores is not None:
            score_text = f" {float(scores[index]):.2f}"

        if speed is None:
            speed_text = "speed n/a"
        elif is_static:
            speed_text = f"static {speed:.1f} m/s"
        else:
            speed_text = f"{speed:.1f} m/s"

        label_text = f"{name}{score_text}\n{speed_text}"
        corner_xy = corners[index, [0, 3, 7, 4], :2]
        text_x = float(np.min(corner_xy[:, 0]) + text_offset)
        text_y = float(np.max(corner_xy[:, 1]) + text_offset)
        ax.text(
            text_x,
            text_y,
            label_text,
            color="white",
            fontsize=font_size,
            ha="left",
            va="bottom",
            bbox=BEV_TEXT_BOX,
            zorder=9,
        )

    legend_handles = []
    if draw_heading:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_mpl_color(HEADING_ARROW_COLOR),
                lw=3,
                marker=">",
                markersize=8,
                label="heading = bbox yaw",
            )
        )
    if draw_velocity:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_mpl_color(VELOCITY_ARROW_COLOR),
                lw=3,
                marker=">",
                markersize=8,
                label="velocity = box vx, vy",
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="white",
                lw=0,
                label=f"static < {min_speed:g} m/s",
            )
        )

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            loc="upper left",
            facecolor="black",
            edgecolor="white",
            framealpha=0.8,
            fontsize=font_size,
        )
        for text in legend.get_texts():
            text.set_color("white")


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int64),
                    coords[index, end].astype(np.int64),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
    draw_heading: bool = True,
    draw_velocity: bool = True,
    vel_scale: float = 2.0,
    min_speed: float = 0.2,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = _get_box_name(labels, classes, index)
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )
        _draw_bev_box_overlays(
            ax,
            bboxes=bboxes,
            labels=labels,
            classes=classes,
            scores=scores,
            xlim=xlim,
            ylim=ylim,
            draw_heading=draw_heading,
            draw_velocity=draw_velocity,
            vel_scale=vel_scale,
            min_speed=min_speed,
        )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool_, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
