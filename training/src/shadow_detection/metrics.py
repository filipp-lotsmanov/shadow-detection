"""Bounding-box evaluation metrics: pixel deviation, IoU, GIoU."""

from __future__ import annotations

from collections.abc import Mapping


def compute_bbox_deviation(pred: Mapping[str, float], gt: Mapping[str, float]) -> float:
    """Mean absolute deviation across the 4 bbox coords. This is the competition metric."""
    keys = ("xmin", "ymin", "xmax", "ymax")
    return sum(abs(pred[k] - gt[k]) for k in keys) / 4.0


def _box_components(
    pred: Mapping[str, float], gt: Mapping[str, float]
) -> tuple[float, float, float]:
    """Return (intersection_area, pred_area, gt_area)."""
    x1 = max(pred["xmin"], gt["xmin"])
    y1 = max(pred["ymin"], gt["ymin"])
    x2 = min(pred["xmax"], gt["xmax"])
    y2 = min(pred["ymax"], gt["ymax"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    pred_area = (pred["xmax"] - pred["xmin"]) * (pred["ymax"] - pred["ymin"])
    gt_area = (gt["xmax"] - gt["xmin"]) * (gt["ymax"] - gt["ymin"])
    return inter, pred_area, gt_area


def compute_iou(pred: Mapping[str, float], gt: Mapping[str, float]) -> float:
    inter, pred_area, gt_area = _box_components(pred, gt)
    union = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0


def compute_giou(pred: Mapping[str, float], gt: Mapping[str, float]) -> float:
    """Generalized IoU - extends IoU to non-overlapping boxes.

    Range: [-1, 1]. When boxes don't overlap, GIoU penalizes by the gap relative
    to the smallest enclosing box, so it has a usable gradient unlike plain IoU.
    """
    inter, pred_area, gt_area = _box_components(pred, gt)
    union = pred_area + gt_area - inter
    iou = inter / union if union > 0 else 0.0

    enc_x1 = min(pred["xmin"], gt["xmin"])
    enc_y1 = min(pred["ymin"], gt["ymin"])
    enc_x2 = max(pred["xmax"], gt["xmax"])
    enc_y2 = max(pred["ymax"], gt["ymax"])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    if enc_area > 0:
        return iou - (enc_area - union) / enc_area
    return iou
