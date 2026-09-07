"""Target denormalization and bounding-box reconstruction.

The model predicts decomposed, z-score-normalized targets rather than raw
coordinates, so turning its output back into a bbox is pure arithmetic. That
arithmetic lives here, in plain Python with no torch or numpy import, for two
reasons:

  1. It is the inverse of the decomposition done at training time
     (shadow_detection.data), so it needs to be testable against that side.
  2. The lightweight backend test environment has no torch, and a test that
     re-implements this maths instead of importing it would keep passing if
     the served code broke.
"""

from __future__ import annotations

from collections.abc import Sequence

# The training resolution. Predictions are produced in this coordinate space
# and then scaled to whatever the caller uploaded.
IMG_W = 720
IMG_H = 480

# Floors applied to the regressed box dimensions. A degenerate or negative
# prediction would otherwise produce an inverted rectangle.
MIN_BBOX_WIDTH = 10.0
MIN_BBOX_HEIGHT = 50.0

# Regression head output order.
REGRESSION_KEYS = ("distance_from_edge", "bbox_width", "bbox_height", "y_center")

SIDE_LEFT = 0
SIDE_RIGHT = 1


def denormalize(value: float, stats: dict[str, float]) -> float:
    """Invert the z-score normalization applied to a regression target.

    `stats` is the {"mean", "std"} entry for one target. The 1e-8 guard mirrors
    shadow_detection.data.normalize_target exactly, so the two cancel even when
    a target has zero variance.
    """
    return value * (stats["std"] + 1e-8) + stats["mean"]


def reconstruct_bbox(
    side: int,
    reg: Sequence[float],
    target_stats: dict[str, dict[str, float]],
    image_width: int = IMG_W,
    image_height: int = IMG_H,
) -> tuple[float, float, float, float]:
    """Rebuild (xmin, ymin, xmax, ymax) from the decomposed prediction.

    `reg` holds the four normalized regression outputs in REGRESSION_KEYS
    order. `side` says which frame edge the pedestrian is past, which is what
    makes the sign of the x offsets unambiguous.

    The returned box is deliberately allowed to fall outside the frame - that
    is the whole point of the task - and is scaled from the 720x480 training
    space to the supplied image dimensions.
    """
    if len(reg) != len(REGRESSION_KEYS):
        raise ValueError(f"Expected {len(REGRESSION_KEYS)} regression outputs, got {len(reg)}")

    dist = max(denormalize(float(reg[0]), target_stats["distance_from_edge"]), 0.0)
    width = max(denormalize(float(reg[1]), target_stats["bbox_width"]), MIN_BBOX_WIDTH)
    height = max(denormalize(float(reg[2]), target_stats["bbox_height"]), MIN_BBOX_HEIGHT)
    y_center = denormalize(float(reg[3]), target_stats["y_center"])

    if side == SIDE_LEFT:
        xmin = -dist
        xmax = xmin + width
    else:
        xmax = IMG_W + dist
        xmin = xmax - width
    ymin = y_center - height / 2
    ymax = y_center + height / 2

    sx = image_width / IMG_W
    sy = image_height / IMG_H
    return xmin * sx, ymin * sy, xmax * sx, ymax * sy
