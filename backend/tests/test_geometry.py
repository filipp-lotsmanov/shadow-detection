"""Bounding-box reconstruction, tested against the code the server runs.

app.geometry is the inverse of the target decomposition in
shadow_detection.data. These tests import it directly rather than restating
the arithmetic, so a change to the served reconstruction breaks them.
"""

from __future__ import annotations

import pytest

from app.geometry import (
    IMG_H,
    IMG_W,
    MIN_BBOX_HEIGHT,
    MIN_BBOX_WIDTH,
    SIDE_LEFT,
    SIDE_RIGHT,
    denormalize,
    reconstruct_bbox,
)

# Roughly the real target_stats.json shipped with the model.
STATS = {
    "distance_from_edge": {"mean": 208.58, "std": 45.16},
    "bbox_width": {"mean": 80.85, "std": 29.86},
    "bbox_height": {"mean": 172.90, "std": 34.67},
    "y_center": {"mean": 309.33, "std": 14.27},
}


def _normalize(value: float, key: str) -> float:
    """The training-side transform, so tests can specify targets in pixels."""
    s = STATS[key]
    return (value - s["mean"]) / (s["std"] + 1e-8)


def _reg(dist: float, width: float, height: float, y_center: float):
    return (
        _normalize(dist, "distance_from_edge"),
        _normalize(width, "bbox_width"),
        _normalize(height, "bbox_height"),
        _normalize(y_center, "y_center"),
    )


def test_denormalize_inverts_normalize():
    assert denormalize(_normalize(175.0, "distance_from_edge"), STATS["distance_from_edge"]) == (
        pytest.approx(175.0)
    )


def test_denormalize_survives_zero_variance():
    """A constant target has std 0; the 1e-8 guard must keep this finite and
    return the mean rather than dividing by zero on the training side."""
    stats = {"mean": 240.0, "std": 0.0}
    assert denormalize(0.0, stats) == pytest.approx(240.0)


def test_left_side_box_is_off_frame_to_the_left():
    xmin, ymin, xmax, ymax = reconstruct_bbox(SIDE_LEFT, _reg(150.0, 50.0, 146.0, 296.0), STATS)
    assert xmin == pytest.approx(-150.0)
    assert xmax == pytest.approx(-100.0)
    assert ymin == pytest.approx(223.0)
    assert ymax == pytest.approx(369.0)
    assert xmax < 0


def test_right_side_box_is_off_frame_to_the_right():
    xmin, xmax = reconstruct_bbox(SIDE_RIGHT, _reg(110.0, 70.0, 200.0, 310.0), STATS)[0::2]
    assert xmax == pytest.approx(IMG_W + 110.0)
    assert xmin == pytest.approx(IMG_W + 40.0)
    assert xmin > IMG_W


def test_width_and_height_round_trip():
    dist, width, height, y_center = 200.0, 95.0, 180.0, 300.0
    xmin, ymin, xmax, ymax = reconstruct_bbox(SIDE_LEFT, _reg(dist, width, height, y_center), STATS)
    assert xmax - xmin == pytest.approx(width)
    assert ymax - ymin == pytest.approx(height)
    assert (ymin + ymax) / 2 == pytest.approx(y_center)


def test_negative_distance_is_clamped_to_the_frame_edge():
    """The regressor can predict a negative distance; the box must not cross
    to the wrong side of the edge."""
    xmin, _, xmax, _ = reconstruct_bbox(SIDE_LEFT, _reg(-50.0, 60.0, 150.0, 300.0), STATS)
    assert xmin == pytest.approx(0.0)
    assert xmax == pytest.approx(60.0)


def test_degenerate_dimensions_are_floored():
    xmin, ymin, xmax, ymax = reconstruct_bbox(SIDE_LEFT, _reg(100.0, -5.0, 0.0, 300.0), STATS)
    assert xmax - xmin == pytest.approx(MIN_BBOX_WIDTH)
    assert ymax - ymin == pytest.approx(MIN_BBOX_HEIGHT)


def test_box_scales_to_the_uploaded_image():
    reg = _reg(150.0, 50.0, 146.0, 296.0)
    base = reconstruct_bbox(SIDE_LEFT, reg, STATS)
    scaled = reconstruct_bbox(SIDE_LEFT, reg, STATS, image_width=IMG_W * 2, image_height=IMG_H * 3)
    assert scaled[0] == pytest.approx(base[0] * 2)
    assert scaled[2] == pytest.approx(base[2] * 2)
    assert scaled[1] == pytest.approx(base[1] * 3)
    assert scaled[3] == pytest.approx(base[3] * 3)


def test_native_dimensions_are_a_no_op():
    reg = _reg(150.0, 50.0, 146.0, 296.0)
    assert reconstruct_bbox(SIDE_LEFT, reg, STATS) == reconstruct_bbox(
        SIDE_LEFT, reg, STATS, image_width=IMG_W, image_height=IMG_H
    )


def test_wrong_number_of_regression_outputs_is_rejected():
    with pytest.raises(ValueError):
        reconstruct_bbox(SIDE_LEFT, (0.0, 0.0, 0.0), STATS)
