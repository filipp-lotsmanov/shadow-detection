"""Target decomposition and normalization invariants.

The model predicts decomposed targets (side, distance_from_edge, width, height,
y_center) rather than raw coordinates. Reconstruction at inference time must
invert the decomposition done at data-loading time, and the z-score
normalization must invert the denormalization used in the backend.
"""

from __future__ import annotations

import numpy as np
from shadow_detection.data import normalize_target

IMG_W = 720


def _reconstruct_x_bounds(side: int, distance_from_edge: float, width: float):
    """Mirror of the bbox reconstruction in data._verify_decomposition and
    inference.ShadowDetector.predict."""
    if side == 0:
        xmin = -distance_from_edge
        xmax = xmin + width
    else:
        xmax = IMG_W + distance_from_edge
        xmin = xmax - width
    return xmin, xmax


def test_decomposition_roundtrip_left():
    xmin, xmax = -150.0, -100.0
    side, distance_from_edge, width = 0, abs(xmin), xmax - xmin
    rx_min, rx_max = _reconstruct_x_bounds(side, distance_from_edge, width)
    assert np.isclose(rx_min, xmin)
    assert np.isclose(rx_max, xmax)


def test_decomposition_roundtrip_right():
    xmin, xmax = 760.0, 830.0
    side, distance_from_edge, width = 1, xmax - IMG_W, xmax - xmin
    rx_min, rx_max = _reconstruct_x_bounds(side, distance_from_edge, width)
    assert np.isclose(rx_min, xmin)
    assert np.isclose(rx_max, xmax)


def test_normalize_denormalize_inverse():
    """normalize_target divides by (std + 1e-8); the backend denormalize formula
    multiplies by the same (std + 1e-8) and re-adds the mean. The two cancel."""
    stats = {"distance_from_edge": {"mean": 120.0, "std": 40.0}}
    value = 175.0

    norm = normalize_target(value, "distance_from_edge", stats)
    s = stats["distance_from_edge"]
    denorm = norm * (s["std"] + 1e-8) + s["mean"]

    assert np.isclose(denorm, value, atol=1e-3)


def test_normalize_handles_zero_std():
    """A constant target collapses to zero after normalization without dividing
    by zero (the 1e-8 guard)."""
    stats = {"y_center": {"mean": 240.0, "std": 0.0}}
    norm = normalize_target(240.0, "y_center", stats)
    assert np.isfinite(norm)
    assert np.isclose(norm, 0.0)
