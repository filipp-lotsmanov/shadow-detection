"""Target decomposition and its inverse in the deployed backend.

The training pipeline decomposes a bbox into (side, distance_from_edge, width,
height, y_center) and z-score normalizes the four regression targets. The
backend denormalizes and reconstructs. Those two halves live in different
packages and must be exact inverses, so these tests drive the real functions
from both sides rather than restating the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadow_detection.data import compute_target_stats, decompose_bbox, normalize_target

IMG_W = 720

# Realistic stats, in the shape compute_target_stats produces.
STATS = {
    "distance_from_edge": {"mean": 208.58, "std": 45.16},
    "bbox_width": {"mean": 80.85, "std": 29.86},
    "bbox_height": {"mean": 172.90, "std": 34.67},
    "y_center": {"mean": 309.33, "std": 14.27},
}

# (xmin, ymin, xmax, ymax) - off-frame left and right, drawn from the real
# sample manifest plus two synthetic extremes.
BOXES = [
    (-202.77, 223.64, -155.28, 369.26),
    (-306.85, 223.57, -169.84, 438.95),
    (872.49, 223.14, 989.09, 423.59),
    (797.40, 229.46, 874.05, 350.01),
    (-150.0, 200.0, -100.0, 350.0),
    (760.0, 210.0, 830.0, 380.0),
]


def _round_trip(box, backend_geometry, stats=None):
    """Decompose with the training code, reconstruct with the backend's."""
    stats = stats or STATS
    xmin, ymin, xmax, ymax = box
    targets = decompose_bbox(xmin, ymin, xmax, ymax, IMG_W)
    reg = tuple(
        normalize_target(targets[key], key, stats) for key in backend_geometry.REGRESSION_KEYS
    )
    return backend_geometry.reconstruct_bbox(int(targets["side"]), reg, stats)


@pytest.mark.parametrize("box", BOXES)
def test_decompose_then_reconstruct_recovers_the_box(box, backend_geometry):
    """The invariant the whole decomposed-target design rests on."""
    assert np.allclose(_round_trip(box, backend_geometry), box, atol=1e-6)


@pytest.mark.parametrize("box", BOXES)
def test_side_matches_which_edge_the_box_is_past(box, backend_geometry):
    xmin, ymin, xmax, ymax = box
    side = decompose_bbox(xmin, ymin, xmax, ymax, IMG_W)["side"]
    assert side == (backend_geometry.SIDE_LEFT if xmin < 0 else backend_geometry.SIDE_RIGHT)


def test_distance_from_edge_is_never_negative(backend_geometry):
    for box in BOXES:
        targets = decompose_bbox(*box, IMG_W)
        assert targets["distance_from_edge"] >= 0


def test_round_trip_holds_for_stats_derived_from_the_boxes(backend_geometry):
    """Uses compute_target_stats on the boxes themselves, so the normalization
    is the one training would actually have used."""
    samples = [decompose_bbox(*box, IMG_W) for box in BOXES]
    stats = compute_target_stats(samples)
    for box in BOXES:
        assert np.allclose(_round_trip(box, backend_geometry, stats), box, atol=1e-6)


def test_normalize_and_denormalize_cancel(backend_geometry):
    """Both sides add the same 1e-8 guard, so they cancel exactly."""
    norm = normalize_target(175.0, "distance_from_edge", STATS)
    back = backend_geometry.denormalize(norm, STATS["distance_from_edge"])
    assert np.isclose(back, 175.0, atol=1e-6)


def test_zero_variance_target_collapses_to_its_mean(backend_geometry):
    """A constant target gives std 0. Normalization must not divide by zero,
    and denormalization must hand back the mean."""
    stats = {"y_center": {"mean": 240.0, "std": 0.0}}
    norm = normalize_target(240.0, "y_center", stats)
    assert np.isfinite(norm)
    assert np.isclose(norm, 0.0)
    assert np.isclose(backend_geometry.denormalize(norm, stats["y_center"]), 240.0)
