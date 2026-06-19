"""Sanity tests for the geometric feature extractor and its flip transform."""

from __future__ import annotations

import numpy as np
from shadow_detection.features import extract_geometric_features, flip_geometric_features


def test_feature_shape_and_dtype(shadow_image):
    f = extract_geometric_features(shadow_image)
    assert f.shape == (19,)
    assert f.dtype == np.float32


def test_features_finite_on_degenerate_inputs():
    """An all-black image and a uniform mid-gray image both fall into the
    fallback branch (f[3:14] = 0.5) and the blur-difference guard. Neither may
    produce NaN or Inf."""
    degenerate = [
        np.zeros((480, 720, 3), dtype=np.uint8),
        np.full((480, 720, 3), 128, dtype=np.uint8),
    ]
    for img in degenerate:
        f = extract_geometric_features(img)
        assert f.shape == (19,)
        assert np.all(np.isfinite(f))


def test_features_accept_rgba_input():
    """The extractor drops a 4th channel rather than crashing on RGBA."""
    rgba = np.full((480, 720, 4), 128, dtype=np.uint8)
    f = extract_geometric_features(rgba)
    assert f.shape == (19,)
    assert np.all(np.isfinite(f))


def test_flip_is_involution(shadow_image):
    f = extract_geometric_features(shadow_image)
    twice = flip_geometric_features(flip_geometric_features(f))
    assert np.allclose(twice, f, atol=1e-6)


def test_flip_mirrors_x_features():
    """Flipping swaps the left/right pairs and mirrors x-coordinates around 0.5."""
    geo = np.zeros(19, dtype=np.float32)
    geo[0], geo[1] = 0.2, 0.8        # left/right bottom-strip intensities
    geo[3] = 0.25                    # x-centroid
    geo[8] = 0.30                    # left/right mass ratio
    geo[9] = 0.40                    # argmax column density
    geo[10] = 0.45                   # weighted-mean column density
    geo[11], geo[12] = 0.10, 0.60    # left/right edge density

    g = flip_geometric_features(geo)

    assert g[0] == 0.8 and g[1] == 0.2
    assert np.isclose(g[3], 0.75)
    assert np.isclose(g[8], 0.70)
    assert np.isclose(g[9], 0.60)
    assert np.isclose(g[10], 0.55)
    assert g[11] == 0.60 and g[12] == 0.10


def test_flip_does_not_mutate_input():
    geo = np.linspace(0, 1, 19).astype(np.float32)
    original = geo.copy()
    flip_geometric_features(geo)
    assert np.array_equal(geo, original)
