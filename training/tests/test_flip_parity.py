"""The backend and training copies of flip_geometric_features must agree.

The two files are maintained as verbatim copies; this test guards against either
copy being edited without the other.
"""

from __future__ import annotations

import numpy as np
from shadow_detection.features import flip_geometric_features as train_flip


def test_backend_flip_matches_training_flip(backend_features):
    rng = np.random.default_rng(0)
    for _ in range(1000):
        geo = rng.random(19).astype(np.float32)
        a = train_flip(geo.copy())
        b = backend_features.flip_geometric_features(geo.copy())
        assert np.array_equal(a, b)
