"""Parity between the deployed and training feature extractors.

This is the most important invariant in the project. The backend serves a model
trained against training/src/shadow_detection/features.py, so backend/app/
features.py must produce bit-identical output. Any drift silently degrades the
served model without raising an error anywhere else.
"""

from __future__ import annotations

import numpy as np
from shadow_detection.features import extract_geometric_features


def test_backend_features_match_training_exactly(shadow_image, backend_features):
    train_f = extract_geometric_features(shadow_image)
    backend_f = backend_features.extract_geometric_features(shadow_image)

    assert train_f.shape == backend_f.shape == (19,)
    assert np.array_equal(train_f, backend_f)


def test_backend_declares_19_features(backend_features):
    assert backend_features.NUM_FEATURES == 19
