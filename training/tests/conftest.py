"""Shared fixtures for the training test suite.

The synthetic shadow image is generated deterministically rather than committed
as a binary, and is shaped so that every branch of extract_geometric_features
executes (the road-region block, the PCA block, and the blur-difference block).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# training/tests/ -> training/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_FEATURES_PATH = REPO_ROOT / "backend" / "app" / "features.py"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def backend_features() -> ModuleType:
    """The deployed backend's feature module, loaded directly from its file.

    Loaded by path so the test does not depend on the backend package being
    installed in the training environment. The module only needs numpy + scipy.
    """
    return _load_module_from_path("backend_app_features", BACKEND_FEATURES_PATH)


@pytest.fixture(scope="session")
def shadow_image() -> np.ndarray:
    """Deterministic 480x720 RGB image with a dark elliptical shadow blob in the
    road region. The blob is large and dark enough to exercise every feature
    branch (road density, PCA orientation, and blur-difference scores)."""
    rng = np.random.default_rng(42)
    h, w = 480, 720
    img = rng.integers(150, 200, size=(h, w, 3), dtype=np.uint8)

    yy, xx = np.ogrid[:h, :w]
    cy, cx = 360, 200
    theta = np.deg2rad(30.0)
    u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    blob = (u / 130.0) ** 2 + (v / 70.0) ** 2 <= 1.0
    img[blob] = rng.integers(20, 50, size=(int(blob.sum()), 3), dtype=np.uint8)
    return img
