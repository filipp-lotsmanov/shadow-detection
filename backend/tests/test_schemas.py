"""API contract tests for the prediction response schema.

These only import app.schemas (pure Pydantic) and do not load the model or
torch, so they run in the lightweight backend environment.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BBox, PredictionResponse


def _valid_payload() -> dict:
    return {
        "bbox": BBox(xmin=-82.4, ymin=213.1, xmax=18.7, ymax=412.5),
        "side": 0,
        "side_confidence": 0.998,
        "direction": 1,
        "direction_confidence": 0.74,
        "image_width": 720,
        "image_height": 480,
        "inference_ms": 142.3,
    }


def test_valid_response_accepted():
    resp = PredictionResponse(**_valid_payload())
    assert resp.side == 0
    assert resp.bbox.xmin == -82.4


def test_abstain_direction_accepted():
    payload = _valid_payload()
    payload["direction"] = -1
    resp = PredictionResponse(**payload)
    assert resp.direction == -1


def test_negative_bbox_coordinates_allowed():
    """Off-screen boxes legitimately have negative or out-of-frame coordinates."""
    bbox = BBox(xmin=-202.8, ymin=223.6, xmax=-155.3, ymax=369.3)
    assert bbox.xmax < 0


@pytest.mark.parametrize("field", ["side_confidence", "direction_confidence"])
def test_confidence_above_one_rejected(field):
    payload = _valid_payload()
    payload[field] = 1.5
    with pytest.raises(ValidationError):
        PredictionResponse(**payload)


@pytest.mark.parametrize("field", ["side_confidence", "direction_confidence"])
def test_confidence_below_zero_rejected(field):
    payload = _valid_payload()
    payload[field] = -0.1
    with pytest.raises(ValidationError):
        PredictionResponse(**payload)
