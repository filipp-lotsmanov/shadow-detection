"""Request and response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Predicted bounding box in image coordinates.

    Note: the bbox is intentionally off-screen - coordinates may be negative or
    exceed image dimensions, since the task is to localize a pedestrian that is
    not in the visible frame.
    """

    xmin: float = Field(..., description="Left edge (may be negative)")
    ymin: float = Field(..., description="Top edge")
    xmax: float = Field(..., description="Right edge (may exceed image width)")
    ymax: float = Field(..., description="Bottom edge")


class PredictionResponse(BaseModel):
    bbox: BBox
    side: int = Field(..., description="0 = off-screen left, 1 = off-screen right")
    side_confidence: float = Field(..., ge=0.0, le=1.0)
    direction: int = Field(
        ..., description="0 = walking out of frame, 1 = walking into frame, -1 = abstain"
    )
    direction_confidence: float = Field(..., ge=0.0, le=1.0)
    image_width: int
    image_height: int
    inference_ms: float = Field(..., description="Wall-clock inference time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
