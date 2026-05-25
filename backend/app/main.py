"""FastAPI inference server for the shadow detection model.

Endpoints:
    GET  /health   - liveness probe + device info
    POST /predict  - accepts an image file, returns predicted bbox + classifications
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .inference import ShadowDetector
from .schemas import BBox, HealthResponse, PredictionResponse

log = logging.getLogger("shadow-detector-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "model.pt"
STATS_PATH = MODELS_DIR / "target_stats.json"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Shadow Detector API",
    description="Predicts off-screen pedestrian bounding boxes from shadow imagery.",
    version="0.1.0",
)

# Permissive CORS for local development. The frontend runs on port 3000 by
# default; this lets the browser talk to the backend on port 8000.
# Allow any localhost origin. The frontend may run on a different local port
# (3000 default, or anything else when behind a tunnel like coder port-forward).
# Tightening this is appropriate when deploying to a real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector: ShadowDetector | None = None


@app.on_event("startup")
def _load_model() -> None:
    global detector
    try:
        detector = ShadowDetector(MODEL_PATH, STATS_PATH)
        log.info("Model loaded on device=%s", detector.device)
    except FileNotFoundError as e:
        log.error("Model files missing: %s", e)
        # Server starts; /health reports model_loaded=False so the user knows to export


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if detector is None:
        return HealthResponse(status="model_missing", model_loaded=False, device="cpu")
    return HealthResponse(status="ok", model_loaded=True, device=str(detector.device))


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run 'uv run python ../scripts/export_model.py' first "
                "to populate backend/models/."
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Expected an image, got: {file.content_type}")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes")

    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e

    pred = detector.predict(image)

    return PredictionResponse(
        bbox=BBox(
            xmin=pred["xmin"], ymin=pred["ymin"], xmax=pred["xmax"], ymax=pred["ymax"]
        ),
        side=pred["side"],
        side_confidence=pred["side_confidence"],
        direction=pred["direction"],
        direction_confidence=pred["direction_confidence"],
        image_width=pred["image_width"],
        image_height=pred["image_height"],
        inference_ms=pred["inference_ms"],
    )
