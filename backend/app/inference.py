"""Shadow detection inference for the FastAPI backend.

Loads a TorchScript-exported ShadowModel and runs prediction with horizontal-flip
test-time augmentation. Replicates the test-time path from the training repo
exactly, but without the multi-model ensembling (single-model deployment).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .features import extract_geometric_features

# Constants must match training-time settings exactly.
IMG_W = 720
IMG_H = 480
INPUT_SIZE = (384, 384)  # (H, W)
CHANNEL_MEANS = [0.422, 0.413, 0.394]
CHANNEL_STDS = [0.167, 0.174, 0.233]
DIRECTION_CONFIDENCE_THRESHOLD = 0.6
GEO_FLIP_BIN_PAIRS = [(0, 1), (11, 12)]  # indices that swap under horizontal flip
GEO_MIRROR_INDICES = [3, 8, 9, 10]  # indices that map x -> 1 - x


class PredictionDict(TypedDict):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    side: int
    side_confidence: float
    direction: int
    direction_confidence: float
    image_width: int
    image_height: int
    inference_ms: float


def flip_geometric_features(geo: np.ndarray) -> np.ndarray:
    """Mirror the 19 features to match a horizontally flipped image."""
    g = geo.copy()
    for a, b in GEO_FLIP_BIN_PAIRS:
        g[a], g[b] = geo[b], geo[a]
    g[2] = g[0] / (g[1] + 1e-6)
    for i in GEO_MIRROR_INDICES:
        g[i] = 1 - geo[i]
    return g


class ShadowDetector:
    """Singleton wrapper around the TorchScript model + denormalization."""

    def __init__(self, model_path: Path, stats_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        with open(stats_path) as f:
            self.target_stats = json.load(f)

        self.transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(CHANNEL_MEANS, CHANNEL_STDS),
            ]
        )

    def _denormalize(self, value: float, key: str) -> float:
        s = self.target_stats[key]
        return value * (s["std"] + 1e-8) + s["mean"]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionDict:
        """Run TTA prediction on a single PIL image."""
        t_start = time.perf_counter()

        image = image.convert("RGB")
        original_w, original_h = image.size

        # Geo features from the raw image (no resize)
        img_np = np.array(image)
        geo = extract_geometric_features(img_np)
        geo_t = torch.from_numpy(geo).float().unsqueeze(0).to(self.device)

        # Forward - original orientation
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        side_logits, reg_preds, dir_logits = self.model(img_t, geo_t)

        # Forward - horizontally flipped
        image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        img_flip_t = self.transform(image_flip).unsqueeze(0).to(self.device)
        geo_flip = flip_geometric_features(geo)
        geo_flip_t = torch.from_numpy(geo_flip).float().unsqueeze(0).to(self.device)
        side_logits_f, reg_preds_f, dir_logits_f = self.model(img_flip_t, geo_flip_t)

        # Regression: average raw outputs (decomposed targets are flip-invariant)
        rp = ((reg_preds + reg_preds_f) / 2.0).cpu().float().numpy()[0]

        # Classification: softmax-average with index reversal for the flipped pass
        sp1 = F.softmax(side_logits, dim=1).cpu().numpy()[0]
        sp2 = F.softmax(side_logits_f, dim=1).cpu().numpy()[0]
        sp = (sp1 + sp2[::-1]) / 2

        dp1 = F.softmax(dir_logits, dim=1).cpu().numpy()[0]
        dp2 = F.softmax(dir_logits_f, dim=1).cpu().numpy()[0]
        dp = (dp1 + dp2[::-1]) / 2

        side = int(sp.argmax())
        side_confidence = float(sp.max())

        # Reconstruct bbox from decomposed values
        dist = max(self._denormalize(float(rp[0]), "distance_from_edge"), 0.0)
        bw = max(self._denormalize(float(rp[1]), "bbox_width"), 10.0)
        bh = max(self._denormalize(float(rp[2]), "bbox_height"), 50.0)
        yc = self._denormalize(float(rp[3]), "y_center")

        if side == 0:
            xmin = -dist
            xmax = xmin + bw
        else:
            xmax = IMG_W + dist
            xmin = xmax - bw
        ymin = yc - bh / 2
        ymax = yc + bh / 2

        # Scale bbox to actual uploaded image dimensions if it isn't 720x480
        sx = original_w / IMG_W
        sy = original_h / IMG_H
        xmin, xmax = xmin * sx, xmax * sx
        ymin, ymax = ymin * sy, ymax * sy

        dir_confidence = float(dp.max())
        direction = int(dp.argmax()) if dir_confidence > DIRECTION_CONFIDENCE_THRESHOLD else -1

        return {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "side": side,
            "side_confidence": side_confidence,
            "direction": direction,
            "direction_confidence": dir_confidence,
            "image_width": original_w,
            "image_height": original_h,
            "inference_ms": (time.perf_counter() - t_start) * 1000.0,
        }
