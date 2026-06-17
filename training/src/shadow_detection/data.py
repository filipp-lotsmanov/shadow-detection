"""Dataset loading and target decomposition.

The decomposition strategy: rather than predicting raw bbox coordinates
[xmin, ymin, xmax, ymax], we decompose into:
  - side: 0 (off-screen left) or 1 (off-screen right)
  - distance_from_edge: how far past the frame edge (always positive)
  - bbox_width, bbox_height: box dimensions
  - y_center: vertical center

This explicitly encodes the bimodal x-distribution discovered during EDA.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _verify_decomposition(samples: list[dict[str, Any]], img_w: int) -> None:
    """Sanity check - reconstruct bbox from decomposed values and compare."""
    for s in samples[:3]:
        if s["side"] == 0:
            recon_xmin = -s["distance_from_edge"]
            recon_xmax = recon_xmin + s["bbox_width"]
        else:
            recon_xmax = img_w + s["distance_from_edge"]
            recon_xmin = recon_xmax - s["bbox_width"]
        assert abs(recon_xmin - s["xmin"]) < 1e-6, f"xmin mismatch: {recon_xmin} vs {s['xmin']}"
        assert abs(recon_xmax - s["xmax"]) < 1e-6, f"xmax mismatch: {recon_xmax} vs {s['xmax']}"


def load_annotations_with_features(
    data_dir: Path, img_w: int, verbose: bool = True
) -> list[dict[str, Any]]:
    """Same as load_annotations() but also precomputes the 19 geometric features.

    Feature extraction reads each image once, so this is much slower than the bare
    annotation loader. Cache the result if iterating during experimentation.
    """
    from .features import extract_geometric_features

    data_dir = Path(data_dir)
    png_files = {p.stem: p for p in data_dir.rglob("*.png")}
    json_files = sorted(data_dir.rglob("*.json"))

    samples = []
    t0 = time.time()
    for i, jf in enumerate(json_files):
        if verbose and i % 300 == 0:
            print(f"  {i}/{len(json_files)} ({time.time() - t0:.0f}s)")

        with open(jf) as f:
            ann = json.load(f)

        fname = ann["file_name"]
        if fname not in png_files:
            continue

        tl = ann["bbox"]["top_left"]
        br = ann["bbox"]["bottom_right"]
        xmin, ymin = float(tl[0]), float(tl[1])
        xmax, ymax = float(br[0]), float(br[1])

        width = xmax - xmin
        height = ymax - ymin
        y_center = (ymin + ymax) / 2.0

        if xmin < 0:
            side = 0
            distance_from_edge = abs(xmin)
        else:
            side = 1
            distance_from_edge = xmax - img_w

        # Precompute geometric features from the image
        img = np.array(Image.open(png_files[fname]).convert("RGB"))
        geo = extract_geometric_features(img)

        samples.append(
            {
                "img_path": str(png_files[fname]),
                "file_name": fname,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "side": side,
                "distance_from_edge": distance_from_edge,
                "bbox_width": width,
                "bbox_height": height,
                "y_center": y_center,
                "direction": int(ann["walking_into_frame_bool"]),
                "geo": geo,
            }
        )

    _verify_decomposition(samples, img_w)
    return samples


def compute_target_stats(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute z-score normalization stats for the 4 regression targets."""
    keys = ["distance_from_edge", "bbox_width", "bbox_height", "y_center"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([s[k] for s in samples], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def normalize_target(value: float, key: str, stats: dict[str, dict[str, float]]) -> float:
    return (value - stats[key]["mean"]) / (stats[key]["std"] + 1e-8)
