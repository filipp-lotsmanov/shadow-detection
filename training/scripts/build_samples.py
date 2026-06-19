"""Build the demo gallery: pick 8 sample images from the training set, copy
images + GT into frontend/public/samples/, write a samples.json manifest the
frontend loads.

Run once after training the model. Re-run any time you want to refresh the gallery.

Usage:
    python scripts/build_samples.py --train-dir training/train_data --n 8
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "samples",
    )
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.train_dir.exists():
        raise FileNotFoundError(f"Training data dir not found: {args.train_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any previous samples
    for old in args.output_dir.glob("*.png"):
        old.unlink()

    pngs = sorted(args.train_dir.rglob("*.png"))
    jsons = {p.stem: p for p in args.train_dir.rglob("*.json")}

    # Keep only pngs that have matching json annotations
    paired = [p for p in pngs if p.stem in jsons]
    print(f"Found {len(paired)} paired (png, json) entries")

    if len(paired) < args.n:
        raise ValueError(f"Need at least {args.n} samples, found {len(paired)}")

    rng = random.Random(args.seed)
    picks = rng.sample(paired, args.n)

    manifest = []
    for png in picks:
        with open(jsons[png.stem]) as f:
            ann = json.load(f)

        tl = ann["bbox"]["top_left"]
        br = ann["bbox"]["bottom_right"]

        shutil.copy(png, args.output_dir / f"{png.stem}.png")

        manifest.append(
            {
                "id": png.stem,
                "file": f"{png.stem}.png",
                "bbox": {
                    "xmin": float(tl[0]),
                    "ymin": float(tl[1]),
                    "xmax": float(br[0]),
                    "ymax": float(br[1]),
                },
                "direction": int(ann.get("walking_into_frame_bool", 0)),
            }
        )

    manifest_path = args.output_dir / "samples.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Copied {len(manifest)} samples to {args.output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
