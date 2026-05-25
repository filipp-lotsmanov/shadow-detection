"""Convert a trained .pth checkpoint to TorchScript .pt for deployment.

Defaults assume the standard project layout: input from training/outputs/default/,
output to backend/models/.

Usage:
    # From the training/ directory (uses defaults):
    uv run python scripts/export_model.py

    # Or override paths:
    uv run python scripts/export_model.py \
        --checkpoint outputs/smoke/model.pth \
        --target-stats outputs/smoke/target_stats.pth

The resulting .pt file is self-contained: no Python source needed at inference
time, just torch.jit.load().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the training package importable
THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent  # training/
PROJECT_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR / "src"))

import torch  # noqa: E402

from shadow_detection.model import ShadowModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=TRAINING_DIR / "outputs" / "default" / "model.pth",
    )
    p.add_argument(
        "--target-stats",
        type=Path,
        default=TRAINING_DIR / "outputs" / "default" / "target_stats.pth",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "model.pt",
    )
    p.add_argument(
        "--stats-output",
        type=Path,
        default=PROJECT_ROOT / "backend" / "models" / "target_stats.json",
    )
    p.add_argument("--num-geo-features", type=int, default=19)
    args = p.parse_args()

    for f in (args.checkpoint, args.target_stats):
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    print("Building model architecture")
    model = ShadowModel(
        backbone="resnet50",
        pretrained=False,
        dropout=0.3,
        num_geo_features=args.num_geo_features,
    )
    model.load_state_dict(state)
    model.eval()

    print("Tracing to TorchScript")
    # ShadowModel.forward takes (image, geo). Trace with example inputs.
    example_image = torch.randn(1, 3, 384, 384)
    example_geo = torch.randn(1, args.num_geo_features)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_image, example_geo))

    print(f"Saving TorchScript model to {args.output}")
    traced.save(str(args.output))

    print(f"Loading target stats from {args.target_stats}")
    stats = torch.load(args.target_stats, map_location="cpu", weights_only=False)
    print(f"Saving stats as JSON to {args.stats_output}")
    with open(args.stats_output, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Verify the exported model loads cleanly:")
    print(f"  python -c \"import torch; m = torch.jit.load('{args.output}'); print(m)\"")


if __name__ == "__main__":
    main()
