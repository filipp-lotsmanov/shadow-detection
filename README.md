# Shadow Detection

[![ci](https://github.com/filipp-lotsmanov/shadow-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/filipp-lotsmanov/shadow-detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)

> Predicting off-screen pedestrian locations from shadow imagery.
> Winning solution for the **BrabantHack 2026 DEMCON Deep Tech track** with **IoU 0.626** on the official test set.

The task: given a single 720x480 image of a road scene where a pedestrian is *not visible in frame* but their shadow is, predict the off-screen bounding box where the pedestrian would be standing.

This repository contains a runnable demo plus the full training pipeline.

![Demo preview](docs/demo.gif)

## Quick start

> **Prerequisites.** **Node.js 20+** must be installed and on your PATH *before* running the launch script - it is the only thing you need to install yourself.
>
> You do **not** need to install Python. The project targets the Python 3.10 series, and uv provisions a matching interpreter automatically if your machine doesn't have one. Everything else (uv itself, Python dependencies, npm dependencies, and the trained model) is set up on first run.
>
> - Node.js (LTS): https://nodejs.org/

**Linux / macOS:**

```bash
git clone https://github.com/filipp-lotsmanov/shadow-detection.git
cd shadow-detection
chmod +x run.sh
./run.sh
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/filipp-lotsmanov/shadow-detection.git
cd shadow-detection
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

First run downloads the trained model (~90 MB, expanding to ~100 MB on disk) and installs dependencies - CPU PyTorch into `backend/.venv` (~200 MB of wheels) and the Next.js toolchain into `frontend/node_modules` (~410 MB unpacked) - so it takes 3-5 minutes. Subsequent runs start in seconds. When ready, open `http://localhost:3000` in your browser.

The frontend calls the backend at `http://localhost:8000` by default. If you run the backend elsewhere (a different port, or a remote server such as a Coder workspace accessed via web URLs), copy `frontend/.env.local.example` to `frontend/.env.local`, set `NEXT_PUBLIC_API_URL` to the backend's reachable address, and restart `npm run dev`. When the frontend is served from a non-`localhost` origin you must also allow that origin in the backend's CORS configuration (`backend/app/main.py`), which permits `localhost` only by default.

## What you'll see

A demo page where you can either upload your own shadow image or pick from 8 bundled sample images (which have ground-truth bounding boxes for comparison). The page renders:

- The input image with the predicted bounding box overlay (red), and the ground-truth bbox (green) if the sample has one
- Side classification (left vs right of frame) with confidence
- Direction classification (walking into / out of frame) - **the released model abstains here on every input**, see [Limitations](#limitations)
- Predicted bbox coordinates, image dimensions, inference latency

## Approach

### Key architectural decision: decomposed bbox prediction

Rather than regressing raw `[xmin, ymin, xmax, ymax]` coordinates, the bounding box is decomposed into:

- `side`: classification (left vs right of frame)
- `distance_from_edge`: regression (always positive)
- `bbox_width`, `bbox_height`: regression
- `y_center`: regression

This was motivated by EDA: the x-coordinate distribution is bimodal (people are always either left or right of the frame), creating a discontinuity that's hard for a single regressor to model. The dedicated side classifier handles the discontinuity, leaving the regressor to learn simpler continuous offsets.

### Architecture

```
       Image (3, 384, 384)              19 geometric features
              |                                 |
              v                                 |
       ResNet-50 backbone                       |
       (ImageNet-pretrained)                    |
              |                                 |
              v                                 |
       2048-d image features <-----[concat]----+
              |
              v
       Linear(2067 -> 512) + BatchNorm + ReLU + Dropout
              |
       +------+------+------+
       |             |      |
       v             v      v
    side head    reg head   direction head
    (2 logits)   (4 dims)   (2 logits)
```

### The 19 geometric features

Computed from each raw image and concatenated with the ResNet features. These were the largest single improvement during the hackathon - adding them lifted IoU from ~0.57 to ~0.62.

- LAB-colorspace shadow mask + density on the road region
- Shadow centroid (x, y) and spread (std_x, std_y)
- Left/right mass ratio
- Column-density argmax and weighted mean
- Left/right edge-strip density (30 px)
- PCA principal axis angle of the shadow blob
- Sobel edge-response mean and 90th percentile
- Bottom-strip intensities and ratio (left/right corners)

The features are mirrored when the image is horizontally flipped during augmentation.

### Iteration story

| Stage | Change | Approx. IoU |
|---|---|---|
| Baseline | ResNet-50 + 3-head decomposed regression | ~0.52 |
| + Augmentation | Horizontal flip with side-label mirroring, larger input | ~0.57 |
| + Geo features (final) | 19 hand-crafted features alongside ResNet | **0.626** |

### Inference (production)

Each request runs the model on the original image and its horizontal flip (test-time augmentation). Softmax outputs are averaged with reversed class indices for the flipped pass, and regression outputs are averaged across orientations. Direction reports `-1` (abstain) when peak confidence falls below 0.6 - which, for the released model, is every input (see [Limitations](#limitations)).

The deployed model is a single TorchScript-traced ResNet-50: ~25M fp32 parameters, so ~100 MB on disk and ~90 MB as the compressed release asset.

## Repository layout

```
shadow-detection/
+-- backend/                       FastAPI inference server
|   +-- app/
|   |   +-- __init__.py
|   |   +-- main.py                HTTP routes, CORS, model loading
|   |   +-- inference.py           TorchScript model, TTA
|   |   +-- geometry.py            Target denormalization + bbox reconstruction (torch-free)
|   |   +-- images.py              Upload decoding + size contract (torch-free)
|   |   +-- features.py            19 geometric features (matches training-time impl)
|   |   +-- schemas.py             Pydantic request/response models
|   +-- tests/                     Schema, geometry and upload-decoding tests (no torch needed)
|   +-- models/                    Populated at runtime from GitHub releases (model.pt + target_stats.json)
|   +-- pyproject.toml             CPU PyTorch + FastAPI deps
+-- frontend/                      Next.js 15 (App Router, JavaScript, CSS modules)
|   +-- app/
|   |   +-- page.js                Main orchestration
|   |   +-- page.module.css
|   |   +-- layout.js
|   |   +-- globals.css
|   +-- components/
|   |   +-- ImageUploader.js       Drag-drop + sample gallery
|   |   +-- ImageUploader.module.css
|   |   +-- PredictionViewer.js    Canvas-based bbox overlay
|   |   +-- PredictionViewer.module.css
|   |   +-- PredictionStats.js     Side panel with classification stats
|   |   +-- PredictionStats.module.css
|   +-- public/samples/            8 demo images + samples.json (ground-truth manifest)
|   +-- package.json
|   +-- package-lock.json
|   +-- next.config.js
|   +-- jsconfig.json
|   +-- .env.local.example
+-- training/                      Full training pipeline (Hydra-configured)
|   +-- src/shadow_detection/      PyTorch package
|   |   +-- __init__.py
|   |   +-- data.py                Annotation loading + target decomposition
|   |   +-- dataset.py             Dataset + augmentation
|   |   +-- features.py            19 geometric features
|   |   +-- model.py               ResNet-50 + 3-head architecture
|   |   +-- gpu.py                 Free-GPU selection
|   |   +-- runtime.py             Device / AMP / DataLoader helpers
|   +-- scripts/
|   |   +-- train.py               Training entry point
|   |   +-- export_model.py        Checkpoint -> TorchScript for deployment
|   |   +-- build_samples.py       Regenerate the demo gallery (maintainer tool)
|   +-- configs/                   Hydra YAML configs (config, data, model, training)
|   +-- tests/                     Feature, flip and target-decomposition tests
|   +-- setup.sh / setup.ps1       Environment setup (GPU autodetect, CPU fallback)
|   +-- pyproject.toml             CUDA PyTorch deps
|   +-- README.md                  Training-specific documentation
+-- .github/workflows/ci.yml       Tests, ruff, and the frontend build on every push
+-- docs/
|   +-- demo.gif                   Demo preview shown above
+-- run.sh                         One-command launcher (Linux/macOS)
+-- run.ps1                        One-command launcher (Windows)
+-- LICENSE                        MIT
+-- README.md
```

## API

`POST /predict` accepts a multipart image file and returns:

```json
{
  "bbox": { "xmin": -82.4, "ymin": 213.1, "xmax": 18.7, "ymax": 412.5 },
  "side": 0,
  "side_confidence": 0.998,
  "direction": -1,
  "direction_confidence": 0.52,
  "image_width": 720,
  "image_height": 480,
  "inference_ms": 384.2
}
```

Latency is dominated by the two forward passes of the TTA. On CPU, a 720x480 image takes roughly 300-400 ms end to end (measured mean 384 ms over the eight bundled samples); larger uploads cost proportionally more, and a CUDA device brings it down substantially.

`side`: `0` = off-screen left, `1` = off-screen right.
`direction`: `0` = walking out of frame, `1` = walking into frame, `-1` = abstain. The released model returns `-1` for every input; the schema keeps `0`/`1` because the training pipeline still produces the head.

Error responses: `415` if the upload isn't an image, `400` if it can't be decoded (truncated or corrupt) or is empty, `413` over 10 MB, `422` if either dimension is under 64 px, `503` if the model failed to load.

**Input contract.** The model expects 720x480 road scenes. Other sizes are accepted and the returned box is scaled to match, but the geometric features use absolute pixel windows (20 px corner strips, 30 px edge strips), so the further an image is from 720x480 the further those features drift from the training distribution. Anything below 64x64 is rejected outright rather than answered with a meaningless box.

Full OpenAPI docs at `/docs` once the backend is running.

## Tests

Both packages ship a test suite, run on every push by [CI](.github/workflows/ci.yml) along with `ruff` and the frontend build.

```bash
cd backend  && uv run pytest      # 30 tests: schemas, bbox geometry, upload decoding
cd training && uv run pytest      # 25 tests: geometric features, flip parity, target decomposition
```

The suites are deliberately torch-free, so they run in seconds without the multi-gigabyte PyTorch install. Two invariants are worth calling out:

- **Feature parity.** `backend/app/features.py` is a copy of the training-time extractor, and the served model is only correct if the two agree exactly. `test_feature_parity.py` loads the backend file by path and asserts bit-identical output on a synthetic image shaped to exercise every branch, with `test_flip_parity.py` doing the same for the mirror transform over 1000 random vectors.
- **Decomposition round trip.** The training side decomposes a bbox into `(side, distance_from_edge, width, height, y_center)` and normalizes; the backend denormalizes and reconstructs. `test_targets.py` drives the real function from each package and asserts they are exact inverses, so a change to either half fails the test rather than passing against a restatement of the maths.

```bash
cd frontend && npm run lint && npm run build
```

## Reproducing the training

The trained model is bundled via GitHub releases so reviewers don't need a GPU to run the demo. If you want to retrain from scratch, the full training pipeline is in `training/` - see `training/README.md` for details. Training takes ~11 minutes on an NVIDIA L40S and is impractical without a CUDA GPU.

## Team

The winning hackathon submission was a joint effort with Oleksii Krasnoshtanov and Danil Sysenko. The 3-head decomposed-target architecture, 19 hand-crafted geometric features, flip-aware augmentation, and TTA inference were my contributions. Danil and Oleksii ran complementary models in parallel; the original hackathon submission was a weighted blend of all of our individual best results.

## Limitations

A few honest caveats about what this model can and cannot do:

- **Direction prediction does not work at all in the released model.** The side classifier (left/right of frame) is effectively solved at ~100% accuracy and bounding-box regression is strong, but the "walking into vs out of frame" head is uninformative. On the 8 bundled sample images its peak confidence after TTA ranges from 0.500 to 0.541 - the abstain threshold is 0.6, so the demo returns `-1` for all of them, and in practice for any input. Ignoring the threshold entirely, its argmax is correct on 4 of those 8 samples, i.e. chance, and those are images it was trained on. During the hackathon this head reached 65-70% in the best runs and 50-60% in most, so the released weights are at the weak end of a head that was never reliable. Treat the direction output as a documented dead end rather than a feature.

  One untested explanation is worth recording: `dataset.py` mirrors the direction label on horizontal flip (`direction = 1 - direction`), and `inference.py` reverses the flipped pass's class indices to match. But "walking into frame" looks mirror-invariant - a pedestrian off-screen left walking rightwards (into frame) mirrors to one off-screen right walking leftwards, which is still into frame. If that reading is correct, the augmentation feeds roughly 50% label noise to this head on every flipped sample, which would account for it collapsing to chance while side and regression (whose labels are handled correctly under flip) train fine. Retraining with the direction label left unchanged under flip, and dropping the index reversal in `inference.py` to match, would settle it. This has not been tested.
- **Trained and evaluated on synthetic data only.** The entire dataset is computer-generated. The model has never seen a real photograph, and real-world generalization is untested. Lighting, shadow softness, ground textures, and camera characteristics in real scenes differ from the synthetic distribution in ways that would likely degrade performance.
- **No held-out ground-truth test set.** The IoU 0.626 figure comes from the competition leaderboard, which scored a hidden test set. Locally there is no ground-truth test split, so internal validation during the hackathon was indirect (via leaderboard submissions). The model trains on all available annotated samples.
