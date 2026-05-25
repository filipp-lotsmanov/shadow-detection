# Training pipeline

This subdirectory contains the PyTorch + Hydra training pipeline that produces the model used by the demo. For the full project overview, architecture details, and the deployed demo, see the [top-level README](../README.md).

## Setup

```bash
./setup.sh                # Linux/macOS
.\setup.ps1               # Windows PowerShell
```

Detects an NVIDIA GPU and installs the CUDA 12.4 PyTorch build if available, or the CPU build otherwise.

## Data layout

Place the dataset under `training/`:

```
training/
+-- train_data/             image+annotation pairs (.png + .json)
+-- test_data/              .png files for the held-out test set
+-- submission_example.csv  test-set row order + CSV format
```

## Training

```bash
uv run python scripts/train.py                                  # full training run
uv run python scripts/train.py training.num_epochs=2 run_name=smoke   # quick smoke test
```

Outputs land in `outputs/<run_name>/`:

- `model.pth` - trained PyTorch state dict
- `target_stats.pth` - regression-target normalization stats used at inference

## Test-set prediction

```bash
uv run python scripts/predict.py
```

Loads `outputs/<run_name>/model.pth`, runs over the test set with horizontal-flip test-time augmentation, and writes `outputs/<run_name>/submission.csv`.

## Exporting for deployment

After training, export the model to TorchScript so the FastAPI backend can serve it without the training package installed:

```bash
uv run python scripts/export_model.py
```

By default, reads `outputs/default/model.pth` + `outputs/default/target_stats.pth` and writes `backend/models/model.pt` + `backend/models/target_stats.json`.

## Configuration

All hyperparameters live in `configs/` and can be overridden on the CLI:

```bash
uv run python scripts/train.py data.flip_prob=0.0 run_name=no_flip
uv run python scripts/train.py training.learning_rate=1e-3 training.batch_size=64
```

See `configs/config.yaml` for the full schema.
