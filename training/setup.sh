#!/bin/bash
# Linux/macOS setup. Tries the CUDA 12.4 PyTorch build first; falls back to
# the CPU build if no NVIDIA GPU is detected or the CUDA install fails.
set -e

echo "=== Installing uv (if missing) ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
uv --version

cd "$(dirname "$0")"

# Wipe stale venv / lockfile from previous failed attempts
rm -rf .venv uv.lock

# Detect NVIDIA GPU
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        HAS_NVIDIA=1
    fi
fi

sync_cuda() {
    echo "=== NVIDIA GPU detected - installing CUDA 12.4 PyTorch build ==="
    echo "    (download is ~2.5 GB; this will take several minutes)"
    uv sync
}

sync_cpu() {
    echo "=== Installing CPU PyTorch build ==="
    echo "    (download is ~200 MB; training will run on CPU and be slow)"
    # Temporarily switch the index in pyproject.toml; restore afterward so
    # the file committed to GitHub stays CUDA-default.
    cp pyproject.toml pyproject.toml.bak
    trap 'mv -f pyproject.toml.bak pyproject.toml' EXIT
    sed -i.tmp \
        -e 's|pytorch-cu124|pytorch-cpu|g' \
        -e 's|https://download.pytorch.org/whl/cu124|https://download.pytorch.org/whl/cpu|g' \
        pyproject.toml
    rm -f pyproject.toml.tmp
    uv sync
    mv -f pyproject.toml.bak pyproject.toml
    trap - EXIT
}

if [ "$HAS_NVIDIA" -eq 1 ]; then
    if sync_cuda; then
        :
    else
        echo "=== CUDA sync failed - falling back to CPU ==="
        sync_cpu
    fi
else
    echo "=== No NVIDIA GPU detected ==="
    sync_cpu
fi


echo "=== Verifying install ==="
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "=== Setup complete ==="
echo "Train:   uv run python scripts/train.py"
echo ""
echo "On CPU, smoke test first:"
echo "  uv run python scripts/train.py training.num_epochs=1 training.batch_size=16 run_name=smoke_cpu"
