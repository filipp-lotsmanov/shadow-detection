# Windows PowerShell setup script.
# Tries the CUDA 12.4 PyTorch build first; falls back to CPU build if
# no NVIDIA GPU is detected or the CUDA install fails.
#
# Usage:
#   .\setup.ps1
#
# If you get an execution policy error, run once per session:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

Write-Host "=== Installing uv (if missing) ===" -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
}

$uvBin = "$env:USERPROFILE\.local\bin"
if ($env:PATH -notlike "*$uvBin*") {
    $env:PATH = "$uvBin;$env:PATH"
}
uv --version

Set-Location -Path $PSScriptRoot

# Wipe a stale venv from previous failed attempts. uv.lock is committed and is
# deliberately left alone: it pins the CUDA resolution that `uv sync` reproduces
# on the default path below.
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

# Detect NVIDIA GPU
$hasNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        & nvidia-smi 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $hasNvidia = $true }
    } catch {
        $hasNvidia = $false
    }
}

if ($hasNvidia) {
    Write-Host "=== NVIDIA GPU detected - installing CUDA 12.4 PyTorch build ===" -ForegroundColor Cyan
    Write-Host "    (download is ~2.5 GB; this will take several minutes)"
    $syncOk = $false
    try {
        uv sync
        $syncOk = $true
    } catch {
        Write-Host "    CUDA sync failed: $_" -ForegroundColor Yellow
    }

    if (-not $syncOk) {
        Write-Host "=== Falling back to CPU PyTorch build ===" -ForegroundColor Yellow
        $hasNvidia = $false
    }
}

if (-not $hasNvidia) {
    Write-Host "=== No NVIDIA GPU detected - installing CPU PyTorch build ===" -ForegroundColor Cyan
    Write-Host "    (download is ~200 MB; training will run on CPU and be slow)"

    # uv.lock is backed up alongside pyproject.toml: pointing at a different
    # index makes the committed lock inconsistent, so `uv sync` re-resolves and
    # rewrites it. Restoring both leaves the working tree clean either way.
    Copy-Item pyproject.toml pyproject.toml.bak
    Copy-Item uv.lock uv.lock.bak
    try {
        (Get-Content pyproject.toml) `
            -replace 'pytorch-cu124', 'pytorch-cpu' `
            -replace 'https://download.pytorch.org/whl/cu124', 'https://download.pytorch.org/whl/cpu' `
            | Set-Content pyproject.toml
        uv sync
    } finally {
        Move-Item -Force pyproject.toml.bak pyproject.toml
        Move-Item -Force uv.lock.bak uv.lock
    }
}

Write-Host "=== Verifying install ===" -ForegroundColor Cyan
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Train:   uv run python scripts/train.py"
Write-Host ""
Write-Host "On CPU, smoke test first:"
Write-Host "  uv run python scripts/train.py training.num_epochs=1 training.batch_size=16 run_name=smoke_cpu"