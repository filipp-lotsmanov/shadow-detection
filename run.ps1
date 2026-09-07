# Shadow Detection - Windows launcher.
#
# On first run: installs uv, fetches Python + Node dependencies, and downloads
# the trained model from the latest GitHub release.
# Then opens two new PowerShell windows - one for the backend, one for the
# frontend - so you can see their logs in real time.
#
# This window stays open. Press Ctrl+C or close it to stop the demo (and the
# two child windows will keep running until you close them too).
#
# If PowerShell blocks the script:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

$REPO = "filipp-lotsmanov/shadow-detection"
$MODEL_ARCHIVE = "model_artifacts.tar.gz"
$ROOT = $PSScriptRoot

Set-Location $ROOT

function Log($msg)  { Write-Host "[run] $msg" -ForegroundColor Cyan }
function Err($msg)  { Write-Host "[run] $msg" -ForegroundColor Red }

# Tooling -----------------------------------------------------------
function Ensure-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Log "Installing uv..."
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    }
    $uvBin = "$env:USERPROFILE\.local\bin"
    if ($env:PATH -notlike "*$uvBin*") {
        $env:PATH = "$uvBin;$env:PATH"
    }
}

function Ensure-Node {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Err "Node.js (npm) is not installed."
        Err "Install it from https://nodejs.org/ (LTS) and re-run."
        exit 1
    }
}

# Model artifacts ---------------------------------------------------
function Download-Model {
    if ((Test-Path "backend\models\model.pt") -and (Test-Path "backend\models\target_stats.json")) {
        return
    }
    Log "Downloading model artifacts from GitHub release (latest)..."
    New-Item -ItemType Directory -Force -Path "backend\models" | Out-Null
    Push-Location "backend\models"
    try {
        $url = "https://github.com/$REPO/releases/latest/download/$MODEL_ARCHIVE"
        Invoke-WebRequest -Uri $url -OutFile $MODEL_ARCHIVE -UseBasicParsing
        tar -xzf $MODEL_ARCHIVE
        Remove-Item $MODEL_ARCHIVE -Force
        Log "Model installed."
    } finally {
        Pop-Location
    }
}

# Backend env -------------------------------------------------------
# `uv sync` is idempotent and quick once satisfied, so it runs every time. The
# directory check only decides whether to warn about the download - keying the
# install itself off it would skip repair after an interrupted first run.
function Ensure-BackendEnv {
    if (-not (Test-Path "backend\.venv")) {
        Log "Installing backend dependencies (~200 MB download: CPU PyTorch + FastAPI)..."
    }
    Push-Location backend
    try {
        uv sync
    } finally {
        Pop-Location
    }
}

# Frontend env ------------------------------------------------------
function Ensure-FrontendEnv {
    if (-not (Test-Path "frontend\node_modules")) {
        Log "Installing frontend dependencies (Next.js)..."
    }
    Push-Location frontend
    try {
        # `npm install`, not `npm ci` - it returns in about a second when the
        # tree already matches instead of reinstalling on every launch.
        npm install
    } finally {
        Pop-Location
    }
}

# Main --------------------------------------------------------------
Ensure-Uv
Ensure-Node
Download-Model
Ensure-BackendEnv
Ensure-FrontendEnv

Log "Launching backend in a new window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ROOT\backend'; uv run uvicorn app.main:app --port 8000"
)

# Poll /health rather than sleeping a fixed interval, matching run.sh. Cold
# start is dominated by importing torch and can exceed a fixed wait on a slow
# disk or with an antivirus scanning the 100 MB model.
function Wait-ForBackend {
    Log "Waiting for backend to come up..."
    foreach ($i in 1..30) {
        try {
            Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing `
                -TimeoutSec 2 | Out-Null
            Log "Backend ready."
            return
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    Err "Backend did not respond on http://localhost:8000 within 30 seconds."
    Err "Check the backend window for errors; the frontend will retry on its own."
}

Wait-ForBackend

Log "Launching frontend in a new window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ROOT\frontend'; npm run dev"
)

Write-Host ""
Log "Shadow Detection is running."
Log "  Frontend: http://localhost:3000"
Log "  Backend:  http://localhost:8000 (docs at /docs)"
Log ""
Log "Two new PowerShell windows opened with the backend and frontend logs."
Log "To stop the demo, close those windows (or Ctrl+C in each)."
