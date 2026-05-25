#!/bin/bash
# Shadow Detection - one-command launcher.
#
# On first run: installs uv, fetches Python + Node dependencies, and downloads
# the trained model from the latest GitHub release.
# On subsequent runs: starts the backend and frontend.
#
# Press Ctrl+C to stop both services.

set -e

REPO="filipp-lotsmanov/shadow-detection"
MODEL_ARCHIVE="model_artifacts.tar.gz"
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"

# ---------- Helpers ----------
log() { echo -e "\033[1;36m[run]\033[0m $*"; }
warn() { echo -e "\033[1;33m[run]\033[0m $*"; }
err() { echo -e "\033[1;31m[run]\033[0m $*" >&2; }

cleanup() {
    log "Shutting down..."
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    log "Done."
}
trap cleanup EXIT INT TERM

# ---------- Tooling ----------
ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
}

ensure_node() {
    if ! command -v npm >/dev/null 2>&1; then
        err "Node.js (npm) is not installed."
        err "Install it from https://nodejs.org/ (LTS) or via your package manager."
        exit 1
    fi
}

# ---------- Model artifacts ----------
download_model() {
    if [ -f backend/models/model.pt ] && [ -f backend/models/target_stats.json ]; then
        return
    fi
    log "Downloading model artifacts from GitHub release (latest)..."
    mkdir -p backend/models
    cd backend/models

    local url="https://github.com/${REPO}/releases/latest/download/${MODEL_ARCHIVE}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "$MODEL_ARCHIVE" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$MODEL_ARCHIVE" "$url"
    else
        err "Need curl or wget to download the model. Install one and retry."
        exit 1
    fi

    tar -xzf "$MODEL_ARCHIVE"
    rm -f "$MODEL_ARCHIVE"
    cd "$ROOT"
    log "Model installed."
}

# ---------- Backend ----------
ensure_backend_env() {
    if [ -d backend/.venv ]; then
        return
    fi
    log "Installing backend dependencies (~200 MB CPU PyTorch + FastAPI)..."
    cd backend
    uv sync
    cd "$ROOT"
}

# ---------- Frontend ----------
ensure_frontend_env() {
    if [ -d frontend/node_modules ]; then
        return
    fi
    log "Installing frontend dependencies (Next.js)..."
    cd frontend
    npm install
    cd "$ROOT"
}

# ---------- Start services ----------
start_backend() {
    log "Starting backend on http://localhost:8000..."
    cd backend
    uv run uvicorn app.main:app --port 8000 --host 127.0.0.1 &
    BACKEND_PID=$!
    cd "$ROOT"
}

wait_for_backend() {
    log "Waiting for backend to come up..."
    for _ in $(seq 1 30); do
        if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
            log "Backend ready."
            return
        fi
        sleep 1
    done
    err "Backend failed to respond on http://localhost:8000 within 30 seconds."
    exit 1
}

start_frontend() {
    log "Starting frontend on http://localhost:3000..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd "$ROOT"
}

# ---------- Main ----------
ensure_uv
ensure_node
download_model
ensure_backend_env
ensure_frontend_env

start_backend
wait_for_backend
start_frontend

echo ""
log "Shadow Detection is running."
log "  Frontend: http://localhost:3000"
log "  Backend:  http://localhost:8000 (docs at /docs)"
log "Press Ctrl+C to stop."

wait
