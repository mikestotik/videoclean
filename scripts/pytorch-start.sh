#!/bin/bash
# Boot script for a stock RunPod PyTorch template pod.
# Runs INSIDE the pod as the container start command
# (v2 API `args`: {"entrypoint":["/bin/bash","-lc"],"cmd":[<this file>]}).
# Clone once, then serve. Idempotent: safe to re-run on restart.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/workspace/.videoclean}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export PATH="/root/.local/bin:/usr/local/bin:$PATH"

apt-get update
apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip python3.11 python3.11-venv || apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

# NOTE: ollama is NOT installed here on purpose. It is an on-demand
# component: install + start it from the Config page in the WebUI
# (POST /api/ollama/install, POST /api/ollama/start).

mkdir -p /workspace
cd /workspace
if [ ! -d videoclean/.git ]; then
  git clone https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main

uv python pin 3.11 || true
uv sync --extra gpu --extra lama --extra web --no-dev --no-install-package torch --no-install-package torchvision
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch>=2.5" torchvision
uv pip install --python .venv/bin/python "git+https://github.com/facebookresearch/sam2.git" hf-transfer matplotlib imageio
# The lockfile pins torch==2.2.2 for local machines. Without this, every
# `uv run` below would re-sync the venv and downgrade torch back,
# silently killing CUDA on the pod.
export UV_NO_SYNC=1

mkdir -p "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"

# WebUI: build from source when missing or stale (static_dist is a build
# artifact and is NOT committed to git). Never fatal: the API works without UI,
# so a broken frontend toolchain must not kill the pod.
if [ ! -f server/static_dist/index.html ] || [ -n "$(find webui -path webui/node_modules -prune -o -type f -newer server/static_dist/index.html -print | head -1)" ]; then
  if ! command -v bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash || echo "WARNING: bun install failed, UI will be unavailable"
  fi
  export PATH="/root/.bun/bin:$PATH"
  if command -v bun >/dev/null 2>&1; then
    (cd webui && bun install && bun run build) || echo "WARNING: webui build failed, serving API only"
  fi
fi

uv run videoclean doctor --device cuda || true
exec uv run videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
