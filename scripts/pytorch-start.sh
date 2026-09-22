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
apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates python3.11 python3.11-venv || apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve >/workspace/ollama.log 2>&1 &
fi

mkdir -p /workspace
cd /workspace
if [ ! -d videoclean/.git ]; then
  git clone https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main

uv python pin 3.11 || true
uv sync --extra gpu --extra lama --extra web --no-dev
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch>=2.5" torchvision
uv pip install --python .venv/bin/python "git+https://github.com/facebookresearch/sam2.git" hf-transfer matplotlib imageio

mkdir -p "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"
uv run videoclean doctor --device cuda || true
exec uv run videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
