#!/bin/bash
# Boot script for a stock RunPod PyTorch template pod.
# Runs INSIDE the pod as the container start command
# (v2 API `args`: {"entrypoint":["/bin/bash","-lc"],"cmd":[<this file>]}).
# Clone once, then serve. Idempotent: safe to re-run on restart.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
# NOTE: keep VIDEOCLEAN_DATA_DIR on a local filesystem. Network volumes
# (mfs) do not support SQLite locking, and jobs.sqlite fails with
# "disk I/O error". Pod volumes are local disk and are fine; with a network
# volume, point this at container disk (e.g. /root/.videoclean) and keep
# HF_HOME on the volume.
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/workspace/.videoclean}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export PATH="/root/.local/bin:/usr/local/bin:$PATH"
# uv cache on container disk, not on /workspace: a 37G cache on a network
# volume trips its quota and breaks git/pip with "Disk quota exceeded".
export UV_CACHE_DIR="/root/.cache/uv"

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
# --inexact: never prune pip-installed extras (sam2 from git is not in the
# lockfile; a pruning sync deletes it and masks fail with "No module named
# 'sam2'"). Locked packages are still synced to their pinned versions.
uv sync --inexact --extra gpu --extra lama --extra web --no-dev --no-install-package torch --no-install-package torchvision
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv/bin/python "git+https://github.com/facebookresearch/sam2.git" hf-transfer matplotlib imageio
# The lockfile pins torch==2.8.0 for local machines. The pod needs the cu128
# build of the SAME version, so install torch from the CUDA index and keep
# `uv run` from re-syncing the venv back (build tags always differ).
export UV_NO_SYNC=1
# uv cache and .venv live on different filesystems here; copy instead of
# hardlinking to silence the warning on every install.
export UV_LINK_MODE=copy

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
