#!/bin/bash
# Boot script for the stock RunPod PyTorch 2.8 template
# (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# Runs INSIDE the pod as the container start command
# (v2 API `args`: {"entrypoint":["/bin/bash","-lc"],"cmd":[<this file>]}).
# Clone once, then serve. Idempotent: safe to re-run on restart.
#
# Torch, torchvision, torchaudio and numpy stay on the image interpreter.
# A fresh venv does not see that torch, and `torch==2.8.0` does not match
# `2.8.0+cu128`, so uv sync would replace the CUDA build.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
# NOTE: keep VIDEOCLEAN_DATA_DIR on a local filesystem. Network volumes
# (mfs) do not support SQLite locking, and jobs.sqlite fails with
# "disk I/O error". Pod volumes are local disk and are fine; with a network
# volume, point this at container disk (e.g. /root/.videoclean) and keep
# HF_HOME on the volume.
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/root/.videoclean}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export PATH="/usr/local/bin:/usr/bin:$PATH"
# uv cache on container disk, not on /workspace: a 37G cache on a network
# volume trips its quota and breaks git/pip with "Disk quota exceeded".
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

PY="$(command -v python3)"
"$PY" -c 'import torch; ok = torch.cuda.is_available(); print(torch.__version__, "cuda="+str(ok)); raise SystemExit(0 if ok else 1)'

apt-get update
apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
fi

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

# Pin the image copies so resolution cannot replace them. Local versions
# (+cu128) are written as installed; overrides beat the pyproject pin
# numpy<2, which would otherwise downgrade the image's numpy.
"$PY" - <<'PY' > /tmp/image-pins.txt
import importlib
for name in ("torch", "torchvision", "torchaudio", "numpy"):
    try:
        mod = importlib.import_module(name)
    except Exception:
        continue
    ver = getattr(mod, "__version__", None)
    if ver:
        print(f"{name}=={ver}")
PY

uv pip install --python "$PY" --system --break-system-packages \
  --overrides /tmp/image-pins.txt \
  --extra gpu --extra lama --extra web \
  -e .
uv pip install --python "$PY" --system --break-system-packages \
  --overrides /tmp/image-pins.txt \
  "git+https://github.com/facebookresearch/sam2.git"

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

"$PY" -m videoclean doctor --device cuda || true
exec "$PY" -m videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
