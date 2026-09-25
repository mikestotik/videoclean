#!/bin/bash
# Stock image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404.
# The image entrypoint stays in place. This script only clones, installs
# into the image Python, and serves. It does not create a venv and it does
# not exit when torch.cuda.is_available() is false.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
# Network volumes break SQLite locking. Keep the db on container disk.
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/root/.videoclean}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export PATH="/usr/local/bin:/usr/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

PY="$(command -v python3)"
"$PY" -c 'import torch; print(torch.__version__, "cuda="+str(torch.cuda.is_available()))'

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:${PATH}"
fi

mkdir -p /workspace "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"
cd /workspace
if [ ! -d videoclean/.git ]; then
  git clone --depth 1 --branch main https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main

# Versions already on the image interpreter. uv keeps them, including
# torch==2.8.0+cu128, instead of downloading torch==2.8.0 from PyPI.
"$PY" - <<'PY' > /tmp/image-pins.txt
from importlib.metadata import distributions
seen = set()
for dist in distributions():
    name = dist.metadata.get("Name")
    ver = dist.version
    if not name or not ver or name.lower() in seen:
        continue
    seen.add(name.lower())
    print(f"{name}=={ver}")
PY

uv pip install --python "$PY" --system --break-system-packages \
  --overrides /tmp/image-pins.txt \
  -e ".[gpu,lama,web]"

if ! "$PY" -c 'import sam2' >/dev/null 2>&1; then
  uv pip install --python "$PY" --system --break-system-packages \
    --overrides /tmp/image-pins.txt \
    "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
fi

if [ ! -f server/static_dist/index.html ]; then
  if ! command -v bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash || echo "WARNING: bun install failed, UI will be unavailable"
  fi
  export PATH="/root/.bun/bin:${PATH}"
  if command -v bun >/dev/null 2>&1; then
    (cd webui && bun install --frozen-lockfile && bun run build) \
      || echo "WARNING: webui build failed, serving API only"
  fi
fi

"$PY" -m videoclean doctor --device cuda || true
exec "$PY" -m videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
