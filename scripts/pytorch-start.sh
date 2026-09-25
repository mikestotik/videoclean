#!/bin/bash
# Boot script for the stock RunPod PyTorch 2.8 template
# (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# The workflow passes this file as container CMD. The image ENTRYPOINT
# (/opt/nvidia/nvidia_entrypoint.sh) stays in place so CUDA paths are set.
# Clone once, then serve. Idempotent: a restart with the same git revision
# skips apt, pip, and the WebUI build.
#
# Every distribution already on the image interpreter is pinned with
# --overrides. uv must not replace torch/torchvision/torchaudio/numpy
# (or anything else the image shipped) with the pyproject pins.
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

# The base image already ships ffmpeg, git, curl, zstd, unzip.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip
fi

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
  git clone --depth 1 --branch main https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main

REV="$(git rev-parse HEAD)"
STAMP="/workspace/videoclean/.installed-rev"
if [ "${VIDEOCLEAN_REINSTALL:-0}" != "1" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$REV" ]; then
  echo "revision $REV already installed; skipping pip and webui build"
else
  # Pin every installed distribution. Local versions (+cu128) are written
  # as installed so resolution cannot swap them for the pyproject pins
  # (torch==2.8.0, numpy<2, …).
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
    --extra gpu --extra lama --extra web \
    -e .
  # sam2 is not on the image. Pin matches the init_state copy in sam2_video.py.
  if ! "$PY" -c 'import sam2' >/dev/null 2>&1 || [ "${VIDEOCLEAN_REINSTALL:-0}" = "1" ]; then
    uv pip install --python "$PY" --system --break-system-packages \
      --overrides /tmp/image-pins.txt \
      "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
  fi

  mkdir -p "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"

  # WebUI: static_dist is gitignored. Build only on a new revision.
  # Never fatal: the API works without UI.
  if [ ! -f server/static_dist/index.html ]; then
    if ! command -v bun >/dev/null 2>&1; then
      curl -fsSL https://bun.sh/install | bash || echo "WARNING: bun install failed, UI will be unavailable"
    fi
    export PATH="/root/.bun/bin:$PATH"
    if command -v bun >/dev/null 2>&1; then
      (cd webui && bun install --frozen-lockfile && bun run build) || echo "WARNING: webui build failed, serving API only"
    fi
  fi

  if [ -f server/static_dist/index.html ]; then
    printf '%s\n' "$REV" > "$STAMP"
  fi
fi

mkdir -p "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"
"$PY" -m videoclean doctor --device cuda || true
exec "$PY" -m videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
