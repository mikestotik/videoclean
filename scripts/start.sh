#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIDEOCLEAN_PORT:-}" && -n "${PORT:-}" ]]; then
  export VIDEOCLEAN_PORT="${PORT}"
fi
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/root/.videoclean}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

echo "videoclean data=${VIDEOCLEAN_DATA_DIR} hf=${HF_HOME} port=${VIDEOCLEAN_PORT}"
echo "HF weights are not in the image; download them on the Config page after login."

if ! command -v videoclean >/dev/null 2>&1; then
  echo "videoclean not on PATH" >&2
  exit 1
fi

echo "=== videoclean doctor --device cuda ==="
videoclean doctor --device cuda || true

echo "=== videoclean serve --host 0.0.0.0 --port ${VIDEOCLEAN_PORT} ==="
exec videoclean serve --host 0.0.0.0 --port "${VIDEOCLEAN_PORT}"
