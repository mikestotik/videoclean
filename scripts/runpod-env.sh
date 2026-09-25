#!/usr/bin/env bash
# Shared env for RunPod volume layout. Source from other runpod-*.sh scripts.
set -euo pipefail

export WORKSPACE="${WORKSPACE:-/workspace}"
export APP_DIR="${APP_DIR:-${WORKSPACE}/videoclean}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE}/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE}/uv-cache}"
export HF_HOME="${HF_HOME:-${WORKSPACE}/hf}"
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-${WORKSPACE}/data}"
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
export VIDEOCLEAN_AUTH="${VIDEOCLEAN_AUTH:-off}"
export VIDEOCLEAN_REF="${VIDEOCLEAN_REF:-main}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export SAM2_GIT_REF="${SAM2_GIT_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
export RUN_DIR="${RUN_DIR:-${WORKSPACE}/run}"
export PID_FILE="${PID_FILE:-${RUN_DIR}/videoclean.pid}"
export PATH="${WORKSPACE}/bin:${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"

mkdir -p \
  "${WORKSPACE}/bin" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${VIDEOCLEAN_DATA_DIR}" \
  "${RUN_DIR}"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  echo "installing uv into ${WORKSPACE}/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORKSPACE}/bin" sh
}

uv_sync_app() {
  ensure_uv
  cd "${APP_DIR}"
  if [[ ! -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
    uv venv --system-site-packages "${UV_PROJECT_ENVIRONMENT}"
  fi
  # Torch/CUDA stay on the RunPod base image; do not reinstall into the venv.
  uv sync --extra gpu --extra lama --extra web --no-dev \
    --no-install-package torch --no-install-package torchvision
  # sam2-video path (same pin as Dockerfile.api)
  if ! "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import sam2" >/dev/null 2>&1; then
    uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
      "git+https://github.com/facebookresearch/sam2.git@${SAM2_GIT_REF}"
  fi
}

stop_serve() {
  if [[ -f "${PID_FILE}" ]]; then
    local old
    old="$(cat "${PID_FILE}" || true)"
    if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
      echo "stopping serve pid=${old}"
      kill "${old}" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "${old}" 2>/dev/null || break
        sleep 1
      done
      kill -9 "${old}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  # Fallback if pid file was lost
  pkill -f "videoclean serve" 2>/dev/null || true
}

start_serve_background() {
  cd "${APP_DIR}"
  echo "=== videoclean doctor --device cuda ==="
  videoclean doctor --device cuda || true
  echo "=== videoclean serve (background) port=${VIDEOCLEAN_PORT} ==="
  nohup videoclean serve --host 0.0.0.0 --port "${VIDEOCLEAN_PORT}" \
    >"${RUN_DIR}/serve.log" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "started pid=$(cat "${PID_FILE}") log=${RUN_DIR}/serve.log"
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "UI: https://${RUNPOD_POD_ID}-${VIDEOCLEAN_PORT}.proxy.runpod.net"
  fi
}

# Keep container PID 1 alive so hot-restart can replace the serve child.
hold_container() {
  echo "holding container; serve log → ${RUN_DIR}/serve.log"
  tail -F "${RUN_DIR}/serve.log"
}
