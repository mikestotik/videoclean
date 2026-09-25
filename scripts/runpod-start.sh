#!/usr/bin/env bash
# Cold start on a RunPod volume: sync deps (skip torch), serve in foreground.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-env.sh
source "${SCRIPT_DIR}/runpod-env.sh"

REPO_URL="${VIDEOCLEAN_REPO_URL:-https://github.com/mikestotik/videoclean.git}"

if [[ ! -d "${APP_DIR}/.git" ]]; then
  echo "cloning ${REPO_URL} → ${APP_DIR}"
  git clone --branch "${VIDEOCLEAN_REF}" "${REPO_URL}" "${APP_DIR}"
else
  echo "fetching ${VIDEOCLEAN_REF}"
  cd "${APP_DIR}"
  git fetch origin "${VIDEOCLEAN_REF}"
  git reset --hard "origin/${VIDEOCLEAN_REF}"
fi

# Ensure apt tools that the base image may lack (best-effort).
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git curl ca-certificates \
    || echo "warn: could not apt-install ffmpeg" >&2
fi

uv_sync_app
stop_serve
start_serve_background
hold_container
