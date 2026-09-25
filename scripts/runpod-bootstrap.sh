#!/usr/bin/env bash
# First-boot entrypoint for empty Network Volume: clone repo, then cold-start.
# Used as dockerStartCmd so the pod does not need scripts already on the volume.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${APP_DIR:-${WORKSPACE}/videoclean}"
REPO_URL="${VIDEOCLEAN_REPO_URL:-https://github.com/mikestotik/videoclean.git}"
VIDEOCLEAN_REF="${VIDEOCLEAN_REF:-main}"

mkdir -p "${WORKSPACE}"

if [[ ! -d "${APP_DIR}/.git" ]]; then
  echo "cloning ${REPO_URL} → ${APP_DIR}"
  git clone --branch "${VIDEOCLEAN_REF}" "${REPO_URL}" "${APP_DIR}"
else
  echo "repo already present at ${APP_DIR}"
fi

exec bash "${APP_DIR}/scripts/runpod-start.sh"
