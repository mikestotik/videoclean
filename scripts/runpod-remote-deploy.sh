#!/usr/bin/env bash
# Run ON the RunPod volume. Pulls latest git ref, builds webui, restarts serve.
# Called by GitHub Actions over SSH after every push (while the pod is RUNNING).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-env.sh
source "${SCRIPT_DIR}/runpod-env.sh"

REPO_URL="${VIDEOCLEAN_REPO_URL:-https://github.com/mikestotik/videoclean.git}"
REF="${VIDEOCLEAN_REF:-main}"

ensure_bun() {
  if command -v bun >/dev/null 2>&1; then
    return 0
  fi
  echo "installing bun into ${WORKSPACE}"
  curl -fsSL https://bun.sh/install | bash
  if [[ -x "${HOME}/.bun/bin/bun" ]]; then
    mkdir -p "${WORKSPACE}/bin"
    ln -sfn "${HOME}/.bun/bin/bun" "${WORKSPACE}/bin/bun"
  fi
  command -v bun >/dev/null 2>&1
}

if [[ ! -d "${APP_DIR}/.git" ]]; then
  git clone --branch "${REF}" "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"
echo "pulling origin/${REF}"
git fetch origin "${REF}"
git reset --hard "origin/${REF}"
echo "HEAD=$(git rev-parse --short HEAD)"

ensure_bun
cd "${APP_DIR}/webui"
bun install --frozen-lockfile
bun run build
test -f "${APP_DIR}/server/static_dist/index.html"
echo "UI build ok"

uv_sync_app
stop_serve
start_serve_background

echo "deployed $(git -C "${APP_DIR}" rev-parse --short HEAD)"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "UI: https://${RUNPOD_POD_ID}-${VIDEOCLEAN_PORT}.proxy.runpod.net"
fi
