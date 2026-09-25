#!/usr/bin/env bash
# Hot restart after rsync from GitHub Actions: uv sync + background serve.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-env.sh
source "${SCRIPT_DIR}/runpod-env.sh"

uv_sync_app
stop_serve
start_serve_background
