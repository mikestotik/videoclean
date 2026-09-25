#!/bin/bash
# Stable container command. The pod keeps this text; code updates come from git.
# Restart pulls main and runs scripts/pytorch-start.sh from that checkout.
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:${PATH}"
mkdir -p /workspace
cd /workspace
if [ ! -d videoclean/.git ]; then
  git clone --depth 1 --branch main https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main
exec bash scripts/pytorch-start.sh
