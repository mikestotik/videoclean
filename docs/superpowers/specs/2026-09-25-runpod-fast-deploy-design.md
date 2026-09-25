# RunPod fast deploy (dev CUDA loop)

Date: 2026-09-25  
Status: draft for review

## Intent

Frequent CUDA checks of videoclean on a RunPod RTX 4090 during active development. Deploy must be fast. Full CUDA/torch image builds are out of scope: use the existing base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

Usage pattern:

- Spin up a pod for a short verification session (often ~1 hour).
- Push many commits while the pod is alive; each push should update the running app without terminating the pod.
- Terminate the pod when stepping away for larger local work (cost control).
- Later start a new pod that reuses Network Volume state (venv, HF weights, data).

Success: push → hot update on a live pod in tens of seconds; cold start reuses volume caches and does not reinstall torch.

## Approach

**Network Volume + code sync over SSH.** No app image build for the RunPod path.

Existing `Dockerfile` / `Dockerfile.api` / `Dockerfile.web` stay for one-box / split production. RunPod-dev is a separate path.

## Architecture

```
GitHub push / workflow_dispatch
        │
        ▼
GitHub Actions
  1. bun build → server/static_dist
  2. RunPod REST: find RUNNING pod named videoclean-dev
  3. rsync app tree + static_dist over SSH
  4. remote: uv sync (skip torch) + soft restart serve
        │
        ▼
RunPod pod
  image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
  GPU: NVIDIA GeForce RTX 4090
  volume: Network Volume → /workspace
  ports: 7860/http, 22/tcp
```

### Volume layout

```
/workspace/
  videoclean/     # git clone of public repo (bootstrap) + rsync target
  .venv/          # UV_PROJECT_ENVIRONMENT
  uv-cache/       # UV_CACHE_DIR
  hf/             # HF_HOME
  data/           # VIDEOCLEAN_DATA_DIR
  run/            # pid / restart helpers
```

### Env on pod

- `UV_PROJECT_ENVIRONMENT=/workspace/.venv`
- `UV_CACHE_DIR=/workspace/uv-cache`
- `HF_HOME=/workspace/hf`
- `VIDEOCLEAN_DATA_DIR=/workspace/data`
- `VIDEOCLEAN_PORT=7860`
- `VIDEOCLEAN_AUTH=off` (dev loop; pod URL is obscure; can tighten later)
- `SAM2_BUILD_CUDA=0`
- Torch/CUDA come from the base image; never reinstalled by uv.

## Components

### 1. `scripts/runpod-start.sh`

Cold-start entrypoint for the RunPod template / `dockerStartCmd`:

1. Create volume dirs if missing.
2. If `/workspace/videoclean/.git` missing → `git clone https://github.com/mikestotik/videoclean.git /workspace/videoclean`.
3. `git fetch origin && git reset --hard origin/main` (default branch; override via `VIDEOCLEAN_REF`).
4. Ensure `uv` on PATH (install to `/workspace/bin` once if needed).
5. `cd /workspace/videoclean && uv sync --extra gpu --extra lama --extra web --no-dev --no-install-package torch --no-install-package torchvision`.
6. Best-effort `videoclean doctor --device cuda`.
7. `exec` serve on `0.0.0.0:$VIDEOCLEAN_PORT`.

Web UI static: preferred path is CI-built `static_dist` already present after first Actions deploy. On pure manual cold start before any Actions run, start script may skip UI or build later; first `deploy` workflow fills `server/static_dist`.

### 2. `scripts/runpod-restart.sh`

Hot restart after rsync:

1. `uv sync` with the same torch-skip flags (usually no-op).
2. Stop previous serve (pid file under `/workspace/run/` or `pkill` scoped to videoclean serve).
3. Start serve in background; write new pid; print proxy URL hint.

### 3. GitHub Actions `.github/workflows/runpod-deploy.yml`

Triggers:

- `push` to `main` (paths: library, server, webui, scripts, pyproject/lock) → **deploy if pod running**, else succeed with “pod offline” notice (no auto-start; saves money).
- `workflow_dispatch` with input `action`: `deploy` | `start` | `stop`.

Jobs:

| Action | Behavior |
|---|---|
| `deploy` | Build webui → locate RUNNING pod `videoclean-dev` → rsync → remote restart. Fail clearly if no running pod. |
| `start` | `POST https://rest.runpod.io/v1/pods` with image, RTX 4090, `networkVolumeId`, ports `7860/http,22/tcp`, name `videoclean-dev`, start cmd → wait until SSH up → deploy once. |
| `stop` | Terminate the named pod; volume kept. |

SSH: account-level RunPod SSH public key; Actions uses matching private key secret. Connection via `publicIp` + mapped port `22` from pod `portMappings`.

Rsync includes at least: `videoclean/`, `server/` (including `static_dist`), `pyproject.toml`, `uv.lock`, `scripts/`, `README.md`. Excludes `.venv`, `node_modules`, `.git` optional (volume keeps its own git for cold start; rsync of working tree is source of truth for hot updates).

### 4. Docs

Short `docs/RUNPOD.md`: one-time manual setup checklist, how to start/stop/deploy, expected times, troubleshooting (`doctor`, SSH, volume DC must match GPU DC).

## Data flow (hot update)

1. Developer pushes to `main`.
2. Actions builds `webui` → `server/static_dist`.
3. Actions lists pods; finds `videoclean-dev` RUNNING.
4. rsync over SSH updates `/workspace/videoclean`.
5. Remote restart runs `uv sync` + serve restart.
6. Browser hits RunPod HTTP proxy on port 7860.

## Error handling

- No running pod on push-deploy: workflow green with explicit summary “pod offline — start via workflow_dispatch or RunPod UI”.
- SSH not ready after start: retry with backoff (~2–3 min), then fail.
- `uv sync` failure: leave old process running if possible; fail the job.
- CUDA doctor failure on start: log and continue serve (weights may be missing until Config download).
- Wrong data center (volume vs GPU): start fails at API; doc warns to create volume in the DC where 4090s are rented.

## Out of scope

- Building/pushing a custom CUDA image for RunPod.
- Auto-starting a pod on every push (cost).
- Production auth, TLS, multi-user.
- Migrating production Dockerfiles to the RunPod base image.
- Pre-baking HF weights into any image (download via Config/API; cache on volume).

## What the human does once (manual)

1. Create a **Network Volume** in the RunPod DC where you rent RTX 4090 (size ~50–100 GB recommended).
2. Add an **SSH key** to the RunPod account; keep the private key for GitHub.
3. Create GitHub repo secrets/vars:
   - `RUNPOD_API_KEY`
   - `RUNPOD_NETWORK_VOLUME_ID`
   - `RUNPOD_SSH_PRIVATE_KEY`
   - `RUNPOD_POD_NAME`=`videoclean-dev` (or repo variable)
4. Optional: create a Pod **template** in UI with the pytorch image, volume, ports, start command — Actions can also pass the same fields on `start` without a saved template.
5. First session: run workflow `start` (or deploy pod in UI with the same settings), open `https://<pod-id>-7860.proxy.runpod.net`, download weights from Config if needed.

Everything else (scripts, workflow, docs in repo) is implemented in code.

## Testing / verification

- Dry-run: Actions `deploy` with no pod → “offline” message, no failure spam.
- Live: `start` → curl health/OpenAPI on 7860 proxy → `doctor` CUDA in logs.
- Hot path: two consecutive pushes while pod up → second restart serves new commit; UI static hash changes when webui changes.
- `stop` → pod gone; volume still has `hf/` and `.venv/`; next `start` skips heavy downloads.

## Implementation plan handoff

After this spec is approved, write a step-by-step plan covering: scripts, workflow YAML, docs, minimal secrets documentation, and a smoke checklist. No changes to core pipeline logic.
