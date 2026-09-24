# RunPod

Web UI (FastAPI) on port **7860**. The image has CUDA torch, FFmpeg, and extras (`gpu`, `lama`, `web`). It does **not** bake Hugging Face weights, so the first `serve` is minutes after pull, not a multi-GB model download.

GPU: минимум **16 GB** (RTX 2000 Ada / RTX 4000 Ada) для `grounding-dino-tiny + sam2-tiny + lama`. ProPainter@1080p на 16 GB — на грани (OOM возможен), пробовать на коротком клипе; `sam2-large` на 16 GB не брать.

Auth is required: `VIDEOCLEAN_UI_USER` + `VIDEOCLEAN_UI_PASSWORD`. Serve refuses to start without a password.

## 0. Without publishing a Docker image (git clone + uv)

Use a stock RunPod **PyTorch** template, Python **3.11 or 3.12** (not 3.13), CUDA 12.4+. Official image example:

`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

Pod settings:

- GPU: RTX 2000 Ada / RTX 4000 Ada (16 GB), хватит; 24 GB для ProPainter без риска
- Expose **HTTP** port `7860` (not only TCP)
- Container disk: **40 GB**
- Network volume **50 GB+**, mount **`/workspace`**
- Env:

| Variable | Value |
|---|---|
| `VIDEOCLEAN_UI_USER` | `admin` |
| `VIDEOCLEAN_UI_PASSWORD` | a real password |
| `VIDEOCLEAN_PORT` | `7860` |
| `VIDEOCLEAN_DATA_DIR` | `/workspace/.videoclean` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `SAM2_BUILD_CUDA` | `0` |

Start command (clone once, then serve). Repo is public: `https://github.com/mikestotik/videoclean.git`.
Automated: workflow **Deploy PyTorch pod** (Actions → Run workflow) creates the pod
with `scripts/pytorch-start.sh` as the start command — same script as below.

```bash
bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export VIDEOCLEAN_PORT="${VIDEOCLEAN_PORT:-7860}"
export VIDEOCLEAN_DATA_DIR="${VIDEOCLEAN_DATA_DIR:-/workspace/.videoclean}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}"
export PATH="/root/.local/bin:/usr/local/bin:$PATH"

apt-get update
apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip python3.11 python3.11-venv || apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates zstd unzip

curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

mkdir -p /workspace
cd /workspace
if [ ! -d videoclean/.git ]; then
  git clone https://github.com/mikestotik/videoclean.git videoclean
fi
cd videoclean
git fetch --depth 1 origin main
git checkout -B main origin/main

uv python pin 3.11 || true
# --inexact keeps pip-installed extras (sam2 from git) across syncs.
uv sync --inexact --extra gpu --extra lama --extra web --no-dev --no-install-package torch --no-install-package torchvision
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv/bin/python "git+https://github.com/facebookresearch/sam2.git" hf-transfer matplotlib imageio
# Build tags (+cu128) always differ from the lockfile, so keep `uv run`
# from re-syncing the venv back to the locked CPU build. Copy instead of
# hardlinking (cache and .venv are on different filesystems).
export UV_NO_SYNC=1
export UV_LINK_MODE=copy

mkdir -p "$VIDEOCLEAN_DATA_DIR" "$HF_HOME"

# WebUI: build from source when missing (static_dist is NOT committed).
# Never fatal: the API works without UI.
if [ ! -f server/static_dist/index.html ]; then
  if ! command -v bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash || echo "WARNING: bun install failed, UI will be unavailable"
  fi
  export PATH="/root/.bun/bin:$PATH"
  if command -v bun >/dev/null 2>&1; then
    (cd webui && bun install && bun run build) || echo "WARNING: webui build failed, serving API only"
  fi
fi

uv run videoclean doctor --device cuda || true
exec uv run videoclean serve --host 0.0.0.0 --port "$VIDEOCLEAN_PORT"
'
```

> **Gate check (10 секунд, до долгой установки).** Стоковый torch образа должен видеть CUDA, иначе хост битый и всё остальное бессмысленно:
> ```bash
> /usr/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1
> ```
> `True` — едем дальше. `False` — terminate под и бери другой дата-центр, скрипт тут не поможет.
>
> **Network volumes (mfs) + SQLite.** `jobs.sqlite` не работает на сетевых томах (`disk I/O error`): держи `VIDEOCLEAN_DATA_DIR` на локальном диске контейнера (`/root/.videoclean`), а `HF_HOME` — на томе. Pod volumes — локальный диск, там всё ок.

First boot: several minutes (`uv sync` + CUDA torch + sam2). Weights are **not** downloaded here. After the UI is up, open `https://<POD_ID>-7860.proxy.runpod.net`, log in, **Конфиг**, download `grounding-dino`, `sam2-tiny`, then `propainter` for max quality. Ollama: if the process is up, models appear under LLM; pull a tag there (for example `llama3.2`).

Local LLM without Ollama will stay grey. Cloud LLM is disabled by default — everything runs locally.

## 1. GitHub Actions (образ + автодеплой пода)

1. GitHub secrets: `RUNPOD_API_KEY` (RunPod Console → Settings → API Keys),
   `VIDEOCLEAN_UI_PASSWORD`. Optional repo var: `VIDEOCLEAN_UI_USER` (default `admin`).
2. Workflow **Build image** (`workflow_dispatch` или push в `main`) собирает монолит
   (`Dockerfile`) для `linux/amd64` и пушит в GHCR:
   `ghcr.io/mikestotik/videoclean:runpod` (+ `:sha`).
3. **Сделать пакет Public** после первого пуша: страница пакета
   `ghcr.io/mikestotik/videoclean` → **Package settings** → **Change visibility** → **Public**.
   По умолчанию GHCR-пакет private (даже в public-репо) — RunPod тянет образ без авторизации.
4. По завершении build запускается **Deploy RunPod**: скрипт `scripts/runpod-deploy.sh`
   находит под по имени (`videoclean-test`) и, если найден, PATCH (новый образ, рестарт),
   иначе создаёт (POST `/v1/pods`). GPU/cloud/размеры — inputs workflow с дефолтами.
   Data-центр подбирается сам (или укажите `data_center_ids`).
5. Автодеплой после каждого build выключен: включите repo variable `AUTO_DEPLOY = true`,
   чтобы под рестартовался сам; иначе — только кнопкой (workflow_dispatch). Так push в `main`
   не прерывает работающие джобы.

`env` под заменяется целиком при деплое: ручные переменные пода (напр. `VIDEOCLEAN_API_TOKEN`)
будут перезаписаны (переживают только то, что в коде/script). Volume: дефолт `--volume 50`
создаёт **pod volume** — переживает рестарты и деплои, но **не переживает terminate пода**.
Чтобы веса пережили удаление пода, создайте network volume в консоли (Storage) и укажите
`network_volume_id` (скрипт сам подставит его datacenter в под).

Тот же скрипт для локального ручного деплоя:

```bash
bash scripts/runpod-deploy.sh --dry-run          # план без API
RUNPOD_API_KEY=... VIDEOCLEAN_UI_PASSWORD=... bash scripts/runpod-deploy.sh
```

## 2. Build and push the image (вручную)

From this repo (linux/amd64; RunPod cannot pull a Mac ARM image):

```bash
docker build --platform linux/amd64 -t mikestotik/videoclean:runpod .
docker push mikestotik/videoclean:runpod
```

## 3. Create a GPU pod

1. [RunPod console → Pods](https://www.runpod.io/console/pods) → **Deploy**.
2. GPU: **RTX 2000 Ada / RTX 4000 Ada** (16 GB) — минимум для полного стека в fp16; **RTX 4090** (24 GB) для ProPainter без риска OOM. CUDA 12.4 drivers are fine.
3. **Edit template**:
   - Container image: `ghcr.io/mikestotik/videoclean:runpod` (или `mikestotik/videoclean:runpod` для ручного пуша)
   - Expose **HTTP** port `7860` (not only TCP).
   - Container disk: **40 GB** (torch + extras; weights go on the volume).
   - Volume: **50 GB+** (models, jobs, uploads). Mount path **`/workspace`**.
   - Start command: leave empty (image `CMD` runs `scripts/start.sh` → `videoclean serve`).

## 4. Environment

| Variable | Value |
|---|---|
| `VIDEOCLEAN_UI_USER` | `admin` (or whatever you want) |
| `VIDEOCLEAN_UI_PASSWORD` | a real password — not `change-me` |
| `VIDEOCLEAN_PORT` | `7860` |
| `VIDEOCLEAN_DATA_DIR` | `/workspace/.videoclean` — jobs, uploads, ProPainter vendor+weights, LaMa `big-lama.pt` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `HF_HUB_CACHE` | optional; defaults to `$HF_HOME/hub` |
| `VIDEOCLEAN_PROPAINTER_ROOT` | optional override (else `$VIDEOCLEAN_DATA_DIR/vendor/ProPainter`) |
| `VIDEOCLEAN_PROPAINTER_WEIGHTS` | optional override (else `$VIDEOCLEAN_DATA_DIR/weights/propainter`) |
| `LAMA_MODEL` | optional path to `big-lama.pt` (else `$VIDEOCLEAN_DATA_DIR/weights/lama/big-lama.pt`) |

Point data + HF cache at `/workspace` so downloads survive pod stop/terminate when a network volume is attached. Defaults in the image are `/root/.videoclean` and `/root/.cache/huggingface` (container disk only). Catalog status honors `HF_HUB_CACHE` / `HF_HOME` (huggingface_hub), not only `~/.cache/huggingface`.

## 5. Start and open the UI

Deploy the pod. Logs should show `videoclean doctor --device cuda` then `videoclean serve`.

Proxy URL:

```
https://<POD_ID>-7860.proxy.runpod.net
```

Log in with the UI user/password (HTTP Basic). There is no model fetch until you click Download on **Конфиг**.

API for other services (Swagger: `/api/docs`, ReDoc: `/api/redoc`):

1. `POST /api/jobs` multipart: `video` + `prompt` (optional `formats=mp4,webm`, `webhook_url`, `webhook_secret`, `profile`).
2. Poll `GET /api/jobs/{id}` and/or wait for the terminal webhook (`COMPLETED` / `FAILED`).
3. Download `GET /api/jobs/{id}/output` (use `?fmt=` when multiple outputs).

Auth: `Authorization: Bearer $VIDEOCLEAN_API_TOKEN` (or the UI password if the token is unset). Absolute webhook URLs need `VIDEOCLEAN_PUBLIC_BASE_URL` (this proxy URL). Index: `GET /api`.

## 6. Config — download order

**Minimum path** (short clip):

1. `detector:grounding-dino` (~650 MB)
2. `segmenter:sam2-tiny` (~160 MB)
3. `inpainter:lama` (`big-lama.pt`)

Then **Clean** works: grounding-dino + sam2 + lama.

**Max quality on 4090** (profile «Качество» / propainter):

4. `inpainter:propainter` (~400 MB + git clone of vendor)
5. Last / hungry: `segmenter:sam2-large` (~900 MB weights, ~24 GB VRAM — easy OOM on 4090; prefer tiny)

Ollama rows stay unavailable unless you run Ollama yourself. Cloud LLM is disabled by default.

Do not start a second cleanup while one is RUNNING — it queues FIFO.

## 7. Workspace

Upload a short mp4, prompt required (e.g. `remove the channel logo`). Device should default to `cuda`. Run on the workspace screen. Download the output when state is `COMPLETED`.

## Local GPU (`docker compose`)

Copy `.env.example` to `.env` and set a password. Compose reads `env_file: .env`. Then:

```bash
docker compose up --build
```

Open `http://127.0.0.1:7860`. Compose mounts named volumes for `/root/.videoclean` and `/root/.cache/huggingface` and passes `gpus: all`.

## Notes

- `videoclean serve` binds `0.0.0.0` and reads `VIDEOCLEAN_PORT` (RunPod `PORT` is also accepted by `scripts/start.sh`).
- Image override: `torch==2.8.0` + `torchvision==0.23.0` from the cu128 wheel index (pyproject pins the same versions for CPU/Mac).
- `sam2` Python package is installed with `SAM2_BUILD_CUDA=0` (no nvcc in the runtime image). Weights still come from the Models tab.
- After a kill/restart, orphan RUNNING jobs are marked FAILED (`interrupted`).
