# RunPod

Web UI (FastAPI) on port **7860**. The image has CUDA torch, FFmpeg, and extras (`gpu`, `lama`, `web`). It does **not** bake Hugging Face weights, so the first `serve` is minutes after pull, not a multi-GB model download.

GPU: минимум **16 GB** (RTX 2000 Ada / RTX 4000 Ada) для `grounding-dino-tiny + sam2-tiny + lama`. ProPainter@1080p на 16 GB — на грани (OOM возможен), пробовать на коротком клипе; `sam2-large` на 16 GB не брать.

Auth is required: `VIDEOCLEAN_UI_USER` + `VIDEOCLEAN_UI_PASSWORD`. Serve refuses to start without a password.

## 0. Without publishing a Docker image (git clone + uv)

Use the stock RunPod **PyTorch 2.8** template. Image:

`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

That image is Ubuntu 24.04, CUDA 12.8.1, Python 3.12, and `torch 2.8.0+cu128` already on the system interpreter. The start script installs the app and sam2 into that interpreter. It does not create a venv and does not reinstall torch, torchvision, torchaudio, or numpy.

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
| `VIDEOCLEAN_DATA_DIR` | `/root/.videoclean` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `SAM2_BUILD_CUDA` | `0` |

Start command: `scripts/pytorch-start.sh` (the workflow inlines that file as container CMD and leaves the image entrypoint alone). Repo is public: `https://github.com/mikestotik/videoclean.git`. No custom image: packages already in the template stay at those versions.

Code updates: push `main`, then run **Deploy PyTorch pod** again with the same pod name. The workflow restarts that pod. It does not terminate it. Restart wipes the container disk and keeps `/workspace` (clone, venv, Hugging Face cache). The boot script pulls `main` and reinstalls only when the git revision changed. Terminate only when the GPU, image, or disk size must change. A pod created before the venv-on-volume change still has the old start command; terminate that one once.
Automated: workflow **Deploy PyTorch pod** (Actions → Run workflow).

> **Gate check (10 секунд, до долгой установки).** Стоковый torch образа должен видеть CUDA, иначе хост битый и всё остальное бессмысленно:
> ```bash
> /usr/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1
> ```
> `True` — едем дальше. `False` — terminate под и бери другой дата-центр, скрипт тут не поможет.
>
> **Network volumes (mfs) + SQLite.** `jobs.sqlite` не работает на сетевых томах (`disk I/O error`): держи `VIDEOCLEAN_DATA_DIR` на локальном диске контейнера (`/root/.videoclean`), а `HF_HOME` — на томе. Pod volumes — локальный диск, там всё ок.

First boot: several minutes (app deps + sam2; torch stays the image build). Weights are **not** downloaded here. After the UI is up, open `https://<POD_ID>-7860.proxy.runpod.net`, log in, **Конфиг**, download `grounding-dino`, `sam2-tiny`, then `propainter` for max quality. Ollama: if the process is up, models appear under LLM; pull a tag there (for example `llama3.2`).

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
| `VIDEOCLEAN_DATA_DIR` | `/root/.videoclean` — sqlite, пресеты, mmap кадров. Не сетевой том |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `HF_HUB_CACHE` | optional; defaults to `$HF_HOME/hub` |
| `VIDEOCLEAN_PROPAINTER_ROOT` | optional override (else `$VIDEOCLEAN_DATA_DIR/vendor/ProPainter`) |
| `VIDEOCLEAN_PROPAINTER_WEIGHTS` | optional override (else `$VIDEOCLEAN_DATA_DIR/weights/propainter`) |
| `LAMA_MODEL` | optional path to `big-lama.pt` (else `$VIDEOCLEAN_DATA_DIR/weights/lama/big-lama.pt`) |

`HF_HOME` на томе `/workspace`, чтобы веса пережили рестарт. `VIDEOCLEAN_DATA_DIR` остаётся на диске контейнера: sqlite и mmap кадров на сетевом томе не живут. Каталог моделей смотрит `HF_HUB_CACHE` / `HF_HOME`.

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
- The GHCR image installs `torch==2.8.0` + `torchvision==0.23.0` from the cu128 wheel index (pyproject pins the same versions for CPU/Mac). The stock PyTorch pod does not: it keeps the template's `+cu128` build.
- `sam2` Python package is installed with `SAM2_BUILD_CUDA=0` (no nvcc in the runtime image). Weights still come from the Models tab.
- After a kill/restart, orphan RUNNING jobs are marked FAILED (`interrupted`).
