# RunPod (RTX 4090)

Gradio UI on port **7860**. The image has CUDA torch, FFmpeg, and extras (`gpu`, `lama`, `web`). It does **not** bake Hugging Face weights, so the first `serve` is minutes after pull — not a multi-GB model download.

Auth is required: `VIDEOCLEAN_UI_USER` + `VIDEOCLEAN_UI_PASSWORD`. Serve refuses to start without a password.

## 1. Build and push the image

From this repo (linux/amd64; RunPod cannot pull a Mac ARM image):

```bash
docker build --platform linux/amd64 -t YOUR_DOCKERHUB_USER/videoclean:runpod .
docker push YOUR_DOCKERHUB_USER/videoclean:runpod
```

## 2. Create a GPU pod

1. [RunPod console → Pods](https://www.runpod.io/console/pods) → **Deploy**.
2. GPU: **RTX 4090** (24 GB). CUDA 12.4 drivers are fine.
3. **Edit template**:
   - Container image: `YOUR_DOCKERHUB_USER/videoclean:runpod`
   - Expose **HTTP** port `7860` (not only TCP).
   - Container disk: **40 GB** (torch + extras; weights go on the volume).
   - Volume: **50 GB+** (models, jobs, uploads). Mount path **`/workspace`**.
   - Start command: leave empty (image `CMD` runs `scripts/start.sh` → `videoclean serve`).

## 3. Environment

| Variable | Value |
|---|---|
| `VIDEOCLEAN_UI_USER` | `admin` (or whatever you want) |
| `VIDEOCLEAN_UI_PASSWORD` | a real password — not `change-me` |
| `VIDEOCLEAN_PORT` | `7860` |
| `VIDEOCLEAN_DATA_DIR` | `/workspace/.videoclean` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `XAI_API_KEY` | optional cloud LLM |
| `OPENAI_API_KEY` | optional |

Point data + HF cache at `/workspace` so downloads survive pod stop/terminate when a network volume is attached. Defaults in the image are `/root/.videoclean` and `/root/.cache/huggingface` (container disk only).

## 4. Start and open the UI

Deploy the pod. Logs should show `videoclean doctor --device cuda` then `videoclean serve`.

Proxy URL:

```
https://<POD_ID>-7860.proxy.runpod.net
```

Log in with the UI user/password. First load can take a minute while Gradio starts; there is no model fetch until you click Download.

## 5. Models tab — download order

`opencv-telea` is always ready. Everything else is opt-in.

**Minimum path** (short clip, CPU-style quality on GPU):

1. `detector:grounding-dino` (~650 MB)
2. `segmenter:sam2-tiny` (~160 MB)

Then **Clean** works: grounding-dino + sam2 + opencv-telea.

**Max quality on 4090** (enable the Max quality preset):

3. `inpainter:propainter` (~400 MB + git clone of vendor)
4. Optional: `inpainter:lama`, `detector:owlvit`
5. Last / hungry: `segmenter:sam2-large` (~900 MB weights, ~24 GB VRAM — easy OOM on 4090; prefer tiny)

Ollama rows stay unavailable unless you run Ollama yourself. Cloud LLM: set `XAI_API_KEY` or `OPENAI_API_KEY` and pick `cloud` on Clean.

Do not start a second cleanup while one is RUNNING — it queues FIFO.

## 6. Clean

Upload a short mp4, prompt required (e.g. `remove the channel logo`). Device should default to `cuda`. Submit → Jobs tab. Download the output when COMPLETED.

## Local GPU (`docker compose`)

Copy `.env.example` to `.env` and set a password. Compose reads `env_file: .env`. Then:

```bash
docker compose up --build
```

Open `http://127.0.0.1:7860`. Compose mounts named volumes for `/root/.videoclean` and `/root/.cache/huggingface` and passes `gpus: all`.

## Notes

- `videoclean serve` binds `0.0.0.0` and reads `VIDEOCLEAN_PORT` (RunPod `PORT` is also accepted by `scripts/start.sh`).
- Image override: `torch>=2.5` from the cu124 wheel index (pyproject still pins `2.2.2` for CPU/Mac).
- `sam2` Python package is installed with `SAM2_BUILD_CUDA=0` (no nvcc in the runtime image). Weights still come from the Models tab.
- After a kill/restart, orphan RUNNING jobs are marked FAILED (`interrupted`).
