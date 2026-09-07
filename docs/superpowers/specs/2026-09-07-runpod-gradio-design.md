# videoclean: Gradio UI + RunPod (RTX 4090)

Date: 2026-09-07  
Status: ready for review

## Goal

Поднять `videoclean` на RunPod (RTX 4090, CUDA) так, чтобы:

1. контейнер стартовал быстро (без обязательной загрузки тяжёлых весов);
2. из браузера можно было качать модели с прогрессом, ставить cleanup-задачи в очередь, смотреть Jobs и перезапускать зависшие;
3. CLI и существующий пайплайн остались рабочими;
4. известные баги, мешающие max-стеку на CUDA, были поправлены.

## Decisions (locked)

| Тема | Решение |
|---|---|
| UI | Gradio 4.x, три вкладки: Clean / Models / Jobs |
| Auth | `VIDEOCLEAN_UI_USER` + `VIDEOCLEAN_UI_PASSWORD` (Gradio auth) |
| LLM | `auto` / `local` / `cloud` в UI; Ollama в образе (опциональный сервис); cloud через env keys |
| Старт | образ с кодом + CUDA deps + FFmpeg; веса не в bake-слое |
| Модели | ручной download с прогрессом; недоступные бэкенды disabled в селектах |
| GPU jobs | одна running cleanup-задача; остальные QUEUED (FIFO) |
| Downloads | отдельная очередь на диск/сеть; параллельно с cleanup разрешено (inference VRAM не трогают) |
| Deploy | Dockerfile + `docker-compose`/`start.sh` + `docs/RUNPOD.md` |
| Порт | `7860` (Gradio), env `VIDEOCLEAN_PORT` |

## Architecture

Существующие слои (`domain` / `application` / `adapters` / `composition` / `cli`) сохраняются.

Добавляется:

```
videoclean/
  application/
    ports/
      progress.py          # уже есть; web-адаптер пишет в JobStore
      model_catalog.py     # NEW: статус/скачивание компонентов
      job_control.py       # NEW: list/cancel/retry/delete + queue
    use_cases/
      run_cleanup.py       # as-is + мелкие фиксы
      download_component.py# NEW
      manage_jobs.py       # NEW
    jobs/
      worker.py            # NEW: single-worker queue loop
  adapters/
    web/
      gradio_app.py        # NEW: UI
      progress_bridge.py   # NEW: ProgressPort → sqlite/events
    models/
      catalog.py           # NEW: registry of downloadable units
      downloaders.py       # NEW: HF / git clone / ollama pull
  cli.py                   # + `videoclean serve`
```

Правило зависимостей: `application` не импортирует `adapters` (сейчас `run_cleanup.py` тянет `adapters.prompt.frames` — это баг архитектуры, перенести `sample_frame_indices` в `application/` или `domain/`).

Web-слой только собирает request → `JobQueue.submit` / `DownloadService` / `JobIndex`. Пайплайн тот же, что у CLI.

## Data model

Расширить sqlite (`~/.videoclean/jobs.sqlite` или `VIDEOCLEAN_DATA_DIR`):

**jobs** (существующая + поля):

| column | notes |
|---|---|
| id | as now |
| created_at | as now |
| updated_at | NEW |
| state | QUEUED \| RUNNING \| COMPLETED \| FAILED \| CANCELLED (COMPLETED = как в CLI report) |
| input_path, output_path, prompt | as now |
| report_json | as now |
| request_json | NEW: полный PipelineConfig + paths для Retry |
| progress_json | NEW: stage, fraction, detail, heartbeat_at |
| error | NEW: короткий текст |
| pid / cancel_flag | NEW optional: cooperative cancel |

**downloads** (новая таблица):

| column | notes |
|---|---|
| id | uuid |
| component_id | e.g. `detector:grounding-dino` |
| state | queued \| running \| done \| failed \| cancelled |
| progress | 0..1 |
| bytes_done / bytes_total | nullable |
| message | text |
| updated_at | iso |

Файлы джобы: `DATA_DIR/jobs/<id>/` как сейчас (`JobPaths`). Uploads: `DATA_DIR/uploads/`. Outputs доступны для скачивания из UI.

## UI

### Tab Clean

- Video upload (mp4/mov/mkv/webm), text prompt (required).
- Device: default `cuda` если `torch.cuda.is_available()`, иначе `cpu`.
- Detector / segmenter / inpainter / llm place / llm model / format.
- Advanced accordion: threshold, mask dilate, telea radius, verify, prompt-frame-stride/max, overwrite.
- Preset button **Max quality**: device=cuda, detector=grounding-dino, segmenter=sam2-video, inpainter=propainter. Активен только если все три компонента `ready`.
- Селекты показывают только `ready` варианты (+ всегда `opencv-telea`). Под селектом список missing: «чтобы включить sam2-video → Models».
- Если Gradio позволит per-choice disable без ломки UX — использовать; иначе filtered dropdown (предпочтительно предсказуемое поведение).
- Submit → создаёт job `QUEUED`, показывает job_id и ссылку «смотреть в Jobs».
- Live panel текущей running-задачи: стадии как в CLI (`STAGES` из `progress.py`), % , ETA.

### Tab Models

Каталог компонентов (фиксированный registry):

| id | what |
|---|---|
| `detector:grounding-dino` | HF `IDEA-Research/grounding-dino-tiny` |
| `detector:owlvit` | HF `google/owlvit-base-patch32` |
| `segmenter:sam2-tiny` | HF `facebook/sam2-hiera-tiny` |
| `segmenter:sam2-large` | HF `facebook/sam2-hiera-large` |
| `inpainter:lama` | `simple-lama` + `big-lama.pt` |
| `inpainter:propainter` | git clone vendor + HF weights → `~/.videoclean/...` |
| `llm:ollama-llama3.2` | `ollama pull llama3.2` |
| `llm:ollama-llava-phi3` | `ollama pull llava-phi3` (vision parse) |

Для каждой строки: status badge, size hint, Download, Cancel. Общий progress bar активного download. Кнопка **Refresh status** + блок doctor (ffmpeg/torch/cuda).

Download никогда не блокирует старт контейнера и не обязателен при `serve`.

### Tab Jobs

Таблица всех jobs (limit 100, фильтр по state).

Колонки: id, state, prompt (truncate), created, updated, progress, device/backends summary.

Actions:

- **Cancel** — для QUEUED удаляет из очереди; для RUNNING ставит cancel flag (cooperative: между стадиями / на tick progress). Если процесс не отвечает N минут — Mark failed.
- **Retry** — клонирует `request_json` в новую job (тот же input файл на диске).
- **Delete** — удаляет row + `jobs/<id>/` (+ optional upload если никто не ссылается).
- **Download output** — если COMPLETED.
- **Mark failed** — для RUNNING без heartbeat дольше `VIDEOCLEAN_STALE_SECONDS` (default 900).
- Auto-refresh каждые 2–3 с (Gradio timer).

Пустые состояния и ошибки — человекочитаемый текст из `PipelineError` / `AdapterUnavailable`, не traceback в UI (traceback в `logs/`).

## Job worker

Один фоновый поток (или `threading` + queue) внутри процесса `videoclean serve`:

1. Берёт oldest QUEUED.
2. Ставит RUNNING, heartbeat в `progress_json`.
3. Строит `RunCleanup` через `composition.build_run_cleanup` с `ProgressBridge`.
4. По успеху COMPLETED + пути outputs; по ошибке FAILED; по cancel CANCELLED.
5. Следующая из очереди.

Cleanup и model-download: downloads пишут на диск и не конкурируют за inference; если download идёт во время cleanup — ок. Запрет: не стартовать второй cleanup.

Graceful shutdown: SIGTERM → дождаться текущей стадии или cancel, сохранить state.

## Model catalog / download

Порт `ModelCatalog`:

- `list() -> list[ComponentStatus]`
- `download(component_id, progress_cb) -> None`
- `cancel(download_id) -> None`

Реализация:

- HF: `huggingface_hub.snapshot_download` с `tqdm`/callback → progress.
- ProPainter vendor: `git clone --depth 1` если нет `VIDEOCLEAN_PROPAINTER_ROOT` / `~/.videoclean/vendor/ProPainter`.
- Ollama: HTTP `POST /api/pull` stream JSON (если `ollama` недоступен — status `unavailable: start ollama`).

`hf_cached()` уже есть; catalog использует его + adapter `.status()`.

## RunPod / Docker

Файлы:

- `Dockerfile` — nvidia CUDA runtime (12.4), Python 3.11, system ffmpeg, `uv sync --extra gpu --extra lama`, torch cu124 wheels, optional ollama install script.
- `docker-compose.yml` — service `videoclean` + optional `ollama`, volume `videoclean-data:/root/.videoclean`, volume HF cache, ports `7860:7860`, `gpus: all`.
- `scripts/start.sh` — doctor summary → `videoclean serve --host 0.0.0.0 --port ${PORT:-7860}`.
- `docs/RUNPOD.md` — пошагово: создать pod template, mount volume, env vars, открыть HTTP 7860, первый заход → Models → Download max stack → Clean.

Env:

```
VIDEOCLEAN_DATA_DIR=/root/.videoclean
VIDEOCLEAN_UI_USER=admin
VIDEOCLEAN_UI_PASSWORD=...
VIDEOCLEAN_PORT=7860
VIDEOCLEAN_STALE_SECONDS=900
XAI_API_KEY=...          # optional cloud LLM
OPENAI_API_KEY=...
HF_HOME=/root/.cache/huggingface
VIDEOCLEAN_PROPAINTER_ROOT=...
VIDEOCLEAN_PROPAINTER_WEIGHTS=...
```

Образ **не** качает HF веса на `docker build`. Первый `serve` поднимается за минуты после pull образа.

## CLI additions

```
videoclean serve [--host 0.0.0.0] [--port 7860]
videoclean models list
videoclean models download <component_id>
videoclean jobs cancel|retry|delete <id>
```

`serve` = Gradio + worker. Без UI-флагов поведение CLI `run` без изменений.

## Bugfix pass (in scope)

Обязательно до/вместе с UI:

1. **Architecture test fail**: `application` → `adapters.prompt.frames` — перенести sampling в application/domain.
2. **`cli.py`**: `PipelineConfig` в аннотации `_flags` не импортирован (работает только из‑за `from __future__ import annotations`) — поправить импорт.
3. **JobStore для web**: нет `updated_at`, `QUEUED`, cancel/retry, progress heartbeat — расширить schema с миграцией `CREATE/ALTER` идемпотентно.
4. **ProgressPort**: CLI `JobProgress` завязан на Rich Live; для web нужен bridge без TTY.
5. **CUDA/torch pin**: `pyproject` pin `torch==2.2.2`; README для CUDA/sam2-video требует torch≥2.5. В GPU-образе и docs явно: override torch/vision на cu124 ≥2.5 для sam2-video+propainter; `doctor` предупреждает о несовместимости.
6. **ProPainter paths / download**: единый путь через catalog; `allow_download` только из Models tab / CLI models download, не из случайного Clean run (Clean по умолчанию `allow_download=False`).
7. **Stale RUNNING** после kill контейнера: при старте `serve` помечать orphan RUNNING → FAILED с reason `interrupted`.
8. Пройти `pytest`; добавить тесты: queue ordering, cancel queued, catalog status without network, schema migration.

Вне scope: новые detector/segmenter модели, биллинг, multi-user ACL, Kubernetes.

## Testing

- Unit: catalog status mocks, job queue FIFO, retry clones request, orphan recovery.
- Integration (optional GPU marker): skip на CI без CUDA.
- Manual на RunPod: cold start → UI login → download grounding-dino+sam2-tiny+telea path → short clip → затем propainter/sam2-video.

## Success criteria

1. `docker compose up` (или RunPod template) даёт Gradio login < 5 минут после pull, без обязательных GB download.
2. Models tab качает компонент с видимым progress; после ready опция в Clean включается.
3. Две подряд Clean-задачи: вторая QUEUED, стартует после первой.
4. Jobs: видно состояние, Cancel/Retry/Delete/Download работают; после kill+restart orphan не висит RUNNING вечно.
5. Max quality path на 4090: grounding-dino + sam2-video + propainter + local или cloud LLM отрабатывает example clip.
6. `pytest` зелёный.
