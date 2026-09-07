# Gradio UI + RunPod Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Gradio web UI with job queue and model downloads, plus Docker/RunPod packaging for RTX 4090 CUDA testing, while fixing blockers for the max stack.

**Architecture:** Keep domain/application/adapters. Add JobIndex schema + single-worker queue, ModelCatalog downloads, Gradio tabs (Clean/Models/Jobs), and `videoclean serve`. Docker starts fast without baked weights.

**Tech Stack:** Python 3.11, Gradio 4.x, sqlite JobIndex, huggingface_hub, FFmpeg, PyTorch CUDA (image overrides to ≥2.5), optional Ollama, Docker nvidia runtime.

**Spec:** `docs/superpowers/specs/2026-09-07-runpod-gradio-design.md`

## Global Constraints

- Python `>=3.11,<3.13`; keep CLI `videoclean run` behavior.
- Clean runs use `allow_download=False`; downloads only via Models tab / `videoclean models download`.
- One RUNNING cleanup job at a time; FIFO QUEUED.
- Job states: `QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED`.
- Data dir: `VIDEOCLEAN_DATA_DIR` or `~/.videoclean`.
- UI auth: `VIDEOCLEAN_UI_USER` + `VIDEOCLEAN_UI_PASSWORD`.
- Port: `VIDEOCLEAN_PORT` default `7860`.
- `application/` must not import `videoclean.adapters` / `cli` / `composition`.
- Repo may have no git yet: Task 0 inits git; later commits assume it exists.
- Prefer TDD; run targeted pytest after each task.

## File map

| Path | Role |
|---|---|
| `videoclean/application/frames_sample.py` | `sample_frame_indices` (moved out of adapters) |
| `videoclean/adapters/prompt/frames.py` | Re-export + `bgr_to_jpeg` only |
| `videoclean/store.py` | Extended JobIndex + downloads table |
| `videoclean/application/ports/model_catalog.py` | Catalog protocol + dataclasses |
| `videoclean/application/ports/job_control.py` | Queue/control protocol types |
| `videoclean/application/jobs/worker.py` | Single-worker cleanup loop |
| `videoclean/application/use_cases/manage_jobs.py` | submit/cancel/retry/delete/orphan |
| `videoclean/application/use_cases/download_component.py` | Orchestrate one download |
| `videoclean/adapters/web/progress_bridge.py` | ProgressPort → JobIndex |
| `videoclean/adapters/web/gradio_app.py` | Gradio UI |
| `videoclean/adapters/models/catalog.py` | Component registry + status |
| `videoclean/adapters/models/downloaders.py` | HF / git / ollama / lama download |
| `videoclean/cli.py` | `serve`, `models`, jobs cancel/retry/delete; fix PipelineConfig import |
| `Dockerfile`, `docker-compose.yml`, `scripts/start.sh`, `docs/RUNPOD.md` | Deploy |
| `pyproject.toml` | `web` extra: gradio, huggingface_hub |

---

### Task 0: Init git (if missing)

**Files:**
- Create: `.gitignore` (if missing)

- [ ] **Step 1: Check git**

```bash
cd /Users/mikestotik/Projects/videoclean
git status 2>&1 | head -5
```

If not a repo:

```bash
git init
printf '%s\n' '.venv/' '__pycache__/' '*.pyc' '.pytest_cache/' '.videoclean/' 'output/' '.env' 'uv.lock' >> .gitignore
# keep uv.lock tracked if preferred — if file exists and team wants lockfile, remove that line
git add -A
git commit -m "chore: initial commit before Gradio/RunPod work"
```

Expected: `git status` clean or only new untracked after this plan starts.

---

### Task 1: Fix architecture import + PipelineConfig import

**Files:**
- Create: `videoclean/application/frames_sample.py`
- Modify: `videoclean/adapters/prompt/frames.py`
- Modify: `videoclean/application/use_cases/run_cleanup.py` (import path)
- Modify: `videoclean/cli.py` (import `PipelineConfig` + sample path ok via adapters re-export or application)
- Modify: `tests/test_prompt_frames.py`
- Test: `tests/test_architecture.py`, `tests/test_prompt_frames.py`

**Interfaces:**
- Produces: `videoclean.application.frames_sample.sample_frame_indices(n_frames: int, stride: int, max_frames: int) -> list[int]`
- Consumes: none new

- [ ] **Step 1: Write failing assertion that application no longer mentions adapters** (already fails)

Run: `uv run pytest tests/test_architecture.py::test_application_does_not_import_adapters_or_cli -v`  
Expected: FAIL mentioning `run_cleanup.py` / `videoclean.adapters`

- [ ] **Step 2: Move pure sampling into application**

Create `videoclean/application/frames_sample.py` with the current `sample_frame_indices` body from `adapters/prompt/frames.py` (no cv2/numpy needed for indices).

Update `adapters/prompt/frames.py`:

```python
from videoclean.application.frames_sample import sample_frame_indices

__all__ = ["sample_frame_indices", "bgr_to_jpeg"]
# keep bgr_to_jpeg implementation here
```

Update `run_cleanup.py` import to:

```python
from videoclean.application.frames_sample import sample_frame_indices
```

Update `tests/test_prompt_frames.py` to import from `videoclean.application.frames_sample`.

- [ ] **Step 3: Fix `cli.py` PipelineConfig import**

In `videoclean/cli.py` add `PipelineConfig` to the import from `videoclean.application.config`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_architecture.py tests/test_prompt_frames.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/frames_sample.py videoclean/adapters/prompt/frames.py \
  videoclean/application/use_cases/run_cleanup.py videoclean/cli.py tests/test_prompt_frames.py
git commit -m "fix: move sample_frame_indices into application layer"
```

---

### Task 2: Extend JobIndex schema

**Files:**
- Modify: `videoclean/store.py`
- Create: `tests/test_job_store.py`

**Interfaces:**
- Produces on `JobIndex`:
  - `upsert(..., request: dict | None = None, progress: dict | None = None, error: str | None = None, cancel_requested: bool | None = None)`
  - `list_jobs(limit: int = 100, state: str | None = None) -> list[sqlite3.Row]`
  - `get(job_id: str) -> sqlite3.Row | None`
  - `request_cancel(job_id: str) -> bool`
  - `is_cancel_requested(job_id: str) -> bool`
  - `mark_orphans_failed(reason: str = "interrupted") -> int`
  - `update_progress(job_id: str, progress: dict) -> None`
  - downloads: `upsert_download`, `get_download`, `list_downloads`, `request_download_cancel`
- Job row columns: `updated_at`, `request_json`, `progress_json`, `error`, `cancel_requested` (INTEGER 0/1)
- Downloads table per spec

- [ ] **Step 1: Write failing tests**

```python
# tests/test_job_store.py
from pathlib import Path
from videoclean.store import JobIndex

def test_schema_has_queue_fields(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("j1", "QUEUED", prompt="x", request={"device": "cuda"})
    row = idx.get("j1")
    assert row["state"] == "QUEUED"
    assert row["request_json"]
    assert row["updated_at"]

def test_mark_orphans_failed(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("a", "RUNNING")
    idx.upsert("b", "QUEUED")
    n = idx.mark_orphans_failed("interrupted")
    assert n == 1
    assert idx.get("a")["state"] == "FAILED"
    assert idx.get("b")["state"] == "QUEUED"

def test_cancel_flag(tmp_path: Path):
    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("c", "RUNNING")
    assert idx.request_cancel("c") is True
    assert idx.is_cancel_requested("c") is True
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
uv run pytest tests/test_job_store.py -v
```

- [ ] **Step 3: Implement schema migration in `JobIndex.__init__`**

Idempotent: `CREATE TABLE IF NOT EXISTS` then for each new column try `ALTER TABLE jobs ADD COLUMN ...` catching "duplicate column". Same for `downloads` table. Always set `updated_at` on upsert.

Keep existing `list_jobs` working; extend signature with optional `state`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_job_store.py tests/test_run_cleanup.py -v
```

- [ ] **Step 5: Commit**

```bash
git add videoclean/store.py tests/test_job_store.py
git commit -m "feat: extend JobIndex for queue, progress, downloads"
```

---

### Task 3: ProgressBridge + JobWorker + manage_jobs

**Files:**
- Create: `videoclean/adapters/web/progress_bridge.py`
- Create: `videoclean/application/jobs/__init__.py`
- Create: `videoclean/application/jobs/worker.py`
- Create: `videoclean/application/use_cases/manage_jobs.py`
- Create: `videoclean/application/ports/job_control.py`
- Test: `tests/test_job_worker.py`

**Interfaces:**
- `ProgressBridge(jobs: JobIndex, job_id: str)` implements `ProgressPort`; on start/tick/finish writes `progress_json` with `stage`, `fraction`, `detail`, `heartbeat_at` (use `STAGES` weights from `videoclean.progress` or duplicate weight map in bridge to avoid Rich dependency — prefer importing `STAGES` only from `progress.py`).
- `ManageJobs`:
  - `submit(request_dict, input_path, output_path, prompt) -> job_id` → state QUEUED
  - `cancel(job_id) -> None` (QUEUED→CANCELLED; RUNNING→cancel flag)
  - `retry(job_id) -> new_job_id`
  - `delete(job_id, data_dir) -> None`
  - `mark_failed(job_id, reason) -> None`
  - `recover_orphans() -> int`
- `JobWorker(data_dir, jobs, build_runner)`:
  - `start()` / `stop()` background thread
  - loop: claim oldest QUEUED → RUNNING → run cleanup with ProgressBridge → COMPLETED/FAILED/CANCELLED
  - before run check cancel; between stages if possible check `is_cancel_requested`
  - never start second cleanup (single thread)

- [ ] **Step 1: Failing tests for FIFO + cancel queued**

```python
# tests/test_job_worker.py
import time
from pathlib import Path
from videoclean.store import JobIndex
from videoclean.application.use_cases.manage_jobs import ManageJobs

class FakeRunner:
    def __init__(self):
        self.ran = []
    def execute(self, req, data_dir):
        self.ran.append(req.job_id)
        time.sleep(0.05)
        return {"jobId": req.job_id, "state": "COMPLETED", "outputs": {}}

def test_submit_queued(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({"device": "cpu"}, tmp_path / "in.mp4", tmp_path / "out.mp4", "remove logo")
    assert jobs.get(jid)["state"] == "QUEUED"

def test_cancel_queued(tmp_path: Path):
    jobs = JobIndex(tmp_path / "j.sqlite")
    mgr = ManageJobs(jobs)
    jid = mgr.submit({}, tmp_path / "a.mp4", tmp_path / "b.mp4", "x")
    mgr.cancel(jid)
    assert jobs.get(jid)["state"] == "CANCELLED"
```

Add worker test that processes two jobs in order with a fake `build_run_cleanup` injected.

- [ ] **Step 2: Run — FAIL**

```bash
uv run pytest tests/test_job_worker.py -v
```

- [ ] **Step 3: Implement ManageJobs, ProgressBridge, JobWorker**

`build_runner` callable: `(cfg, progress, jobs, job_id) -> RunCleanup` — worker deserializes `request_json` into `PipelineConfig` + `RunCleanupRequest` via helpers in `manage_jobs.py` or `composition` (worker may live in application and receive injected factory from composition/cli to avoid application→composition import).

Put factory wiring in `videoclean/composition.py`:

```python
def build_job_worker(data_dir: Path, jobs: JobIndex) -> JobWorker:
    def factory(cfg, progress, jobs, job_id):
        return build_run_cleanup(cfg, progress, jobs, job_id=job_id)
    return JobWorker(data_dir=data_dir, jobs=jobs, build_runner=factory)
```

- [ ] **Step 4: Tests PASS**

```bash
uv run pytest tests/test_job_worker.py tests/test_job_store.py -v
```

- [ ] **Step 5: Commit**

```bash
git add videoclean/adapters/web/progress_bridge.py videoclean/application/jobs \
  videoclean/application/use_cases/manage_jobs.py videoclean/application/ports/job_control.py \
  videoclean/composition.py tests/test_job_worker.py
git commit -m "feat: job queue worker and progress bridge"
```

---

### Task 4: Model catalog + downloaders

**Files:**
- Create: `videoclean/application/ports/model_catalog.py`
- Create: `videoclean/adapters/models/__init__.py`
- Create: `videoclean/adapters/models/catalog.py`
- Create: `videoclean/adapters/models/downloaders.py`
- Create: `videoclean/application/use_cases/download_component.py`
- Test: `tests/test_model_catalog.py`
- Modify: `pyproject.toml` — optional-dep `web` includes `gradio>=4,<6`, `huggingface_hub>=0.23`

**Interfaces:**
- `@dataclass ComponentInfo`: `id: str`, `title: str`, `kind: str`, `model_ref: str`, `size_hint: str`
- `@dataclass ComponentStatus`: `info: ComponentInfo`, `state: str` (`ready|missing|downloading|error`), `message: str`
- `ModelCatalog.list_status() -> list[ComponentStatus]`
- `ModelCatalog.is_ready(component_id: str) -> bool`
- `DownloadComponent.execute(component_id, jobs: JobIndex, progress_cb) -> None`
- Registry ids exactly as spec: `detector:grounding-dino`, `detector:owlvit`, `segmenter:sam2-tiny`, `segmenter:sam2-large`, `inpainter:lama`, `inpainter:propainter`, `llm:ollama-llama3.2`, `llm:ollama-llava-phi3`
- Mapping for UI selects: detector name → component id; `sam2` uses `segmenter:sam2-tiny` (or matching `segmenter_model`); `sam2-video` same weights id; `propainter` → `inpainter:propainter`; `lama` → `inpainter:lama`; `opencv-telea` always ready

- [ ] **Step 1: Failing tests (no network)**

```python
# tests/test_model_catalog.py
from videoclean.adapters.models.catalog import ModelCatalog, COMPONENT_IDS

def test_registry_ids():
    assert "detector:grounding-dino" in COMPONENT_IDS
    assert "inpainter:propainter" in COMPONENT_IDS

def test_telea_always_ready_helper():
    from videoclean.adapters.models.catalog import backend_ready
    assert backend_ready("inpainter", "opencv-telea") is True

def test_list_status_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cat = ModelCatalog()
    rows = cat.list_status()
    assert len(rows) >= 6
    assert all(r.state in {"ready", "missing", "downloading", "error"} for r in rows)
```

- [ ] **Step 2: Run — FAIL**

```bash
uv run pytest tests/test_model_catalog.py -v
```

- [ ] **Step 3: Implement catalog using `hf_cached`, ProPainter `find_vendor`/`find_weights`, lama `find_weights`, Ollama HTTP `GET /api/tags`**

Downloaders:

- HF: `snapshot_download(repo_id=..., local_files_only=False)` with tqdm callback writing JobIndex download progress
- ProPainter: clone `https://github.com/sczhou/ProPainter.git` to `~/.videoclean/vendor/ProPainter` if missing, then snapshot weights to `~/.videoclean/weights/propainter`
- LaMa: curl/urllib to known `LAMA_MODEL_URL` from lama adapter (import URL constant or duplicate)
- Ollama: stream `POST http://127.0.0.1:11434/api/pull` JSON lines

Cancel: set download `cancel_requested`; downloaders check between chunks.

- [ ] **Step 4: Tests PASS + add web extra**

```bash
uv add --optional web 'gradio>=4,<6' 'huggingface_hub>=0.23'
uv run pytest tests/test_model_catalog.py -v
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock videoclean/adapters/models videoclean/application/ports/model_catalog.py \
  videoclean/application/use_cases/download_component.py tests/test_model_catalog.py
git commit -m "feat: model catalog and downloaders"
```

---

### Task 5: Gradio app (Clean / Models / Jobs)

**Files:**
- Create: `videoclean/adapters/web/__init__.py`
- Create: `videoclean/adapters/web/gradio_app.py`
- Create: `videoclean/adapters/web/app_state.py` (holds data_dir, jobs, worker, catalog, manage)
- Test: `tests/test_gradio_wiring.py` (logic helpers, not full browser)

**Interfaces:**
- `build_ui(state: AppState) -> gr.Blocks`
- `launch_ui(state, host, port, auth: tuple[str,str] | None)`
- Helpers (unit-tested):
  - `ready_choices(port: str, catalog) -> list[str]`
  - `serialize_clean_form(...) -> dict` request_json for ManageJobs.submit
  - `format_jobs_table(rows) -> list[list]`

UI behavior per spec:
- Tab Clean: upload, prompt, filtered selects, Max quality preset, submit → QUEUED, show job id + live progress of active job
- Tab Models: dataframe/status + Download buttons per component + refresh + doctor text
- Tab Jobs: table, filter, Cancel/Retry/Delete/Mark failed/Download output; Timer refresh 2s
- Auth from env when both user+password set; if missing password, refuse launch with clear error (safer default for RunPod)

- [ ] **Step 1: Helper tests**

```python
# tests/test_gradio_wiring.py
from videoclean.adapters.web.gradio_app import ready_choices

class FakeCat:
    def is_ready(self, cid: str) -> bool:
        return cid in {"detector:grounding-dino", "inpainter:propainter"}

def test_ready_detector_choices():
    assert ready_choices("detector", FakeCat()) == ["grounding-dino"]
```

- [ ] **Step 2: Implement Gradio Blocks**

Keep UI code readable: functions per tab. Save uploads under `data_dir/uploads/<job_id>/input`. Output default `data_dir/jobs/<job_id>/output/cleaned.mp4`.

Worker must already be `start()`ed by caller before launch.

- [ ] **Step 3: Tests**

```bash
uv sync --extra web
uv run pytest tests/test_gradio_wiring.py -v
```

- [ ] **Step 4: Manual smoke (local, no GPU ok)**

```bash
VIDEOCLEAN_UI_USER=admin VIDEOCLEAN_UI_PASSWORD=admin \
  uv run videoclean serve --host 127.0.0.1 --port 7860
```

(Command added in Task 6 — if serve not yet wired, `python -c` launch from `gradio_app` temporarily.)

- [ ] **Step 5: Commit**

```bash
git add videoclean/adapters/web tests/test_gradio_wiring.py
git commit -m "feat: Gradio Clean/Models/Jobs UI"
```

---

### Task 6: CLI `serve` + models + job control commands

**Files:**
- Modify: `videoclean/cli.py`
- Modify: `videoclean/composition.py` (build helpers if not done)
- Test: `tests/test_cli_serve_helpers.py` (pure helpers) or invoke Typer runner lightly

**Interfaces:**
- `videoclean serve --host 0.0.0.0 --port 7860`
  - resolve data_dir
  - `JobIndex(...); mark_orphans_failed(); worker.start(); build_ui; launch`
- `videoclean models list`
- `videoclean models download COMPONENT_ID`
- `videoclean jobs cancel|retry|delete JOB_ID`

- [ ] **Step 1: Add commands to cli.py** following existing Typer style (`jobs_app`, new `models_app`)

Serve body sketch:

```python
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(int(__import__("os").environ.get("VIDEOCLEAN_PORT", "7860")), "--port"),
) -> None:
    import os
    from videoclean.adapters.web.gradio_app import launch_from_env
    launch_from_env(host=host, port=port, data_dir=data_dir())
```

Prefer implementing `launch_from_env` in `gradio_app.py` / `app_state.py` that reads auth env and starts worker.

- [ ] **Step 2: doctor note for torch&lt;2.5 when segmenter sam2-video**

In `doctor_sections`, if cfg.segmenter == `sam2-video` and torch version parsed &lt; 2.5, append warning line.

- [ ] **Step 3: Run unit tests + help**

```bash
uv run videoclean serve --help
uv run videoclean models --help
uv run pytest -q
```

Expected: full suite green (or only skipped GPU).

- [ ] **Step 4: Commit**

```bash
git add videoclean/cli.py videoclean/composition.py
git commit -m "feat: videoclean serve and models CLI"
```

---

### Task 7: Docker + RunPod docs

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Create: `scripts/start.sh`
- Create: `docs/RUNPOD.md`
- Modify: `README.md` (short section linking RunPod + `serve`)

**Dockerfile sketch:**

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# install uv, copy project, uv sync --extra gpu --extra lama --extra web
# pip/uv install torch torchvision from cu124 index (torch>=2.5)
ENV VIDEOCLEAN_DATA_DIR=/root/.videoclean
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 7860
CMD ["scripts/start.sh"]
```

`start.sh`: print `videoclean doctor --device cuda` summary; exec `videoclean serve`.

`docker-compose.yml`: gpus all, ports 7860, volumes for data + HF cache, env_file `.env.example`.

`docs/RUNPOD.md`: create GPU pod, select 4090, expose 7860, set env password, attach volume, pull/start, open proxy URL, Models download order (dino → sam2-tiny → … → propainter), then Clean.

`.env.example`:

```
VIDEOCLEAN_UI_USER=admin
VIDEOCLEAN_UI_PASSWORD=change-me
XAI_API_KEY=
```

- [ ] **Step 1: Write files above**
- [ ] **Step 2: Validate compose config**

```bash
docker compose config
```

Expected: valid YAML (build may be skipped on machine without Docker GPU).

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore scripts/start.sh docs/RUNPOD.md README.md .env.example
git commit -m "chore: Docker and RunPod deployment docs"
```

---

### Task 8: End-to-end verification

**Files:** touch only if fixes needed

- [ ] **Step 1: Full pytest**

```bash
uv run pytest -q
```

Expected: all pass

- [ ] **Step 2: Local serve smoke**

```bash
VIDEOCLEAN_UI_USER=a VIDEOCLEAN_UI_PASSWORD=b uv run videoclean serve --host 127.0.0.1 --port 7860
```

Open UI (browser tools if available): login, Models list renders, Jobs empty state, Clean rejects empty prompt.

- [ ] **Step 3: Spec checklist**

Confirm each Success criterion in the design doc has a corresponding implemented path (note GPU criterion as “verified on RunPod by user” if no 4090 locally).

- [ ] **Step 4: Final commit if fixes**

```bash
git add -A && git status
git commit -m "fix: polish Gradio/RunPod integration"
```

---

## Self-review (plan vs spec)

| Spec item | Task |
|---|---|
| Gradio 3 tabs | 5 |
| Auth env | 5–6 |
| Fast start / no bake weights | 7 |
| Manual download + progress | 4–5 |
| Ready-only selects | 4–5 |
| FIFO one GPU job | 3 |
| Jobs cancel/retry/delete/stale/orphan | 2–3, 5–6 |
| LLM auto/local/cloud | 5 (form) + existing resolve_llm |
| Docker/RunPod | 7 |
| Architecture import bug | 1 |
| PipelineConfig import | 1 |
| ProgressBridge | 3 |
| torch/sam2-video doctor warning | 6 |
| pytest green | 8 |

No TBD placeholders left in tasks. Types: Job states `COMPLETED` consistent with CLI; catalog ids match spec.
