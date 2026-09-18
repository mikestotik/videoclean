# React Editor Phase 1 (+ library/server/webui разделение) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разделить репозиторий на library (`videoclean/`, CA) / server (`server/`, FastAPI) / webui (`webui/`, React+FSD), и собрать фазу 1 редактора: филмстрип с мультивыбором кадров, превью на выделении, таргеты, полные прогоны, конфиг — на React.

**Architecture:** `videoclean` остаётся чистой библиотекой (domain → application → adapters); FastAPI-хост переезжает в `server/` (зависимость one-way server → videoclean); фронт — Vite+React+TS+Tailwind4+shadcn(base-ui) в FSD-слоях app/pages/widgets/features/entities/shared, билд отдаётся FastAPI из `server/static_dist/`.

**Tech Stack:** Python 3.11, FastAPI, pytest, hatchling; bun, Vite 8, React 19, TypeScript, Tailwind 4, shadcn (base-mira/base-ui).

**Spec:** `docs/superpowers/specs/2026-09-08-react-editor-phase1.md`

## Global Constraints

- Python пакетный менеджер: `uv`. Тесты: `uv run pytest -q` (существующие ~199 тестов должны остаться зелёными).
- Frontend пакетный менеджер: **bun** (не npm). Команды: `cd webui && bun run lint && bun run typecheck && bun run build`.
- **Тесты — минимум.** Фронтовые тесты НЕ пишем (vitest не добавляем). На бэкенде только тесты на новые эндпоинты, суммарно ≤ 8 новых pytest-тестов (уже покрытую библиотеку не трогаем). Цель — быстро запуститься.
- **Стили фронта: только shadcn.** Никаких самодельных панелей/кнопок/инпутов: используем компоненты shadcn (`card, button, input, textarea, label, select, tabs, table, dialog, badge, scroll-area, separator`). Кастомные классы — только для сетки/отступов (Tailwind utility) и точечной подгонки.
- Правило зависимостей: `domain` ← `application` ← `adapters` ← `composition`; `server → videoclean` one-way; `videoclean` НЕ импортирует `server`; webui говорит с сервером только HTTP.
- Никаких новых рантайм-зависимостей python; на фронте не добавлять роутер/стейт-менеджеры (только shadcn-компоненты через CLI).
- Пользователь русскоязычный: UI-тексты на русском, код/комменты/имена — en.
- Коммитить после каждой задачи. Не трогать `docs/superpowers/` (в .gitignore).
- `tests/test_architecture.py` — сторож правил; расширять при переносе.

---

### Task 1: AGENTS.md — архитектурные правила репозитория

**Files:**
- Create: `AGENTS.md`

**Interfaces:**
- Produces: `AGENTS.md` с правилами структуры (library/server/webui), командами и FSD/CA-правилами для будущих агентов.

- [ ] **Step 1: Создать AGENTS.md**

```markdown
# AGENTS.md — videoclean

CLI + WebUI для удаления объектов/текста/логотипов из видео по текстовому промпту.

## Структура репозитория (три контура)

| Контур | Путь | Что это |
|---|---|---|
| library | `videoclean/` | Python-библиотека, Clean Architecture |
| server  | `server/`    | FastAPI-хост WebUI и job-воркера |
| webui   | `webui/`     | React-фронт (Vite + FSD), билд отдаёт server |

### library: `videoclean/` (Clean Architecture)

- `domain/` — сущности и правила (Target, Track, MediaManifest). Не импортирует ничего из внешних кругов.
- `application/` — use cases (`use_cases/`), порты (`ports/`), конфиг, выбор треков. Импортирует только domain.
- `adapters/` — реализации портов: media (ffmpeg), detectors, segmenters, inpainters, prompt/llm, models, progress. Импортируют application/domain.
- `composition.py` — composition root: собирает зависимости. Может импортировать всё из библиотеки.
- `cli.py` — тонкий CLI-вход (typer) поверх use cases.

Правило зависимостей — source dependencies указывают внутрь:
`domain ← application ← adapters ← composition`. Контролируется `tests/test_architecture.py`.

### server: `server/`

FastAPI-хост: `fastapi_app.py` (роуты, auth), `app_state.py`, `service.py`, статика.
- Зависит от `videoclean` (one-way). `videoclean` НЕ импортирует `server`.
- Новые эндпоинты — тонкие: парсинг формы → use case из библиотеки. Бизнес-логики в server нет.

### webui: `webui/` (Feature-Sliced Design)

Слои (импорт только «вниз»; между слайсами одного слоя — нельзя):
- `app/` — корень: провайдеры, табы-роутинг
- `pages/` — workspace, config
- `widgets/` — крупные блоки: filmstrip, viewer, preview-grid, jobs-sidebar, target-table, run-form
- `features/` — сценарии: preview-run, frame-selection
- `entities/` — job, frame, preview, targets: типы + API-клиенты + чистая логика
- `shared/` — api-клиент, `ui/` (shadcn), `lib/`, `hooks/`

Public API слайса — `index.ts`. Снаружи слайс импортируется только через него.
`components.json` aliases указывают на `@/shared/*`.
UI собирается ТОЛЬКО из shadcn-компонентов (`bunx shadcn@latest add <name>`),
кастомные стили не выдумываем — Tailwind utilities для layout и всё.

## Команды

```bash
# Python (из корня)
uv run pytest -q
uv run videoclean --help
uv run videoclean serve            # http://127.0.0.1:7860 (admin/admin)

# Frontend (из webui/)
bun install
bun run dev        # vite dev (proxy /api → :7860)
bun run lint && bun run typecheck && bun run build   # билд → server/static_dist/
bunx shadcn@latest add <component>                   # новые shadcn-компоненты
```

## Документация

- `docs/PARAMS.md` — все параметры пайплайна (CLI/WebUI/doctor).
- `docs/MODELS.md` — модели: что брать/не брать и почему.
- `docs/superpowers/specs|plans/` — спеки и планы фич (в .gitignore, не коммитятся).

## Конвенции

- Python: python 3.11, типизация, docstring кратко по-английски, без комментариев-мусора.
- Тесты: `tests/test_<area>.py`, pytest. Новые тесты — только на новые эндпоинты/фиксы (максимум необходимого).
- Frontend: TS strict, функции-компоненты, vanilla fetch (без axios/react-query), shadcn-компоненты для всего UI.
- UI-тексты — на русском; идентификаторы/код — en.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md && git commit -m "docs: add AGENTS.md with repo architecture rules (library/server/webui)"
```

---

### Task 2: ProgressBridge уезжает в adapters/progress (CA-фикс)

`composition.py` не должен импортировать web-адаптер. ProgressBridge — обёртка над JobStore, живёт в `adapters/progress/`.

**Files:**
- Create: `videoclean/adapters/progress/job_store.py` (содержимое `videoclean/adapters/web/progress_bridge.py`)
- Delete: `videoclean/adapters/web/progress_bridge.py`
- Modify: `videoclean/composition.py:282` (импорт), `tests/test_job_worker.py:8`

**Interfaces:**
- Produces: `ProgressBridge(jobs: JobIndex, job_id: str) -> ProgressPort` из `videoclean.adapters.progress.job_store`.

- [ ] **Step 1: Перенести файл и править импорты**

В `tests/test_job_worker.py` заменить:
```python
from videoclean.adapters.web.progress_bridge import ProgressBridge
```
на:
```python
from videoclean.adapters.progress.job_store import ProgressBridge
```
Затем:
```bash
git mv videoclean/adapters/web/progress_bridge.py videoclean/adapters/progress/job_store.py
```
В `videoclean/composition.py:282` заменить:
```python
    from videoclean.adapters.web.progress_bridge import ProgressBridge
```
на:
```python
    from videoclean.adapters.progress.job_store import ProgressBridge
```

- [ ] **Step 2: Тесты зелёные**

Run: `uv run pytest tests/test_job_worker.py tests/test_architecture.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: move ProgressBridge to adapters/progress (composition must not import web layer)"
```

---

### Task 3: Перенос web-адаптера в server/

**Files:**
- Create: `server/__init__.py`, `server/fastapi_app.py`, `server/app_state.py`, `server/service.py`, `server/static/*` (из `videoclean/adapters/web/static/`)
- Delete: `videoclean/adapters/web/` (весь пакет)
- Modify: `videoclean/cli.py:751`, `tests/test_tunables.py`, `tests/test_api_preview.py`, `tests/test_cli_serve_helpers.py`, `tests/test_architecture.py`, `pyproject.toml:29`

**Interfaces:**
- Produces: `server.fastapi_app.create_app(state) -> FastAPI`, `server.fastapi_app.launch_from_env(...)` (сигнатуры без изменений), `server.service.*` (те же функции).

- [ ] **Step 1: git mv пакета**

```bash
mkdir server && git mv videoclean/adapters/web/__init__.py server/__init__.py \
  && git mv videoclean/adapters/web/fastapi_app.py server/fastapi_app.py \
  && git mv videoclean/adapters/web/app_state.py server/app_state.py \
  && git mv videoclean/adapters/web/service.py server/service.py \
  && git mv videoclean/adapters/web/static server/static \
  && rm -rf videoclean/adapters/web
```
В `server/__init__.py` заменить импорт `videoclean.adapters.web.app_state` → `server.app_state`.

- [ ] **Step 2: Правка импортов**

Механическая замена по всем затронутым файлам (`server/*.py`, `videoclean/cli.py`, `tests/test_tunables.py`, `tests/test_api_preview.py`, `tests/test_cli_serve_helpers.py`):
- `videoclean.adapters.web.` → `server.`
- `from videoclean.adapters.web import fastapi_app` → `from server import fastapi_app`

В `server/fastapi_app.py`:
```python
from server.app_state import AppState, build_app_state
from server.service import (...)
```
В `server/service.py`: `from server.app_state import AppState`.
В `videoclean/cli.py:751`:
```python
from server.fastapi_app import launch_from_env
```
В `tests/test_cli_serve_helpers.py` monkeypatch-пути: `"server.fastapi_app.launch_from_env"`.

- [ ] **Step 3: pyproject packages + расширенный сторож архитектуры**

`pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel]
packages = ["videoclean", "server"]
```
`tests/test_architecture.py` — добавить (это часть ≤8 разрешённых новых тестов — сторож, не бизнес-тест):
```python
SERVER = Path(__file__).resolve().parents[1] / "server"


def _py_files_in(root: Path) -> list[Path]:
    return list(root.rglob("*.py"))


def test_server_does_not_import_webui():
    for path in _py_files_in(SERVER):
        assert "webui" not in path.read_text(encoding="utf-8"), f"{path} imports webui"


def test_library_does_not_import_server():
    banned = ("from server", "import server")
    for path in _py_files("domain") + _py_files("application") + _py_files_in(ROOT / "adapters"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            assert name not in text, f"{path} imports {name}"
```

- [ ] **Step 4: Полный прогон тестов**

Run: `uv run pytest -q`
Expected: PASS (все, кроме 1 skipped)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: extract FastAPI host to server/ package (library/server/webui split)"
```

---

### Task 4: GET /api/jobs/{id}/probe — манифест входного видео

Фронтовому филмстрипу нужен fps и frame_count выбранного видео.

**Files:**
- Modify: `server/fastapi_app.py` (новый роут после `job_input`)
- Test: `tests/test_api_probe.py` (2 теста)

**Interfaces:**
- Consumes: `videoclean.adapters.media.ffmpeg.FFmpegMedia.probe(path) -> MediaManifest`
- Produces: `GET /api/jobs/{job_id}/probe` → `{"fps": float, "duration_s": float, "width": int, "height": int, "frame_count": int}`; 404 если нет джоба/файла; 502 при ошибке ffmpeg.

- [ ] **Step 1: Тест**

`tests/test_api_probe.py`:
```python
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import build_app_state
from server.service import auth_from_env


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "admin")
    auth_from_env()
    state = build_app_state(tmp_path, worker=False, downloader=False)
    return TestClient(fa.create_app(state)), state, tmp_path


def _make_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True,
    )


def test_probe_returns_manifest(client, tmp_path: Path):
    http, state, _ = client
    video = tmp_path / "in.mp4"
    _make_video(video)
    state.jobs.upsert("j-probe", "COMPLETED", input_path=str(video))
    res = http.get("/api/jobs/j-probe/probe")
    assert res.status_code == 200
    body = res.json()
    assert body["width"] == 160
    assert body["height"] == 120
    assert abs(body["fps"] - 10.0) < 0.01
    assert body["frame_count"] == 10


def test_probe_unknown_job(client):
    http, _, _ = client
    assert http.get("/api/jobs/nope/probe").status_code == 404
```

- [ ] **Step 2: FAIL → реализация → PASS**

Run: `uv run pytest tests/test_api_probe.py -q` → Expected: FAIL (роута нет).

В `server/fastapi_app.py` после роута `job_input`:
```python
    @app.get("/api/jobs/{job_id}/probe")
    def job_probe(job_id: str, st: AppState = Depends(get_state)):
        row = st.jobs.get(job_id)
        if row is None:
            raise HTTPException(404, f"unknown job {job_id}")
        path = Path(row["input_path"] or "")
        if not path.is_file():
            raise HTTPException(404, "input file missing")
        from videoclean.adapters.media.ffmpeg import FFmpegMedia

        try:
            m = FFmpegMedia().probe(path)
        except PipelineError as exc:
            raise HTTPException(502, str(exc)) from exc
        return {
            "fps": m.fps,
            "duration_s": m.duration_s,
            "width": m.width,
            "height": m.height,
            "frame_count": m.frame_count,
        }
```
(`PipelineError` уже импортирован в файле.)

Run: `uv run pytest tests/test_api_probe.py -q` → Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(api): GET /api/jobs/{id}/probe for filmstrip frame mapping"
```

---

### Task 5: POST /api/preview/from-job — превью без re-upload

**Files:**
- Modify: `server/service.py` (новая функция `queue_preview_from_job`), `server/fastapi_app.py` (роут)
- Test: `tests/test_api_preview_from_job.py` (2 теста)

**Interfaces:**
- Consumes: `queue_preview_job(state, src, prompt, request, original_name)` — уже копирует файл в `uploads/{job_id}/input`; новая функция переиспользует существующий input.
- Produces: `POST /api/preview/from-job`, JSON body `{"job_id": str, "prompt": str, "mode": "parse"|"detect", "indices": [int], "targets": [...]}` → `201 {id, state, poll}`. 400: нет джоба/файла/таргетов.

- [ ] **Step 1: Тест**

`tests/test_api_preview_from_job.py`:
```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import build_app_state
from server.service import auth_from_env


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "admin")
    auth_from_env()
    state = build_app_state(tmp_path, worker=False, downloader=False)
    return TestClient(fa.create_app(state)), state, tmp_path


def test_from_job_requires_existing_input(client, tmp_path: Path):
    http, state, _ = client
    state.jobs.upsert("j-src", "COMPLETED", input_path=str(tmp_path / "missing.mp4"))
    res = http.post("/api/preview/from-job", json={"job_id": "j-src", "prompt": "text"})
    assert res.status_code == 400


def test_from_job_queues_preview(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    http, state, tmp = client
    video = tmp / "in.mp4"
    video.write_bytes(b"stub")
    state.jobs.upsert("j-src3", "COMPLETED", input_path=str(video))
    captured: dict = {}

    def fake_queue(state_, src, prompt, request, original_name=""):
        captured.update(request)
        captured["src"] = src
        return "j-preview"

    monkeypatch.setattr(fa, "queue_preview_job", fake_queue)
    res = http.post(
        "/api/preview/from-job",
        json={"job_id": "j-src3", "prompt": "logo", "indices": [3, 7]},
    )
    assert res.status_code == 201
    assert res.json()["id"] == "j-preview"
    assert captured["indices"] == [3, 7]
    assert captured["kind"] == "preview"
    assert captured["src"].name == "in.mp4"
```

- [ ] **Step 2: FAIL → реализация → PASS**

Run: `uv run pytest tests/test_api_preview_from_job.py -q` → Expected: FAIL (404).

`server/service.py`, рядом с `queue_preview_job`:
```python
def queue_preview_from_job(
    state: AppState,
    job_id: str,
    request: dict[str, Any],
    prompt: str = "",
) -> str:
    """Queue a preview run reusing the input video of an existing job."""
    row = state.jobs.get(job_id)
    if row is None:
        raise PipelineError(f"unknown job {job_id}")
    src = Path(row["input_path"] or "")
    if not src.is_file():
        raise PipelineError("source job has no input file")
    return queue_preview_job(state, src, prompt, request, original_name=src.name)
```
В `server/fastapi_app.py` после `create_preview`:
```python
    @app.post("/api/preview/from-job")
    async def create_preview_from_job(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        job_id = str(data.get("job_id") or "").strip()
        if not job_id:
            raise HTTPException(400, "job_id is required")
        payload: dict[str, Any] = {"kind": "preview"}
        for key in (
            "mode", "device", "detector", "detector_model", "detector_threshold",
            "segmenter", "segmenter_model", "mask_dilate_px",
        ):
            if str(data.get(key) or "").strip():
                payload[key] = data[key]
        for key in ("start", "count", "stride"):
            if data.get(key) is not None:
                payload[key] = int(data[key])
        if data.get("indices"):
            payload["indices"] = [int(i) for i in data["indices"]]
        if data.get("targets"):
            payload["targets"] = data["targets"]
        try:
            new_id = queue_preview_from_job(st, job_id, payload, prompt=str(data.get("prompt") or ""))
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(new_id)
        body_out = job_dict(st, row) if row is not None else {"id": new_id, "state": "QUEUED"}
        body_out["poll"] = f"/api/jobs/{new_id}"
        return JSONResponse(body_out, status_code=201)
```
(`queue_preview_from_job` добавить в импорты из `server.service` наверху файла.)
Обновить `/api`-индекс: `"POST /api/preview/from-job": "JSON: reuse input of an existing job"`.

Run: `uv run pytest tests/test_api_preview_from_job.py tests/test_api_preview.py -q` → Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(api): POST /api/preview/from-job — preview on an already uploaded video"
```

---

### Task 6: Раздача собранного webui (static_dist) + заглушка

**Files:**
- Modify: `server/fastapi_app.py:35-36,99,108-111`
- Test: `tests/test_api_static.py` (2 теста)

**Interfaces:**
- Produces: константа `DIST_DIR = server/static_dist`; если `DIST_DIR/index.html` существует, `/` и `/config` отдают его, `/assets` монтируется; иначе — HTML-заглушка «UI not built».

- [ ] **Step 1: Тест**

`tests/test_api_static.py`:
```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.app_state import build_app_state
from server.service import auth_from_env


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "admin")
    auth_from_env()
    state = build_app_state(tmp_path, worker=False, downloader=False)
    return TestClient(fa.create_app(state))


def test_placeholder_when_dist_missing(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fa, "DIST_DIR", Path("/nonexistent-dist"))
    res = client.get("/")
    assert res.status_code == 200
    assert "bun run build" in res.text


def test_spa_served_when_dist_exists(client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>SPA OK</body></html>", encoding="utf-8")
    monkeypatch.setattr(fa, "DIST_DIR", dist)
    assert client.get("/").status_code == 200
    assert "SPA OK" in client.get("/config").text
```

- [ ] **Step 2: FAIL → реализация → PASS**

Run: `uv run pytest tests/test_api_static.py -q` → Expected: FAIL.

В `server/fastapi_app.py` заменить `STATIC_DIR/INDEX_HTML` (строки 35–36):
```python
STATIC_DIR = Path(__file__).resolve().parent / "static"
DIST_DIR = Path(__file__).resolve().parent / "static_dist"

_PLACEHOLDER_HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>videoclean</title><body style="font-family:system-ui;max-width:40rem;margin:4rem auto">
<h1>videoclean webui</h1>
<p>React UI is not built yet. Run:</p>
<pre>cd webui && bun install && bun run build</pre>
<p>Then restart <code>videoclean serve</code>. API docs: <a href="/api/docs">/api/docs</a></p>
</body></html>"""
```
Маунт статики (строка 99) и index-роут (108–111):
```python
    if DIST_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")
```
и
```python
    @app.get("/", response_class=HTMLResponse)
    @app.get("/config", response_class=HTMLResponse)
    def index():
        spa_index = DIST_DIR / "index.html"
        if spa_index.is_file():
            return spa_index.read_text(encoding="utf-8")
        return _PLACEHOLDER_HTML
```
(legacy `/static` маунт и `STATIC_DIR` удаляются в Task 14; до тех пор маунт остаётся.)

Run: `uv run pytest tests/test_api_static.py tests/test_api_preview.py -q` → Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat(server): serve built webui from server/static_dist with placeholder fallback"
```

---

### Task 7: webui FSD-скелет + shadcn-компоненты

Перенести shadcn в `shared/ui`, алиасы, папки слоёв, добрать нужные компоненты.

**Files:**
- Create: `webui/src/app/`, `webui/src/pages/{workspace,config}/`, `webui/src/widgets/`, `webui/src/features/`, `webui/src/entities/`, `webui/src/shared/{ui,lib,hooks,api}/`
- Move: `webui/src/components/ui/button.tsx` → `webui/src/shared/ui/button.tsx`; `webui/src/lib/utils.ts` → `webui/src/shared/lib/utils.ts`
- Delete: `webui/src/components/`, `webui/src/lib/`
- Modify: `webui/components.json`, `webui/vite.config.ts`, `webui/src/App.tsx`, `webui/src/main.tsx`

**Interfaces:**
- Produces: алиасы `@app @pages @widgets @features @entities @shared`; `shared/ui/*` (shadcn), `shared/lib/utils` (`cn`).

- [ ] **Step 1: Перенести файлы**

```bash
cd webui && mkdir -p src/shared/ui src/shared/lib src/shared/api src/shared/hooks \
  src/entities/{job,frame,preview,targets} src/features/{preview-run,frame-selection} \
  src/widgets src/pages/{workspace,config} src/app \
  && git mv src/components/ui/button.tsx src/shared/ui/button.tsx \
  && git mv src/lib/utils.ts src/shared/lib/utils.ts \
  && git mv src/components/theme-provider.tsx src/shared/ui/theme-provider.tsx \
  && rm -rf src/components src/lib src/assets/react.svg
```
В `src/shared/ui/button.tsx` и `theme-provider.tsx` заменить импорты `cn`/`@/lib/utils` → `@/shared/lib/utils` (в template `cn` импортируется из пакета `cn` — заменить на локальный utils; в `utils.ts` должен экспортироваться `cn`).

- [ ] **Step 2: Добавить shadcn-компоненты (CLI, стиль base-mira уже настроен)**

```bash
cd webui && bunx shadcn@latest add card input textarea label select tabs table dialog badge scroll-area separator
```
Компоненты лягут в `src/shared/ui/` (согласно `components.json`). Если CLI положит их в другое место — перенести в `src/shared/ui/` и поправить внутренние импорты `cn` на `@/shared/lib/utils`.

- [ ] **Step 3: components.json + vite aliases**

`webui/components.json` (блок aliases):
```json
  "aliases": {
    "components": "@/shared/ui",
    "utils": "@/shared/lib/utils",
    "ui": "@/shared/ui",
    "lib": "@/shared/lib",
    "hooks": "@/shared/hooks"
  }
```
`webui/vite.config.ts`:
```ts
import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const r = (p: string) => path.resolve(__dirname, p)

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@app": r("./src/app"),
      "@pages": r("./src/pages"),
      "@widgets": r("./src/widgets"),
      "@features": r("./src/features"),
      "@entities": r("./src/entities"),
      "@shared": r("./src/shared"),
      "@": r("./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/health": "http://127.0.0.1:7860",
    },
  },
  build: { outDir: "../server/static_dist", emptyOutDir: true },
})
```
Тот же набор алиасов добавить в `webui/tsconfig.app.json` в `compilerOptions`:
```json
    "baseUrl": ".",
    "paths": {
      "@app/*": ["./src/app/*"],
      "@pages/*": ["./src/pages/*"],
      "@widgets/*": ["./src/widgets/*"],
      "@features/*": ["./src/features/*"],
      "@entities/*": ["./src/entities/*"],
      "@shared/*": ["./src/shared/*"],
      "@/*": ["./src/*"]
    }
```
(объединить с существующими опциями, не заменять файл целиком).

- [ ] **Step 4: App shell с Tabs (shadcn)**

`webui/src/app/App.tsx` (заменить `webui/src/App.tsx`, старый удалить):
```tsx
import { useState } from "react"
import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/tabs"
import { WorkspacePage } from "@pages/workspace"
import { ConfigPage } from "@pages/config"

export function App() {
  const [tab, setTab] = useState("workspace")
  return (
    <div className="mx-auto flex min-h-svh max-w-6xl flex-col gap-4 p-4">
      <header className="flex items-center gap-4">
        <h1 className="text-lg font-medium">videoclean</h1>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="workspace">Рабочая</TabsTrigger>
            <TabsTrigger value="config">Конфиг</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>
      <main className="flex-1">{tab === "workspace" ? <WorkspacePage /> : <ConfigPage />}</main>
    </div>
  )
}

export default App
```
`webui/src/main.tsx` — импорт `./App` → `@app/App`; удалить `webui/src/App.tsx`.

Страницы-плейсхолдеры `webui/src/pages/workspace/index.tsx` и `webui/src/pages/config/index.tsx`:
```tsx
export function WorkspacePage() {
  return <p className="text-sm text-muted-foreground">Рабочая страница — в следующих задачах.</p>
}
```
```tsx
export function ConfigPage() {
  return <p className="text-sm text-muted-foreground">Конфиг — в следующих задачах.</p>
}
```

- [ ] **Step 5: Сборка и проверки**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: все зелёные; `server/static_dist/index.html` создан.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(webui): FSD skeleton (app/pages/widgets/features/entities/shared) + shadcn kit"
```

---

### Task 8: shared/api + entities (job, preview, frame, targets)

Без тестов — чистая логика вынесена в entities и проверяется typecheck'ом.

**Files:**
- Create: `webui/src/shared/api/client.ts`, `webui/src/entities/job/{types.ts,api.ts,index.ts}`, `webui/src/entities/preview/{types.ts,api.ts,index.ts}`, `webui/src/entities/frame/index.ts`, `webui/src/entities/targets/{types.ts,index.ts}`

**Interfaces:**
- Produces:
  - `api<T>(path, init?): Promise<T>` — fetch c JSON, бросает `ApiError {status, message}` при !ok;
  - `entities/job`: `type Job`, `listJobs()`, `getJob(id)`, `cancelJob(id)`, `retryJob(id)`, `deleteJob(id)`, `submitRun(video, prompt, fields)`, `probeJob(id) -> MediaProbe {fps, duration_s, width, height, frame_count}`;
  - `entities/preview`: `type PreviewManifest/PreviewFrame`, `parsePreviewManifest(json)`, `fetchPreviewManifest(jobId)`, `previewArtifactUrl(jobId, name)`, `runPreviewFromJob(body)`;
  - `entities/frame`: `frameTimes(durationS, count)`, `timeToFrameIndex(t, fps, maxFrame)`, `toggleSelect(sel, idx)`, `rangeSelect(sel, idx)`;
  - `entities/targets`: `type TargetRow`, `parseTargetsJson(text)`, `targetsToJson(rows)`.

- [ ] **Step 1: shared/api/client.ts**

```ts
export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) message = String(data.detail)
    } catch {
      // keep default message
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}
```

- [ ] **Step 2: entities/job**

`types.ts`:
```ts
export type JobState = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

export type Job = {
  id: string
  state: JobState
  prompt: string
  created_at: string
  updated_at: string
  error: string
  stage: string
  fraction: number
  detail: string
  eta: string
  kind: "run" | "preview"
  has_output: boolean
  has_input: boolean
  output_url: string | null
  input_url: string | null
  status_url: string
}

export type MediaProbe = {
  fps: number
  duration_s: number
  width: number
  height: number
  frame_count: number
}
```
`api.ts`:
```ts
import { api } from "@/shared/api/client"
import type { Job, MediaProbe } from "./types"

export const listJobs = () => api<{ jobs: Job[] }>("/api/poll").then((r) => r.jobs)
export const getJob = (id: string) => api<Job>(`/api/jobs/${id}`)
export const cancelJob = (id: string) => api<Job>(`/api/jobs/${id}/cancel`, { method: "POST" })
export const retryJob = (id: string) => api<Job>(`/api/jobs/${id}/retry`, { method: "POST" })
export const deleteJob = (id: string) => api<{ ok: boolean }>(`/api/jobs/${id}`, { method: "DELETE" })
export const probeJob = (id: string) => api<MediaProbe>(`/api/jobs/${id}/probe`)

export function submitRun(video: File, prompt: string, fields: Record<string, string>): Promise<Job> {
  const form = new FormData()
  form.append("video", video)
  form.append("prompt", prompt)
  for (const [k, v] of Object.entries(fields)) if (v) form.append(k, v)
  return api<Job>("/api/jobs", { method: "POST", body: form })
}
```
`index.ts`: `export * from "./types"` + `export * from "./api"`.

- [ ] **Step 3: entities/frame**

`index.ts`:
```ts
export function frameTimes(durationS: number, count: number): number[] {
  const n = Math.max(1, count)
  const d = Math.max(0, durationS)
  if (n === 1) return [0]
  const step = d / (n - 1)
  return Array.from({ length: n }, (_, i) => Math.min(d, i * step))
}

export function timeToFrameIndex(t: number, fps: number, maxFrame: number): number {
  const idx = Math.round(t * Math.max(fps, 0.001))
  return Math.min(Math.max(0, maxFrame), Math.max(0, idx))
}

export function toggleSelect(sel: Set<number>, idx: number): Set<number> {
  const next = new Set(sel)
  if (next.has(idx)) next.delete(idx)
  else next.add(idx)
  return next
}

export function rangeSelect(sel: Set<number>, idx: number): Set<number> {
  const next = new Set(sel)
  const anchor = [...sel].sort((a, b) => a - b).at(-1) ?? idx
  const [from, to] = anchor <= idx ? [anchor, idx] : [idx, anchor]
  for (let i = from; i <= to; i++) next.add(i)
  return next
}
```

- [ ] **Step 4: entities/preview**

`types.ts`:
```ts
export type PreviewFrame = {
  index: number
  maskCoverage: number
  boxes: (number[] | null)[]
  artifacts: { raw: string; boxes: string; mask: string }
}

export type PreviewManifest = {
  state: string
  mode?: string
  framesRequested: number[]
  frames: PreviewFrame[]
  meanMaskCoverage?: number
  targets?: { kind: string; query: string }[]
  error?: string
}

export function parsePreviewManifest(data: unknown): PreviewManifest {
  const raw = (data ?? {}) as Record<string, unknown>
  const framesRaw = Array.isArray(raw.frames) ? raw.frames : []
  const frames: PreviewFrame[] = framesRaw.map((f) => {
    const fr = (f ?? {}) as Record<string, unknown>
    const index = Number(fr.index ?? 0)
    return {
      index,
      maskCoverage: Number(fr.maskCoverage ?? 0),
      boxes: Array.isArray(fr.boxes) ? (fr.boxes as (number[] | null)[]) : [],
      artifacts: {
        raw: `${String(index).padStart(6, "0")}_raw.jpg`,
        boxes: `${String(index).padStart(6, "0")}_boxes.jpg`,
        mask: `${String(index).padStart(6, "0")}_mask.jpg`,
      },
    }
  })
  return {
    state: String(raw.state ?? ""),
    mode: raw.mode === undefined ? undefined : String(raw.mode),
    framesRequested: Array.isArray(raw.framesRequested) ? (raw.framesRequested as number[]) : [],
    frames,
    meanMaskCoverage: raw.meanMaskCoverage === undefined ? undefined : Number(raw.meanMaskCoverage),
    targets: Array.isArray(raw.targets) ? (raw.targets as { kind: string; query: string }[]) : [],
    error: raw.error === undefined ? undefined : String(raw.error),
  }
}
```
`api.ts`:
```ts
import type { Job } from "@/entities/job"
import { api } from "@/shared/api/client"
import { parsePreviewManifest, type PreviewManifest } from "./types"

export const fetchPreviewManifest = (jobId: string) =>
  api<unknown>(`/api/jobs/${jobId}/preview/preview.json`).then(parsePreviewManifest)

export const previewArtifactUrl = (jobId: string, name: string) =>
  `/api/jobs/${jobId}/preview/${encodeURIComponent(name)}`

export type PreviewFromJobBody = {
  job_id: string
  prompt?: string
  mode?: "parse" | "detect"
  indices?: number[]
  targets?: { kind: string; query: string; where?: string | null }[]
  mask_dilate_px?: string
}

export const runPreviewFromJob = (body: PreviewFromJobBody) =>
  api<Job>("/api/preview/from-job", { method: "POST", body: JSON.stringify(body) })
```
`index.ts`: экспорт обоих.

- [ ] **Step 5: entities/targets**

`types.ts`:
```ts
export type TargetKind = "watermark" | "text_overlay" | "object"

export type TargetRow = { kind: TargetKind; query: string; where: string | null }

export function parseTargetsJson(text: string): TargetRow[] {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    return []
  }
  if (!Array.isArray(data)) return []
  return data
    .map((item): TargetRow | null => {
      const r = (item ?? {}) as Record<string, unknown>
      const query = String(r.query ?? "").trim()
      if (!query) return null
      const kind = ["watermark", "text_overlay", "object"].includes(String(r.kind))
        ? (r.kind as TargetKind)
        : "object"
      const where = r.where ? String(r.where) : null
      return { kind, query, where }
    })
}

export function targetsToJson(rows: TargetRow[]): string {
  return JSON.stringify(
    rows.map((r) => ({ kind: r.kind, query: r.query, where: r.where })),
    null,
    2,
  )
}
```
`index.ts`: `export * from "./types"`.

- [ ] **Step 6: Проверки**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(webui): shared api client + entities (job, preview, frame, targets)"
```

---

### Task 9: widgets — run-form + jobs-sidebar

Полный прогон (upload + промпт + «Дополнительно») и список джобов со статусами. UI — только shadcn.

**Files:**
- Create: `webui/src/widgets/run-form/index.tsx`, `webui/src/widgets/jobs-sidebar/index.tsx`
- Modify: `webui/src/pages/workspace/index.tsx`

**Interfaces:**
- Consumes: `entities/job` (submitRun, listJobs, cancelJob, retryJob, deleteJob), `shared/ui/*`.
- Produces: `RunForm({ onSubmitted })`, `JobsSidebar({ selectedJob, onSelect })` (внутри свой поллинг каждые 3с).

- [ ] **Step 1: RunForm**

`webui/src/widgets/run-form/index.tsx`:
```tsx
import { useRef, useState } from "react"
import { submitRun, type Job } from "@/entities/job"
import { Button } from "@/shared/ui/button"
import {
  Card, CardContent, CardHeader, CardTitle,
} from "@/shared/ui/card"
import { Input } from "@/shared/ui/input"
import { Label } from "@/shared/ui/label"
import { Textarea } from "@/shared/ui/textarea"
import { Separator } from "@/shared/ui/separator"

const ADVANCED_FIELDS = [
  "detector_threshold", "mask_dilate_px", "telea_radius", "verify",
  "prompt_frame_stride", "prompt_frame_max", "parse_chunk_frames", "vision_batch",
  "detector_keyframes", "detector_nms_iou", "detector_max_box_area",
  "tracker_min_score", "tracker_max_template_area",
  "propainter_mask_dilation", "propainter_ref_stride",
  "propainter_neighbor_length", "propainter_subvideo_length", "propainter_raft_iter",
] as const

type Props = { onSubmitted: (job: Job) => void }

export function RunForm({ onSubmitted }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [prompt, setPrompt] = useState("")
  const [advanced, setAdvanced] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const fileRef = useRef<HTMLInputElement>(null)

  const submit = async () => {
    if (!file || !prompt.trim() || busy) return
    setBusy(true)
    setError("")
    try {
      const job = await submitRun(file, prompt.trim(), values)
      onSubmitted(job)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Полный прогон</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex gap-2">
          <Input
            ref={fileRef}
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file && (
            <Button variant="ghost" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = "" }}>
              Сбросить
            </Button>
          )}
        </div>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Что удалить: напр. «текст в правом нижнем углу и красный логотип»"
        />
        <Button variant="outline" size="sm" className="self-start" onClick={() => setAdvanced((v) => !v)}>
          {advanced ? "Скрыть параметры" : "Дополнительно"}
        </Button>
        {advanced && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            {ADVANCED_FIELDS.map((f) => (
              <div key={f} className="flex flex-col gap-1">
                <Label className="text-muted-foreground">{f}</Label>
                <Input
                  value={values[f] ?? ""}
                  onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Separator />
        <Button disabled={!file || !prompt.trim() || busy} onClick={submit} className="self-start">
          {busy ? "Запуск…" : "Запустить"}
        </Button>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: JobsSidebar**

`webui/src/widgets/jobs-sidebar/index.tsx`:
```tsx
import { useEffect, useState } from "react"
import { cancelJob, deleteJob, listJobs, retryJob, type Job } from "@/entities/job"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { ScrollArea } from "@/shared/ui/scroll-area"

type Props = {
  selectedId: string | null
  onSelect: (job: Job) => void
  refreshKey: number
}

export function JobsSidebar({ selectedId, onSelect, refreshKey }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState("")

  useEffect(() => {
    let alive = true
    const tick = () => {
      listJobs()
        .then((j) => { if (alive) { setJobs(j); setError("") } })
        .catch((e) => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    }
    tick()
    const t = setInterval(tick, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [refreshKey])

  const action = async (fn: (id: string) => Promise<unknown>, id: string) => {
    try { await fn(id) } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }

  return (
    <Card className="w-80 shrink-0">
      <CardHeader><CardTitle className="text-base">Джобы</CardTitle></CardHeader>
      <CardContent>
        {error && <p className="mb-2 text-sm text-destructive">{error}</p>}
        <ScrollArea className="h-[60vh]">
          <div className="flex flex-col gap-2 pr-2">
            {jobs.map((j) => (
              <button
                key={j.id}
                className={`flex flex-col items-start gap-1 rounded-md border p-2 text-left text-xs ${j.id === selectedId ? "border-primary" : ""}`}
                onClick={() => onSelect(j)}
              >
                <span className="flex items-center gap-2">
                  <span className="font-mono">{j.id.slice(0, 8)}</span>
                  <Badge variant={j.state === "COMPLETED" ? "default" : j.state === "FAILED" ? "destructive" : "secondary"}>
                    {j.kind} · {j.state}
                  </Badge>
                </span>
                {j.stage && (
                  <span className="text-muted-foreground">
                    {j.stage} {Math.round(j.fraction * 100)}% {j.eta && `· ETA ${j.eta}`}
                  </span>
                )}
                <span className="flex gap-1">
                  {j.state === "RUNNING" || j.state === "QUEUED" ? (
                    <Button size="xs" variant="outline" onClick={(e) => { e.stopPropagation(); action(cancelJob, j.id) }}>Стоп</Button>
                  ) : null}
                  {j.state === "FAILED" ? (
                    <Button size="xs" variant="outline" onClick={(e) => { e.stopPropagation(); action(retryJob, j.id) }}>Повтор</Button>
                  ) : null}
                  {j.has_output && (
                    <a href={j.output_url ?? "#"} download onClick={(e) => e.stopPropagation()}>
                      <Button size="xs" variant="outline">Скачать</Button>
                    </a>
                  )}
                  <Button size="xs" variant="destructive" onClick={(e) => { e.stopPropagation(); action(deleteJob, j.id) }}>Удалить</Button>
                </span>
                {j.error && <p className="text-destructive">{j.error.slice(0, 200)}</p>}
              </button>
            ))}
            {jobs.length === 0 && <p className="text-xs text-muted-foreground">Пока пусто.</p>}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Workspace-страница**

`webui/src/pages/workspace/index.tsx`:
```tsx
import { useState } from "react"
import { JobsSidebar } from "@widgets/jobs-sidebar"
import { RunForm } from "@widgets/run-form"
import type { Job } from "@/entities/job"

export function WorkspacePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="flex gap-4">
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <RunForm onSubmitted={() => setRefreshKey((k) => k + 1)} />
      </div>
      <JobsSidebar selectedId={selectedId} onSelect={(j: Job) => setSelectedId(j.id)} refreshKey={refreshKey} />
    </div>
  )
}
```

- [ ] **Step 4: Проверки + smoke через dev-сервер**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.
Smoke (вручную): `uv run videoclean serve --port 7860` + `cd webui && bun run dev` → открыть http://localhost:5173, загрузить маленькое видео с промптом «text» → джоб появляется, статус поллится. (CORS не нужен — vite proxy.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(webui): run-form and jobs-sidebar widgets (full pipeline runs)"
```

---

### Task 10: features/frame-selection + widgets/filmstrip + viewer

**Files:**
- Create: `webui/src/features/frame-selection/index.ts`, `webui/src/widgets/filmstrip/index.tsx`, `webui/src/widgets/viewer/index.tsx`
- Modify: `webui/src/pages/workspace/index.tsx` (выбор джоба включает редактор)

**Interfaces:**
- Consumes: `entities/frame` (frameTimes, timeToFrameIndex, toggleSelect, rangeSelect), `entities/job` (Job.input_url, probeJob), `shared/ui/*`.
- Produces:
  - `useFrameSelection(): { selected: Set<number>, current: number | null, toggle(idx, additive, range), setCurrent(idx), clear() }`;
  - `Filmstrip({ src, fps, frameCount, thumbnails, selected, current, onToggle, onCurrent })`;
  - `Viewer({ src, fps, current })`.

- [ ] **Step 1: useFrameSelection**

`webui/src/features/frame-selection/index.ts`:
```ts
import { useState } from "react"
import { rangeSelect, toggleSelect } from "@/entities/frame"

export function useFrameSelection() {
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [current, setCurrent] = useState<number | null>(null)

  const toggle = (idx: number, additive: boolean, range: boolean) => {
    if (range) {
      setSelected((sel) => rangeSelect(sel, idx))
      setCurrent(idx)
      return
    }
    if (additive) setSelected((sel) => toggleSelect(sel, idx))
    else setSelected(new Set([idx]))
    setCurrent(idx)
  }

  const clear = () => {
    setSelected(new Set())
    setCurrent(null)
  }

  return { selected, current, toggle, setCurrent, clear }
}
```

- [ ] **Step 2: Filmstrip**

`webui/src/widgets/filmstrip/index.tsx`:
```tsx
import { useEffect, useRef, useState } from "react"
import { frameTimes, timeToFrameIndex } from "@/entities/frame"
import { cn } from "@/shared/lib/utils"

type Props = {
  src: string
  fps: number
  frameCount: number
  thumbnails: number
  selected: Set<number>
  current: number | null
  onToggle: (idx: number, additive: boolean, range: boolean) => void
  onCurrent: (idx: number) => void
}

export function Filmstrip({ src, fps, frameCount, thumbnails, selected, current, onToggle, onCurrent }: Props) {
  const [thumbs, setThumbs] = useState<Record<number, string>>({})

  useEffect(() => {
    if (fps <= 0 || frameCount <= 0) return
    const v = document.createElement("video")
    v.src = src
    v.muted = true
    v.preload = "auto"
    let cancelled = false

    const grabAll = () => {
      const duration = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : frameCount / Math.max(fps, 0.001)
      const t = frameTimes(duration, thumbnails)
      let i = 0
      const grab = () => {
        if (cancelled || i >= t.length) return
        const target = t[i]
        const frameIdx = timeToFrameIndex(target, fps, frameCount - 1)
        const onSeeked = () => {
          v.removeEventListener("seeked", onSeeked)
          const canvas = document.createElement("canvas")
          canvas.width = 160
          canvas.height = 90
          const ctx = canvas.getContext("2d")
          if (ctx) {
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
            const data = canvas.toDataURL("image/jpeg", 0.6)
            setThumbs((prev) => ({ ...prev, [frameIdx]: data }))
          }
          i += 1
          setTimeout(grab, 0)
        }
        v.addEventListener("seeked", onSeeked)
        v.currentTime = target
      }
      grab()
    }

    v.addEventListener("loadedmetadata", grabAll)
    return () => {
      cancelled = true
      v.pause()
    }
  }, [src, fps, frameCount, thumbnails])

  const indexes = Object.keys(thumbs).map(Number).sort((a, b) => a - b)

  return (
    <div className="flex gap-1 overflow-x-auto rounded-md border p-2">
      {indexes.map((idx) => (
        <button
          key={idx}
          className={cn(
            "relative h-14 w-24 shrink-0 overflow-hidden rounded-md border-2",
            selected.has(idx) ? "border-primary" : "border-transparent",
            current === idx && "ring-2 ring-ring",
          )}
          onClick={(e) => {
            onToggle(idx, e.ctrlKey || e.metaKey, e.shiftKey)
            onCurrent(idx)
          }}
        >
          <img src={thumbs[idx]} alt={`frame ${idx}`} className="h-full w-full object-cover" />
          <span className="absolute right-0 bottom-0 bg-background/80 px-0.5 text-[9px]">{idx}</span>
        </button>
      ))}
    </div>
  )
}
```
Дедупликация: если `frameTimes` даёт дубликаты frame-индексов (короткое видео) — `grab` пропускает уже снятые (`if (thumbs-like set has idx) skip`). Реализовать через локальный `Set<number> seen` внутри `grabAll`.

- [ ] **Step 3: Viewer**

`webui/src/widgets/viewer/index.tsx`:
```tsx
type Props = { src: string; fps: number; current: number | null }

export function Viewer({ src, fps, current }: Props) {
  return (
    <video
      key={src}
      src={src}
      controls
      className="max-h-[50vh] w-full rounded-md border bg-black"
      ref={(el) => {
        if (el && current !== null && fps > 0) el.currentTime = current / fps
      }}
    />
  )
}
```

- [ ] **Step 4: Интеграция в workspace**

В `pages/workspace/index.tsx`: при клике по джобу сохраняем весь `Job` в состояние `selectedJob: Job | null`; `useEffect` на `selectedJob?.id` → `probeJob(id)` → `probe`. Рендер над RunForm:
```tsx
{selectedJob?.input_url && probe && (
  <Viewer src={selectedJob.input_url} fps={probe.fps} current={selection.current} />
)}
{selectedJob?.input_url && probe && (
  <Filmstrip
    src={selectedJob.input_url}
    fps={probe.fps}
    frameCount={probe.frame_count}
    thumbnails={48}
    selected={selection.selected}
    current={selection.current}
    onToggle={selection.toggle}
    onCurrent={(idx) => selection.setCurrent(idx)}
  />
)}
```
`const selection = useFrameSelection()`.

- [ ] **Step 5: Проверки**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(webui): frame-selection feature, filmstrip with canvas thumbnails, viewer"
```

---

### Task 11: features/preview-run + widgets/preview-grid

**Files:**
- Create: `webui/src/features/preview-run/index.ts`, `webui/src/widgets/preview-grid/index.tsx`
- Modify: `webui/src/pages/workspace/index.tsx`

**Interfaces:**
- Consumes: `entities/preview` (runPreviewFromJob, fetchPreviewManifest, previewArtifactUrl, PreviewManifest), `entities/job` (getJob).
- Produces:
  - `framesToIndices(sel): number[]`;
  - `usePreviewRun(job, selectedFrames, targets)` → `{ run(mode, prompt), manifest, jobId, running, error }`;
  - `PreviewGrid({ jobId, manifest })` — Card + Tabs (raw/boxes/mask) + Dialog-лайтбокс.

- [ ] **Step 1: usePreviewRun**

`webui/src/features/preview-run/index.ts`:
```ts
import { useCallback, useState } from "react"
import type { Job } from "@/entities/job"
import { fetchPreviewManifest, runPreviewFromJob } from "@/entities/preview"
import type { PreviewManifest } from "@/entities/preview"

export type PreviewTarget = { kind: string; query: string; where?: string | null }

export function framesToIndices(sel: Set<number>): number[] {
  return [...sel].sort((a, b) => a - b)
}

export function usePreviewRun(job: Job | null, selectedFrames: Set<number>, targets: PreviewTarget[]) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [manifest, setManifest] = useState<PreviewManifest | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")

  const run = useCallback(
    async (mode: "parse" | "detect", prompt: string) => {
      if (!job || running) return
      const indices = framesToIndices(selectedFrames)
      if (indices.length === 0) { setError("Выберите кадры на филмстрипе"); return }
      if (mode === "parse" && !prompt.trim()) { setError("Введите промпт для разбора"); return }
      if (mode === "detect" && targets.length === 0) { setError("Добавьте таргеты для режима detect"); return }
      setRunning(true); setError(""); setManifest(null)
      try {
        const res = await runPreviewFromJob({
          job_id: job.id,
          prompt: prompt.trim(),
          mode,
          indices,
          targets: mode === "detect" ? targets : undefined,
        })
        setJobId(res.id)
        for (let i = 0; i < 600; i++) {
          await new Promise((r) => setTimeout(r, 1000))
          const status = await getJob(res.id)
          if (status.state === "COMPLETED") {
            setManifest(await fetchPreviewManifest(res.id))
            break
          }
          if (status.state === "FAILED" || status.state === "CANCELLED") {
            setError(status.error || status.state)
            break
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setRunning(false)
      }
    },
    [job, running, selectedFrames, targets],
  )

  return { run, jobId, manifest, running, error }
}
```
(`getJob` импортировать из `@/entities/job`.)

- [ ] **Step 2: PreviewGrid (Card + Tabs + Dialog)**

`webui/src/widgets/preview-grid/index.tsx`:
```tsx
import { useState } from "react"
import { previewArtifactUrl, type PreviewManifest } from "@/entities/preview"
import { Badge } from "@/shared/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/shared/ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/tabs"

type Layer = "raw" | "boxes" | "mask"

type Props = { jobId: string; manifest: PreviewManifest }

export function PreviewGrid({ jobId, manifest }: Props) {
  const [layer, setLayer] = useState<Layer>("mask")

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="text-base">Превью масок</CardTitle>
        <Badge variant="secondary">покрытие ~{Math.round((manifest.meanMaskCoverage ?? 0) * 100)}%</Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Tabs value={layer} onValueChange={(v) => setLayer(v as Layer)}>
          <TabsList>
            <TabsTrigger value="raw">raw</TabsTrigger>
            <TabsTrigger value="boxes">boxes</TabsTrigger>
            <TabsTrigger value="mask">mask</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {manifest.frames.map((f) => (
            <Dialog key={f.index}>
              <DialogTrigger asChild>
                <button className="relative overflow-hidden rounded-md border">
                  <img
                    src={previewArtifactUrl(jobId, f.artifacts[layer])}
                    alt={`frame ${f.index}`}
                    className="w-full"
                  />
                  <span className="absolute top-1 left-1 rounded bg-background/80 px-1 text-[10px]">
                    #{f.index} · {(f.maskCoverage * 100).toFixed(1)}%
                  </span>
                </button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl">
                <DialogTitle>Кадр #{f.index}</DialogTitle>
                <img
                  src={previewArtifactUrl(jobId, f.artifacts[layer])}
                  alt={`frame ${f.index} full`}
                  className="max-h-[80vh] w-full object-contain"
                />
              </DialogContent>
            </Dialog>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Интеграция в workspace**

Под филмстрипом:
```tsx
{selectedJob && (
  <Card>
    <CardHeader><CardTitle className="text-base">Превью на выделенных кадрах ({selection.selected.size})</CardTitle></CardHeader>
    <CardContent className="flex flex-col gap-3">
      <Textarea
        value={previewPrompt}
        onChange={(e) => setPreviewPrompt(e.target.value)}
        placeholder="Промпт для разбора (mode=parse)"
      />
      <div className="flex gap-2">
        <Button size="sm" disabled={preview.running} onClick={() => preview.run("parse", previewPrompt)}>
          Разбор + маски
        </Button>
        <Button size="sm" variant="outline" disabled={preview.running} onClick={() => preview.run("detect", "")}>
          Маски по таргетам
        </Button>
      </div>
      {preview.error && <p className="text-sm text-destructive">{preview.error}</p>}
      {preview.running && <p className="text-sm text-muted-foreground">Считаю…</p>}
    </CardContent>
  </Card>
)}
{preview.jobId && preview.manifest && (
  <PreviewGrid jobId={preview.jobId} manifest={preview.manifest} />
)}
```
Состояние `previewPrompt`, `const preview = usePreviewRun(selectedJob, selection.selected, targets)` (таргеты появятся в Task 12 — до этого передавать `[]`).

- [ ] **Step 4: Проверки**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.
Smoke: выбранный джоб → выделить 3–5 кадров → «Разбор + маски» → сетка превью с масками (на example_2_480.mp4).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(webui): preview-run feature and preview-grid (boxes/mask/raw with dialog lightbox)"
```

---

### Task 12: widgets/target-table

**Files:**
- Create: `webui/src/widgets/target-table/index.tsx`
- Modify: `webui/src/pages/workspace/index.tsx`

**Interfaces:**
- Consumes: `entities/targets` (TargetRow, TargetKind), `shared/ui` (Table, Select, Input, Button).
- Produces: `TargetTable({ rows, onChange })` — редактируемая таблица; state `targets` в WorkspacePage передаётся в `usePreviewRun`.

- [ ] **Step 1: TargetTable**

`webui/src/widgets/target-table/index.tsx`:
```tsx
import type { TargetKind, TargetRow } from "@/entities/targets"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"

const KINDS: TargetKind[] = ["text_overlay", "watermark", "object"]
const WHERES = ["", "top", "bottom", "left", "right", "center", "top-left", "top-right", "bottom-left", "bottom-right"]

type Props = { rows: TargetRow[]; onChange: (rows: TargetRow[]) => void }

export function TargetTable({ rows, onChange }: Props) {
  const update = (i: number, patch: Partial<TargetRow>) =>
    onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Таргеты (режим detect)</CardTitle></CardHeader>
      <CardContent className="flex flex-col gap-2">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-sm">
            <Select value={r.kind} onValueChange={(v) => update(i, { kind: v as TargetKind })}>
              <SelectTrigger className="w-40"><SelectValue /></SelectTrigger>
              <SelectContent>
                {KINDS.map((k) => <SelectItem key={k} value={k}>{k}</SelectItem>)}
              </SelectContent>
            </Select>
            <input
              value={r.query}
              onChange={(e) => update(i, { query: e.target.value })}
              className="h-9 flex-1 rounded-md border bg-transparent px-3 text-sm"
              placeholder="query"
            />
            <Select value={r.where ?? ""} onValueChange={(v) => update(i, { where: v || null })}>
              <SelectTrigger className="w-36"><SelectValue placeholder="—" /></SelectTrigger>
              <SelectContent>
                {WHERES.map((w) => <SelectItem key={w} value={w}>{w || "—"}</SelectItem>)}
              </SelectContent>
            </Select>
            <Button size="icon" variant="destructive" onClick={() => onChange(rows.filter((_, j) => j !== i))}>×</Button>
          </div>
        ))}
        <Button size="sm" variant="outline" className="self-start" onClick={() => onChange([...rows, { kind: "object", query: "", where: null }])}>
          + таргет
        </Button>
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 2: Интеграция**

В WorkspacePage: `const [targets, setTargets] = useState<TargetRow[]>([])`, `<TargetTable rows={targets} onChange={setTargets} />`, `usePreviewRun(selectedJob, selection.selected, targets)`.

- [ ] **Step 3: Проверки + Commit**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.

```bash
git add -A && git commit -m "feat(webui): target table for detect mode"
```

---

### Task 13: pages/config — модели, doctor, ollama

**Files:**
- Create: `webui/src/pages/config/index.tsx`
- Modify: ничего (App уже рендерит ConfigPage)

**Interfaces:**
- Consumes: `/api/poll` (`models`, `doctor`, `ollama`), `POST /api/models/download` через `shared/api`.
- Produces: `ConfigPage()` — Card на каждый kind моделей с кнопкой Download, карточки Doctor и Ollama.

- [ ] **Step 1: ConfigPage**

`webui/src/pages/config/index.tsx`:
```tsx
import { useCallback, useEffect, useState } from "react"
import { api } from "@/shared/api/client"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Separator } from "@/shared/ui/separator"

type ModelInfo = { id: string; title: string; kind: string; status?: string; installed?: boolean }
type PollData = {
  models: Record<string, ModelInfo[]>
  doctor: Record<string, unknown>
  ollama: { ok: boolean; base_url: string; models: string[] }
}

export function ConfigPage() {
  const [data, setData] = useState<PollData | null>(null)
  const [busy, setBusy] = useState("")
  const [error, setError] = useState("")

  const refresh = useCallback(() => {
    api<PollData>("/api/poll").then(setData).catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const download = async (id: string) => {
    setBusy(id)
    setError("")
    try {
      await api("/api/models/download", { method: "POST", body: JSON.stringify({ id }) })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
      refresh()
    }
  }

  if (!data) return <p className="text-sm text-muted-foreground">{error || "Загрузка…"}</p>

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="text-sm text-destructive">{error}</p>}
      {Object.entries(data.models).map(([kind, models]) => (
        <Card key={kind}>
          <CardHeader><CardTitle className="text-base">{kind}</CardTitle></CardHeader>
          <CardContent className="flex flex-col gap-2 text-sm">
            {models.map((m) => (
              <div key={m.id} className="flex items-center gap-2">
                <span className="font-mono">{m.title}</span>
                {m.status && <Badge variant="secondary">{m.status}</Badge>}
                <Button size="xs" variant="outline" disabled={busy === m.id} onClick={() => download(m.id)}>
                  {busy === m.id ? "…" : "Download"}
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      ))}
      <Card>
        <CardHeader><CardTitle className="text-base">Doctor</CardTitle></CardHeader>
        <CardContent>
          <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(data.doctor, null, 2)}</pre>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-base">Ollama</CardTitle></CardHeader>
        <CardContent className="text-sm">
          <Separator className="mb-2" />
          <p>
            {data.ollama.base_url} — {data.ollama.ok ? "ok" : "недоступен"} · моделей: {data.ollama.models.length}
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
```

- [ ] **Step 2: Проверить + Commit**

Run: `cd webui && bun run lint && bun run typecheck && bun run build`
Expected: зелёные.

```bash
git add -A && git commit -m "feat(webui): config page — model downloads, doctor, ollama status"
```

---

### Task 14: Выпилить legacy static + Dockerfile node-stage

**Files:**
- Delete: `server/static/` (index.html, app.css, app.js)
- Modify: `server/fastapi_app.py` (убрать `/static` маунт, `STATIC_DIR`), `Dockerfile`

**Interfaces:**
- Produces: `/` отдаёт SPA или placeholder; Docker образ собирает webui.

- [ ] **Step 1: Удалить legacy**

```bash
git rm -r server/static
```
В `server/fastapi_app.py`: удалить `STATIC_DIR = ...` и строку `app.mount("/static", ...)` (маунт `/assets` остаётся из Task 6).

- [ ] **Step 2: Dockerfile**

Добавить bun-стадию перед финальной и скопировать результат:
```dockerfile
FROM oven/bun:1 AS webui-build
WORKDIR /build
COPY webui/package.json webui/bun.lock ./
RUN bun install --frozen-lockfile
COPY webui/ ./
RUN bun run build
```
Vite outDir = `../server/static_dist` — внутри стадии это `/server/static_dist`. В финальной стадии после `COPY videoclean ./videoclean`:
```dockerfile
COPY server ./server
COPY --from=webui-build /server/static_dist ./server/static_dist
```
После сборки образа проверить локально: `docker build -t videoclean:test .` и что в контейнере есть `/app/server/static_dist/index.html`.

- [ ] **Step 3: docker-compose sanity**

Проверить `docker-compose.yml`: если есть build-контекст/пути — поправить под новую структуру (`./server` копируется в образ; volumes пользователя не трогать).

- [ ] **Step 4: Регресс**

Run: `uv run pytest -q`
Expected: PASS. `uv run videoclean serve --port 7872` → `/` отдаёт SPA (static_dist уже собран).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: drop legacy static UI; Dockerfile builds webui (bun stage)"
```

---

### Task 15: Финальная верификация и README

**Files:**
- Modify: `README.md` (раздел WebUI: сборка фронта, FSD/CA-заметка, ссылка на AGENTS.md)

- [ ] **Step 1: Полный прогон**

```bash
uv run pytest -q
cd webui && bun run lint && bun run typecheck && bun run build
```
Expected: всё зелёное.

- [ ] **Step 2: E2E smoke на реальном видео**

```bash
uv run videoclean serve --port 7860
```
Открыть http://127.0.0.1:7860 → рабочая: выбрать существующий job с example_2_480.mp4 → филмстрип → выделить 4–5 кадров → «Разбор + маски» → превью с масками. Конфиг: модели/doctor отображаются.

- [ ] **Step 3: README + commit**

README: раздел «Web UI» — как собрать фронт (`cd webui && bun install && bun run build`), что при `videoclean serve` раздаётся `server/static_dist`, dev-режим через `bun run dev` + proxy, ссылка на AGENTS.md.

```bash
git add -A && git commit -m "docs: README webui build instructions"
```

---

## Self-Review

1. **Spec coverage:** server move (T2–3), progress CA-фикс (T2), probe (T4), from-job (T5), static_dist (T6), FSD-скелет + shadcn kit (T7), api/entities (T8), полные прогоны+jobs (T9), филмстрип+selection (T10), превью на выделении (T11), таргеты (T12), конфиг (T13), legacy out + docker (T14), финал (T15). AGENTS.md — T1. Фаза 2 (rect-рисование, VLM-регион, ручные маски) — вне плана, помечена в спеке.
2. **Тесты по поправке пользователя:** фронтовых нет; новые pytest: probe (2) + from-job (2) + static (2) + архитектурные сторожи (2) = 8 ≤ 10.
3. **Placeholders:** нет TBD; все шаги с кодом; UI только на shadcn-компонентах (card/button/input/textarea/label/select/tabs/table/dialog/badge/scroll-area/separator).
4. **Type consistency:** `framesToIndices` в T11; `rangeSelect(sel, idx)` — anchor внутри (последний выбранный); артефакты `NNNNNN_{raw,boxes,mask}.jpg` совпадают с `RunPreview._write_overlays`; probe-поля совпадают с T4; `runPreviewFromJob` body соответствует T5 роуту.
