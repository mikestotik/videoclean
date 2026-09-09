# Editor Pipeline — План A: Бэкенд (library + server)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Бэкенд для редактора: реестр источников видео с кадрами и масками-аннотациями, use case BuildPrompt (промпт+маски → VLM → промпт+таргеты), ручные входы RunCleanup (tracks_override, masks_override), полный параметрический проход через HTTP, пресеты.

**Architecture:** Все изменения идут слоями: domain (tracks_from_json) → application (RunCleanup overrides, BuildPrompt) → composition/worker (wiring, dispatch kind="prompt") → server (роуты тонкие, логика в service.py). Sources живут в том же jobs.sqlite (SourceIndex), файлы — `data_dir/sources/{id}/`. Job'ы получают `source_id`.

**Tech Stack:** Python 3.11, FastAPI, pytest, sqlite3. Без новых зависимостей (cv2/numpy уже есть).

**Spec:** `docs/superpowers/specs/2026-09-08-editor-pipeline-design.md` — читать вместе с планом.

**План B (фронтенд)** пишется и исполняется после Плана A.

## Global Constraints

- Python 3.11, типизация, docstring кратко по-английски, без комментариев-мусора.
- Зависимости: `domain ← application ← adapters ← composition`; server импортирует videoclean, никогда наоборот. Контролируется `tests/test_architecture.py` — application НЕ может импортировать `videoclean.adapters` (поэтому BuildPrompt получает `system_prompt` и `to_jpeg` через конструктор, а `bgr_to_jpeg` остаётся в adapters).
- `read_bgr` (`videoclean/adapters/images.py:9`) возвращает **BGR**; write_bgr пишет BGR как есть. Оверлей красным в BuildPrompt: `red[..., 2] = 255`.
- Тесты: pytest; сообщения об ошибках PipelineError; новые тесты — максимум необходимого.
- Никаких секретов в коде; `llm_api_key` не логировать и не включать в job_dict/request.
- Коммиты после каждой задачи, conventional commits (`feat:`, `test:`, `docs:`).

---

### Task 1: SourceIndex в store.py

**Files:**
- Modify: `videoclean/store.py`
- Test: `tests/test_source_index.py`

**Interfaces:**
- Produces: `new_source_id() -> str` (`s_YYYYMMDD_HHMMSS_hex8`), `SourceIndex(db_path)` с методами `register(source_id: str, name: str, path: str, probe: dict | None = None) -> None`, `list(limit: int = 200) -> list[sqlite3.Row]`, `get(source_id: str) -> sqlite3.Row | None`, `delete(source_id: str) -> bool`. Колонки sources: id, name, path, created_at, probe_json.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_source_index.py
"""SourceIndex registry: sources survive restarts, delete removes the row."""
from pathlib import Path

from videoclean.store import SourceIndex, new_source_id


def test_register_get_list_delete(tmp_path: Path):
    idx = SourceIndex(tmp_path / "jobs.sqlite")
    sid = new_source_id()
    idx.register(sid, "clip.mp4", "/tmp/x/clip.mp4", probe={"fps": 25.0, "frame_count": 100})
    row = idx.get(sid)
    assert row is not None
    assert row["name"] == "clip.mp4"
    assert row["path"] == "/tmp/x/clip.mp4"
    import json
    probe = json.loads(row["probe_json"])
    assert probe["frame_count"] == 100
    assert len(idx.list()) == 1
    assert idx.delete(sid) is True
    assert idx.get(sid) is None
    assert idx.delete(sid) is False


def test_missing_source_returns_none(tmp_path: Path):
    idx = SourceIndex(tmp_path / "jobs.sqlite")
    assert idx.get("s_missing") is None


def test_new_source_id_format():
    sid = new_source_id()
    assert sid.startswith("s_") and len(sid.split("_")) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_source_index.py -q`
Expected: FAIL — `ImportError: cannot import name 'SourceIndex'`

- [ ] **Step 3: Implement in store.py**

Добавить в `videoclean/store.py` (после JobIndex):

```python
def new_source_id() -> str:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"s_{stamp}_{uuid.uuid4().hex[:8]}"


class SourceIndex:
    """Uploaded videos independent of jobs. Same sqlite file as JobIndex."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    probe_json TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT_S)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def register(self, source_id: str, name: str, path: str, probe: dict[str, Any] | None = None) -> None:
        now = utc_now().isoformat()
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO sources (id, name, path, created_at, probe_json) VALUES (?, ?, ?, ?, ?)",
                (source_id, name, path, now, json.dumps(probe or {}, ensure_ascii=False)),
            )

    def list(self, limit: int = 200) -> list[sqlite3.Row]:
        with self._connect() as con:
            return list(con.execute("SELECT * FROM sources ORDER BY created_at DESC LIMIT ?", (limit,)))

    def get(self, source_id: str) -> sqlite3.Row | None:
        with self._connect() as con:
            return con.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()

    def delete(self, source_id: str) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            return cur.rowcount > 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_source_index.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/store.py tests/test_source_index.py
git commit -m "feat: SourceIndex registry for uploaded video sources"
```

---

### Task 2: jobs.source_id колонка

**Files:**
- Modify: `videoclean/store.py` (JobIndex: `_JOB_COLUMNS`, `upsert`, INSERT)
- Modify: `videoclean/application/use_cases/manage_jobs.py` (`ManageJobs.submit`)
- Test: `tests/test_source_index.py` (дополнить)

**Interfaces:**
- Produces: `JobIndex.upsert(..., source_id: str | None = None)`, `ManageJobs.submit(request_dict, input_path, output_path, prompt, job_id=None, source_id=None) -> str`. Колонка мигрируется через существующий `_JOB_COLUMNS` ALTER-паттерн.

- [ ] **Step 1: Write the failing test (дополнить tests/test_source_index.py)**

```python
def test_job_upsert_persists_source_id(tmp_path: Path):
    from videoclean.store import JobIndex

    idx = JobIndex(tmp_path / "jobs.sqlite")
    idx.upsert("job_s1", "QUEUED", source_id="s_abc")
    idx.upsert("job_s1", "RUNNING", source_id=None)  # COALESCE: не затирает
    row = idx.get("job_s1")
    assert row["source_id"] == "s_abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_source_index.py::test_job_upsert_persists_source_id -q`
Expected: FAIL — `TypeError: upsert() got an unexpected keyword argument 'source_id'`

- [ ] **Step 3: Implement**

В `videoclean/store.py`:
1. `_JOB_COLUMNS`: добавить в кортеж `("source_id", "TEXT")` — ALTER-цикл в `__init__` замигрирует существующие базы.
2. `upsert(...)`: добавить параметр `source_id: str | None = None`; в UPDATE-ветке SQL добавить `source_id = COALESCE(?, source_id),` и в кортеж параметров — `source_id` (перед `job_id`); в INSERT-ветке добавить колонку `source_id` и значение `source_id`.

В `videoclean/application/use_cases/manage_jobs.py`:
```python
    def submit(
        self,
        request_dict: dict[str, Any],
        input_path: Path,
        output_path: Path,
        prompt: str,
        job_id: str | None = None,
        source_id: str | None = None,
    ) -> str:
        ...
        self.jobs.upsert(
            job_id,
            "QUEUED",
            input_path=str(input_path),
            output_path=str(output_path),
            prompt=prompt,
            request=payload,
            source_id=source_id,
        )
        return job_id
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_source_index.py tests/test_job_store.py tests/test_job_worker.py -q`
Expected: все passed (старые тесты не ломаются — source_id опционален)

- [ ] **Step 5: Commit**

```bash
git add videoclean/store.py videoclean/application/use_cases/manage_jobs.py tests/test_source_index.py
git commit -m "feat: link jobs to sources via source_id column"
```

---

### Task 3: tracks_from_json в domain

**Files:**
- Modify: `videoclean/domain/tracks.py`
- Test: `tests/test_tracks_json.py` (новый)

**Interfaces:**
- Produces: `tracks_from_json(data: list[dict] | None) -> list[Track]` — обратная операция к `tracks_to_json` (`{"id","label","motion","part","coverage","notes","boxes":[[x1,y1,x2,y2]|None,...]}`). Некорректные строки пропускает; пустой результат → `ValueError` (domain не может поднимать PipelineError — читающий конвертирует).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracks_json.py
"""tracks_from_json round-trips tracks_to_json and rejects garbage."""
import pytest

from videoclean.domain.tracks import Track, tracks_from_json, tracks_to_json


def test_round_trip():
    tr = Track(track_id=2, label="logo", boxes=[(1, 2, 30, 40), None, (2, 3, 31, 41)], scores=[1.0, 0.0, 1.0], motion="static", notes=["a"])
    data = tracks_to_json([tr])
    out = tracks_from_json(data)
    assert len(out) == 1
    t = out[0]
    assert t.track_id == 2
    assert t.label == "logo"
    assert t.boxes == [(1, 2, 30, 40), None, (2, 3, 31, 41)]
    assert t.motion == "static"
    assert t.notes == ["a"]


def test_skips_unusable_rows():
    data = [
        {"id": 0, "label": "ok", "boxes": [[1, 1, 5, 5]]},
        {"id": 1, "label": "no boxes", "boxes": []},
        {"id": 2, "label": "all nulls", "boxes": [None, None]},
        "not a dict",
        {"id": 3, "label": "bad coords", "boxes": [["x", "y", 5, 5]]},
    ]
    out = tracks_from_json(data)
    assert [t.label for t in out] == ["ok"]


def test_empty_raises_value_error():
    with pytest.raises(ValueError):
        tracks_from_json([])
    with pytest.raises(ValueError):
        tracks_from_json(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracks_json.py -q`
Expected: FAIL — `ImportError: cannot import name 'tracks_from_json'`

- [ ] **Step 3: Implement in domain/tracks.py**

```python
def tracks_from_json(data: list[dict] | None) -> list[Track]:
    """Inverse of tracks_to_json. Unusable rows are skipped; nothing left → ValueError."""
    out: list[Track] = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        boxes: list[tuple[int, int, int, int] | None] = []
        for b in item.get("boxes") or []:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                boxes.append(None)
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(v))) for v in b)
            except (TypeError, ValueError):
                boxes.append(None)
                continue
            boxes.append((x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None)
        if not boxes or not any(b is not None for b in boxes):
            continue
        try:
            track_id = int(item.get("id") or len(out))
        except (TypeError, ValueError):
            track_id = len(out)
        out.append(
            Track(
                track_id=track_id,
                label=str(item.get("label") or f"track{len(out)}"),
                boxes=boxes,
                scores=[1.0] * len(boxes),
                motion=str(item.get("motion") or "static"),
                part=item.get("part"),
                notes=[str(n) for n in (item.get("notes") or [])],
            )
        )
    if not out:
        raise ValueError("tracks payload contains no usable tracks")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracks_json.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/domain/tracks.py tests/test_tracks_json.py
git commit -m "feat: tracks_from_json to feed detector results back into a run"
```

---

### Task 4: RunCleanup — tracks_override и masks_override

**Files:**
- Modify: `videoclean/application/config.py` (`RunCleanupRequest`)
- Modify: `videoclean/application/use_cases/run_cleanup.py`
- Test: `tests/test_run_cleanup_overrides.py` (новый)

**Interfaces:**
- Consumes: `tracks_from_json` (Task 3).
- Produces: `RunCleanupRequest.tracks_override: list[dict] | None`, `RunCleanupRequest.masks_override: list[str] | None` (пути к PNG). В отчёте: `promptParseMode` ∈ `manual-tracks | manual-masks`, `detectorUsed = "manual"`. Синтаксис: masks_override OR-ит все маски, dilate применяется, одна маска на все кадры; tracks_override пропускает parse+detect, сегментер работает по готовым трекам; verify при masks_override пропускается.

- [ ] **Step 1: Write the failing test**

Переиспользуй фейков из `tests/test_run_cleanup.py` (импортируй: `from tests.test_run_cleanup import FakeMedia, FakeParser, FakeDetector, FakeSegmenter, FakeInpainter, FakeJobs, RecordingJobs` — если импорт из теста в тест хрупок, скопируй фейки в новый файл; предпочтителен импорт). 

```python
# tests/test_run_cleanup_overrides.py
"""Manual entry points into RunCleanup: tracks_override and masks_override."""
from pathlib import Path

import cv2
import numpy as np

from tests.test_run_cleanup import (
    FakeDetector, FakeInpainter, FakeJobs, FakeMedia, FakeParser, FakeSegmenter,
)
from videoclean.adapters.progress.silent import SilentProgress
from videoclean.application.config import PipelineConfig, RunCleanupRequest
from videoclean.application.errors import PipelineError
from videoclean.application.use_cases.run_cleanup import RunCleanup
from videoclean.store import JobPaths


def _uc(tmp_path: Path, detector=None, segmenter=None, parser=None) -> RunCleanup:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return RunCleanup(
        media=FakeMedia(),
        parser=parser or FakeParser(),
        detectors=[detector] if detector else [],
        segmenter=segmenter or FakeSegmenter(),
        inpainter=FakeInpainter(),
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "j-manual",
        make_paths=JobPaths.create,
        utc_now=lambda: __import__("datetime").datetime(2026, 1, 1),
        read_image=lambda path: frame,
        write_image=lambda path, image: path.write_bytes(b"img"),
    )


class CountingDetector(FakeDetector):
    def __init__(self):
        self.calls = 0

    def discover(self, frames, queries, on_progress=None):
        self.calls += 1
        return []


class CountingParser(FakeParser):
    def __init__(self):
        self.calls = 0

    def parse(self, *args, **kwargs):
        self.calls += 1
        return super().parse(*args, **kwargs)


class CountingSegmenter(FakeSegmenter):
    def __init__(self):
        self.calls = 0

    def masks(self, frames, tracks):
        self.calls += 1
        return super().masks(frames, tracks)


def _req(tmp_path: Path, **overrides) -> RunCleanupRequest:
    base = dict(
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.mp4",
        prompt="",
        config=PipelineConfig(detectors=["grounding-dino"], formats=["mp4"], verify=False),
        overwrite=True,
        keep_workdir=True,
        job_id="j-manual",
    )
    base.update(overrides)
    return RunCleanupRequest(**base)


def _write_mask(tmp_path: Path, name: str, w: int = 8, h: int = 8) -> Path:
    m = np.zeros((h, w), dtype=np.uint8)
    m[1:4, 1:4] = 255
    p = tmp_path / name
    cv2.imwrite(str(p), m)
    return p


def test_tracks_override_skips_parse_and_detect(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    det, par, seg = CountingDetector(), CountingParser(), CountingSegmenter()
    uc = _uc(tmp_path, detector=det, segmenter=seg, parser=par)
    report = uc.execute(_req(tmp_path, tracks_override=[
        {"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4], [1, 1, 4, 4]]},
    ]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert par.calls == 0 and det.calls == 0, "parse+detect must be skipped"
    assert seg.calls == 1, "segmenter still runs on the given tracks"
    assert report["promptParseMode"] == "manual-tracks"
    assert report["detectorUsed"] == "manual"


def test_masks_override_skips_parse_detect_segment(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    det, par, seg = CountingDetector(), CountingParser(), CountingSegmenter()
    mask = _write_mask(tmp_path, "m.png")
    uc = _uc(tmp_path, detector=det, segmenter=seg, parser=par)
    report = uc.execute(_req(tmp_path, masks_override=[str(mask)]), tmp_path)
    assert report["state"] == "COMPLETED"
    assert par.calls == 0 and det.calls == 0 and seg.calls == 0
    assert report["promptParseMode"] == "manual-masks"
    assert report["detectorUsed"] == "manual"
    masks_dir = tmp_path / "jobs" / "j-manual" / "masks" / "processed"
    assert len(list(masks_dir.iterdir())) == 2, "mask tiled to every frame"


def test_manual_overrides_satisfy_prompt_check(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    det = CountingDetector()
    uc = _uc(tmp_path, detector=det)
    try:
        uc.execute(_req(tmp_path), tmp_path)
    except PipelineError as exc:
        assert "--prompt" in str(exc)
    else:
        raise AssertionError("expected PipelineError without prompt or overrides")


def test_unusable_tracks_raise(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    uc = _uc(tmp_path)
    try:
        uc.execute(_req(tmp_path, tracks_override=[{"id": 0, "label": "x", "boxes": []}]), tmp_path)
    except PipelineError:
        pass
    else:
        raise AssertionError("expected PipelineError for unusable tracks")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_cleanup_overrides.py -q`
Expected: FAIL — `TypeError: RunCleanupRequest got unexpected kwarg 'tracks_override'`

- [ ] **Step 3: Implement**

3a. `videoclean/application/config.py`, `RunCleanupRequest` — добавить поля:

```python
    targets_override: list[dict] | None = None  # manual targets: skips the prompt parser
    tracks_override: list[dict] | None = None  # manual tracks: skips parse + detect
    masks_override: list[str] | None = None  # mask PNG paths: skips parse + detect + segment
```

3b. `videoclean/application/use_cases/run_cleanup.py`:

- импорты: `import cv2`, `from videoclean.domain.tracks import Detection, Track, tracks_from_json, tracks_to_json`, `from videoclean.domain.intent import Intent, Target`.
- `execute()` — заменить проверку промпта:

```python
        if (
            not (req.prompt or "").strip()
            and not req.targets_override
            and not req.tracks_override
            and not req.masks_override
        ):
            raise PipelineError("--prompt is required (or pass manual targets/tracks/masks)")
```

- `_ensure_ready` — сигнатура `_ensure_ready(self, req: RunCleanupRequest) -> None` (вызов в `_run`: `self._ensure_ready(req)`):

```python
    def _ensure_ready(self, req: RunCleanupRequest) -> None:
        if req.masks_override:
            return
        if req.tracks_override:
            if not self.segmenter or not getattr(self.segmenter, "name", ""):
                raise AdapterUnavailable("segmenter unavailable for manual tracks")
            return
        if hasattr(self.parser, "status"):
            st = self.parser.status()
            if not st.startswith("ready"):
                raise AdapterUnavailable(f"prompt-parser llm: {st}")
        notes: list[str] = []
        ready = False
        for detector in self.detectors:
            st = detector.status()
            notes.append(f"{detector.name}: {st}")
            if st.startswith("ready"):
                ready = True
        if self.detectors and not ready:
            raise AdapterUnavailable("no detector could run. " + " | ".join(notes))
```

- в `_run` между extract-frames и существующим parse-блоком вставить ветвление; существующий parse+detect+segment код становится `else`-веткой. Структура:

```python
        used_detector: str = "manual"
        attempts: list[str] = []
        if req.masks_override:
            self.progress.start("parse", detail="manual masks")
            intent = Intent(targets=[], parse_mode="manual-masks", raw=req.prompt or "")
            self.progress.finish("parse", f"{len(req.masks_override)} masks")
            self.progress.start("detect", total=len(frames), detail="manual masks")
            base = self._or_masks(req.masks_override, (h, w))
            if base is None:
                raise PipelineError("masks_override: no readable mask images")
            if cfg.mask_dilate_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (cfg.mask_dilate_px * 2 + 1, cfg.mask_dilate_px * 2 + 1)
                )
                base = cv2.dilate(base, kernel)
            masks = [base.copy() for _ in frames]
            tracks = []
            self.progress.finish("detect", "manual masks")
        elif req.tracks_override:
            self.progress.start("parse", detail="manual tracks")
            try:
                tracks = tracks_from_json(req.tracks_override)
            except ValueError as exc:
                raise PipelineError(f"tracks_override: {exc}") from exc
            labels = list(dict.fromkeys(tr.label for tr in tracks))
            intent = Intent(
                targets=[Target(kind="object", query=lb) for lb in labels] or [Target(kind="object", query="target")],
                parse_mode="manual-tracks",
                raw=req.prompt or "",
            )
            self.progress.finish("parse", ", ".join(labels))
            self.progress.start("detect", total=len(frames), detail="manual tracks")
            masks = self.segmenter.masks(images, tracks)
            self.progress.finish("detect", f"manual: {len(tracks)} tracks")
        else:
            # === существующий код: parse-блок + detect-блок (discover, select_tracks,
            # analysis-файлы, segmenter.masks) без изменений, но присвоить
            # used_detector/attempts из _discover и оставить intent, tracks, masks ===
            ...
```

В `else`-ветке дополнительно перед `self.progress.finish("detect", ...)` (там где `detections` уже посчитан) ничего не менять — важно, что после ветвления все три пути дают: `intent`, `tracks` (list[Track], возможно `[]`), `masks`, `used_detector`, `attempts`.

- общий код после ветвления (write masks, coverage-гейт) остаётся существующим, но для `manual`-путей записи analysis-файлов делаются безопасно:

```python
        if tracks:
            (paths.root / "analysis" / "tracks.json").write_text(
                json.dumps(tracks_to_json(tracks), indent=2), encoding="utf-8"
            )
```
(заменить существующие безусловные записи tracks_raw/tracks/detector на guarded-версию; detector.json — только в else-ветке).

- report: ключи `promptParseMode: intent.parse_mode` и `detectorUsed: used_detector` уже пишутся — проверить, что `attempts` при manual = `[]`.
- verify: заменить `if cfg.verify:` на `if cfg.verify and not req.masks_override:`; в else-ветке (`verify skipped`) добавить деталь `"manual masks"`.

- новые методы класса:

```python
    def _or_masks(self, mask_paths: list[str], shape: tuple[int, int]) -> np.ndarray | None:
        """OR all user masks; resize to frame shape."""
        out: np.ndarray | None = None
        for raw in mask_paths or []:
            m = cv2.imdecode(np.fromfile(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (shape[0], shape[1]):
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            out = m if out is None else np.maximum(out, m)
        return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_run_cleanup_overrides.py tests/test_run_cleanup.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/config.py videoclean/application/use_cases/run_cleanup.py tests/test_run_cleanup_overrides.py
git commit -m "feat: manual tracks_override and masks_override entry points for RunCleanup"
```

---

### Task 5: payload-проход (cleanup_request_from_row + serialize_clean_form)

**Files:**
- Modify: `videoclean/application/use_cases/manage_jobs.py` (`cleanup_request_from_row`)
- Modify: `server/service.py` (`serialize_clean_form`)
- Test: `tests/test_cleanup_payload.py` (новый)

**Interfaces:**
- Produces: `cleanup_request_from_row` читает `targets_override`/`tracks_override`/`masks_override`/`keep_workdir` из payload. `serialize_clean_form` принимает новые поля: `llm_base_url`, `llm_api_key`, `keep_workdir`, `min_mask_coverage`, `verify_max_coverage`, `formats` (список через запятую; legacy `fmt` остаётся). Дефолты = дефолты PipelineConfig.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cleanup_payload.py
"""HTTP payload → RunCleanupRequest: overrides and new config fields pass through."""
import json
from pathlib import Path

from videoclean.application.use_cases.manage_jobs import cleanup_request_from_row, pipeline_config_from_dict


class Row(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)


def _row(payload: dict) -> Row:
    return Row({
        "id": "j1",
        "state": "QUEUED",
        "input_path": "/tmp/in.mp4",
        "output_path": "/tmp/out.mp4",
        "prompt": "p",
        "request_json": json.dumps(payload),
    })


def test_overrides_pass_through():
    req = cleanup_request_from_row(_row({
        "targets_override": [{"kind": "object", "query": "mug"}],
        "tracks_override": [{"id": 0, "label": "logo", "boxes": [[1, 1, 4, 4]]}],
        "masks_override": ["/tmp/m.png"],
        "keep_workdir": True,
    }))
    assert req.targets_override == [{"kind": "object", "query": "mug"}]
    assert req.tracks_override == [{"id": 0, "label": "logo", "boxes": [[1, 1, 4, 4]]}]
    assert req.masks_override == [Path("/tmp/m.png")]
    assert req.keep_workdir is True


def test_no_overrides_means_none():
    req = cleanup_request_from_row(_row({}))
    assert req.targets_override is None
    assert req.tracks_override is None
    assert req.masks_override is None


def test_serialize_new_fields():
    from server.service import serialize_clean_form

    out = serialize_clean_form({
        "llm_base_url": " https://api.example.com/v1 ",
        "llm_api_key": " sk-test ",
        "keep_workdir": "1",
        "min_mask_coverage": "0.01",
        "verify_max_coverage": "0.2",
        "formats": "mp4,webm",
    })
    assert out["llm_base_url"] == "https://api.example.com/v1"
    assert out["llm_api_key"] == "sk-test"
    assert out["keep_workdir"] is True
    assert out["min_mask_coverage"] == 0.01
    assert out["verify_max_coverage"] == 0.2
    assert out["formats"] == ["mp4", "webm"]


def test_serialize_defaults():
    from server.service import serialize_clean_form

    out = serialize_clean_form({})
    assert out["formats"] == ["mp4"]
    assert out["min_mask_coverage"] == 0.0004
    assert out["verify_max_coverage"] == 0.12
    assert out["keep_workdir"] is False
    assert out["llm_base_url"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cleanup_payload.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'targets_override'` (и другие)

- [ ] **Step 3: Implement**

3a. `manage_jobs.py`, `cleanup_request_from_row` — в конструктор `RunCleanupRequest(...)` добавить:

```python
        targets_override=payload.get("targets_override"),
        tracks_override=payload.get("tracks_override"),
        masks_override=[Path(str(p)) for p in (payload.get("masks_override") or [])] or None,
```

3b. `server/service.py`, `serialize_clean_form` — заменить `"formats": [fmt_name],` на:

```python
        "formats": [
            f.strip().lower()
            for f in str(data.get("formats") or data.get("fmt") or "mp4").split(",")
            if f.strip()
        ] or ["mp4"],
        "llm_base_url": str(data.get("llm_base_url") or "").strip(),
        "llm_api_key": str(data.get("llm_api_key") or "").strip(),
        "keep_workdir": _as_bool(data.get("keep_workdir"), False),
        "min_mask_coverage": (
            float(data["min_mask_coverage"]) if str(data.get("min_mask_coverage") or "").strip() else 0.0004
        ),
        "verify_max_coverage": (
            float(data["verify_max_coverage"]) if str(data.get("verify_max_coverage") or "").strip() else 0.12
        ),
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cleanup_payload.py tests/test_run_cleanup.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/use_cases/manage_jobs.py server/service.py tests/test_cleanup_payload.py
git commit -m "feat: pass overrides and full config through HTTP payload"
```

---

### Task 6: interpret_system.md шаблон

**Files:**
- Create: `videoclean/adapters/prompt/prompts/interpret_system.md`
- Modify: `videoclean/adapters/prompt/llm.py` (регистрация + публичный хелпер)
- Test: `tests/test_prompt_templates.py` (новый)

**Interfaces:**
- Produces: `interpret_system_prompt(templates_dir: str | None) -> str` в `videoclean/adapters/prompt/llm.py`; ключ `"interpret_system.md"` участвует в override через `--prompt-templates`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_templates.py
"""interpret_system.md is a builtin template with override support."""
import tempfile
from pathlib import Path

from videoclean.adapters.prompt.llm import interpret_system_prompt


def test_builtin_interpret_prompt():
    text = interpret_system_prompt(None)
    assert "JSON" in text
    assert "targets" in text


def test_custom_override():
    with tempfile.TemporaryDirectory() as d:
        custom = Path(d) / "interpret_system.md"
        custom.write_text("CUSTOM PROMPT", encoding="utf-8")
        assert interpret_system_prompt(d) == "CUSTOM PROMPT"


def test_missing_custom_falls_back():
    with tempfile.TemporaryDirectory() as d:
        assert "JSON" in interpret_system_prompt(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompt_templates.py -q`
Expected: FAIL — `ImportError: cannot import name 'interpret_system_prompt'`

- [ ] **Step 3: Implement**

3a. Создать `videoclean/adapters/prompt/prompts/interpret_system.md`:

```markdown
You interpret a video-cleanup request for Grounding DINO + SAM2.
Inputs: a user request (any language) and frames where the user painted red masks over the areas to remove.

Return ONLY JSON:
{"prompt":"<a clear removal prompt in the user's language>","targets":[{"kind":"watermark|text_overlay|object","query":"short English visual name","where":null,"motion":"any"}]}

Rules:
- prompt: rewrite/expand the user's request into a concrete removal prompt in the user's language. Name each marked area by what it visibly is. If the request is already specific, polish it, don't replace it.
- One target per distinct thing to remove. query: short English visual name the detector can search ("red logo", "caption text"). Never OCR the words into the query — describe the kind of overlay. Never vague ("overlay", "stuff").
- kind: watermark = logo/© mark; text_overlay = letters/captions; object = physical thing.
- where: top|bottom|left|right|top-left|top-right|bottom-left|bottom-right if the location is clear from the mask or request, else null.
- motion: floating if the marked thing visibly moves between frames; static if fixed; else any.
- Marked areas are ground truth: every red mask must be covered by exactly one target. If several masks show the same thing, one target covers them all.
- No prose outside JSON. No boxes, no invented ordinals.
```

3b. `videoclean/adapters/prompt/llm.py` — после `BRIDGE_SYSTEM = _read_prompt("bridge_system.md")`:

```python
INTERPRET_SYSTEM = _read_prompt("interpret_system.md")
```

в `_BUILTIN_PROMPTS` добавить `"interpret_system.md": INTERPRET_SYSTEM,`; в конец файла:

```python
def interpret_system_prompt(templates_dir: str | None = None) -> str:
    """Public accessor for BuildPrompt wiring (application layer cannot import this module)."""
    return _load_prompts(templates_dir)["interpret_system.md"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_prompt_templates.py tests/test_llm_parser.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/adapters/prompt/prompts/interpret_system.md videoclean/adapters/prompt/llm.py tests/test_prompt_templates.py
git commit -m "feat: interpret_system prompt template for mask-driven VLM interpretation"
```

---

### Task 7: BuildPrompt use case

**Files:**
- Create: `videoclean/application/use_cases/build_prompt.py`
- Test: `tests/test_build_prompt.py` (новый)

**Interfaces:**
- Consumes: `LlmClient` порт (`complete(system, user, image_jpeg, images)`), `MediaGateway.extract_frames_subset`, `targets_from_json` (run_preview).
- Produces: `BuildPromptRequest(input_path, prompt: str, annotations: list[dict]  # [{"frame": int, "mask": Path}], config: PipelineConfig, job_id: str | None)`, `BuildPrompt(media, llm, system_prompt, jobs, progress, new_job_id, make_paths, read_image, to_jpeg)` c `execute(req, data_dir) -> dict`. Отчёт: `{jobId, kind:"prompt", state, prompt, userPrompt, targets[], framesUsed[], annotatedFrames[], imagesSent, llmModel, parseMode:"interpret"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_prompt.py
"""BuildPrompt: frames + drawn masks → VLM → editable prompt + detector targets."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from videoclean.adapters.progress.silent import SilentProgress
from videoclean.application.config import PipelineConfig
from videoclean.application.errors import AdapterUnavailable, PipelineError
from videoclean.application.use_cases.build_prompt import BuildPrompt, BuildPromptRequest
from videoclean.domain.media import MediaManifest
from videoclean.store import JobPaths

W, H = 8, 8


class FakeMedia:
    name = "ffmpeg"

    def __init__(self, n: int = 5):
        self.n = n

    def probe(self, path: Path) -> MediaManifest:
        return MediaManifest(
            path=path, duration_s=1.0, width=W, height=H, fps=5.0, fps_ratio="5/1",
            video_codec="h264", pix_fmt="yuv420p", frame_count=self.n, has_audio=False, audio_codec=None,
        )

    def extract_frames_subset(self, src, indices, dest_dir, log_file) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for i in indices:
            p = dest_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(p), np.zeros((H, W, 3), dtype=np.uint8))
            out.append(p)
        return out


class FakeLlm:
    name = "fake"
    model = "fake-vision"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def status(self) -> str:
        return "ready (fake)"

    def complete(self, system, user, image_jpeg=None, images=None):
        self.calls.append({"system": system, "user": user, "images": images})
        return self.reply


REPLY = (
    'Прекрасный выбор! {"prompt": "убери логотип в правом нижнем углу", '
    '"targets": [{"kind": "watermark", "query": "channel logo", "where": "bottom-right", "motion": "static"}]}'
)


def _build_prompt(tmp_path: Path, llm: FakeLlm) -> BuildPrompt:
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    def to_jpeg(img) -> bytes:
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    return BuildPrompt(
        media=FakeMedia(),
        llm=llm,
        system_prompt="SYSTEM",
        jobs=FakeJobs(),
        progress=SilentProgress(),
        new_job_id=lambda: "j-prompt",
        make_paths=JobPaths.create,
        read_image=lambda path: frame,
        to_jpeg=to_jpeg,
    )


class FakeJobs:
    def upsert(self, *args, **kwargs) -> None:
        return None


def _write_mask(tmp_path: Path, name: str) -> Path:
    m = np.zeros((H, W), dtype=np.uint8)
    m[1:4, 1:4] = 255
    p = tmp_path / name
    cv2.imwrite(str(p), m)
    return p


def _req(tmp_path: Path, prompt: str = "", annotations=None) -> BuildPromptRequest:
    return BuildPromptRequest(
        input_path=tmp_path / "in.mp4",
        prompt=prompt,
        annotations=annotations or [],
        config=PipelineConfig(prompt_frame_stride=4, prompt_frame_max=8),
        job_id="j-prompt",
    )


def test_masks_and_text_are_interpreted(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    llm = FakeLlm(REPLY)
    out = _build_prompt(tmp_path, llm).execute(_req(tmp_path, prompt="убери это", annotations=[
        {"frame": 3, "mask": str(mask)},
    ]), tmp_path)
    assert out["state"] == "COMPLETED"
    assert out["prompt"] == "убери логотип в правом нижнем углу"
    assert out["targets"][0]["query"] == "channel logo"
    assert out["framesUsed"] == [3]
    assert out["annotatedFrames"] == [3]
    assert out["llmModel"] == "fake-vision"
    assert len(llm.calls[0]["images"]) == 1, "one jpeg per requested frame"
    assert "убери это" in llm.calls[0]["user"]
    assert "frame 3" in llm.calls[0]["user"]
    assert "red overlay" in llm.calls[0]["user"]


def test_text_only_uses_sampled_frames(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    llm = FakeLlm(REPLY)
    out = _build_prompt(tmp_path, llm).execute(_req(tmp_path, prompt="чисти"), tmp_path)
    assert out["framesUsed"] == [0, 4], "stride 4 over 5 frames"
    assert len(llm.calls[0]["images"]) == 2


def test_empty_input_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    with pytest.raises(PipelineError):
        _build_prompt(tmp_path, FakeLlm(REPLY)).execute(_req(tmp_path), tmp_path)


def test_unavailable_llm_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")

    class Dead(FakeLlm):
        def status(self):
            return "unavailable: down"

    with pytest.raises(AdapterUnavailable):
        _build_prompt(tmp_path, Dead(REPLY)).execute(_req(tmp_path, annotations=[
            {"frame": 0, "mask": str(mask)},
        ]), tmp_path)


def test_nonjson_reply_fails(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    mask = _write_mask(tmp_path, "m.png")
    with pytest.raises(PipelineError):
        _build_prompt(tmp_path, FakeLlm("no json here")).execute(_req(tmp_path, annotations=[
            {"frame": 0, "mask": str(mask)},
        ]), tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_prompt.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'videoclean.application.use_cases.build_prompt'`

- [ ] **Step 3: Implement `videoclean/application/use_cases/build_prompt.py`**

```python
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from videoclean.application.config import PipelineConfig
from videoclean.application.errors import AdapterUnavailable, JobCancelled, PipelineError
from videoclean.application.frames_sample import sample_frame_indices
from videoclean.application.ports.jobs import JobStore
from videoclean.application.ports.media import MediaGateway
from videoclean.application.ports.progress import ProgressPort
from videoclean.application.use_cases.run_preview import targets_from_json
from videoclean.domain.intent import Target


@dataclass
class BuildPromptRequest:
    input_path: Path
    prompt: str
    annotations: list[dict]  # [{"frame": int, "mask": str | Path}]
    config: PipelineConfig
    job_id: str | None = None


@dataclass
class PromptPaths:
    root: Path
    frames_dir: Path
    logs_dir: Path
    ffmpeg_log: Path
    report_file: Path

    @classmethod
    def create(cls, root: Path) -> "PromptPaths":
        return cls(
            root=root,
            frames_dir=root / "process" / "frames",
            logs_dir=root / "logs",
            ffmpeg_log=root / "logs" / "ffmpeg.log",
            report_file=root / "output" / "report.json",
        )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class BuildPrompt:
    """User prompt and/or drawn masks → VLM → editable prompt + detector targets."""

    def __init__(
        self,
        *,
        media: MediaGateway,
        llm,
        system_prompt: str,
        jobs: JobStore,
        progress: ProgressPort,
        new_job_id: callable,
        make_paths: callable,
        read_image: callable,
        to_jpeg: callable,
    ) -> None:
        self.media = media
        self.llm = llm
        self._system_prompt = system_prompt
        self.jobs = jobs
        self.progress = progress
        self._new_job_id = new_job_id
        self._make_paths = make_paths
        self._read_image = read_image
        self._to_jpeg = to_jpeg

    def execute(self, req: BuildPromptRequest, data_dir: Path) -> dict:
        prompt = (req.prompt or "").strip()
        if not prompt and not req.annotations:
            raise PipelineError("build-prompt: provide text or at least one mask annotation")
        if not req.input_path.is_file():
            raise FileNotFoundError(req.input_path)
        job_id = req.job_id or self._new_job_id()
        paths = self._make_paths(data_dir / "jobs" / job_id)
        self.jobs.upsert(job_id, "RUNNING", input_path=str(req.input_path), prompt=prompt)
        report: dict = {"jobId": job_id, "kind": "prompt", "state": "RUNNING", "prompt": prompt}
        try:
            out = self._run(req, job_id, paths, report)
            self.jobs.upsert(job_id, "COMPLETED", report=out)
            return out
        except JobCancelled:
            report["state"] = "CANCELLED"
            report["error"] = "cancelled"
            self.jobs.upsert(job_id, "CANCELLED", report=report, error="cancelled")
            raise
        except Exception as exc:
            report["state"] = "FAILED"
            report["error"] = str(exc)
            try:
                paths.report_file.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
            self.jobs.upsert(job_id, "FAILED", report=report, error=str(exc)[:1500])
            raise

    def _run(self, req: BuildPromptRequest, job_id: str, paths, report: dict) -> dict:
        manifest = self.media.probe(req.input_path)
        ann_by_frame = self._annotations(req)
        idxs = (
            sorted(ann_by_frame)
            if ann_by_frame
            else sample_frame_indices(manifest.frame_count, req.config.prompt_frame_stride, req.config.prompt_frame_max)
        )
        if not idxs:
            raise PipelineError("build-prompt: no frames to look at")

        paths.frames_dir.mkdir(parents=True, exist_ok=True)
        self.progress.start("normalize", total=len(idxs))
        frame_paths = self.media.extract_frames_subset(req.input_path, idxs, paths.frames_dir, paths.ffmpeg_log)
        images = [self._read_image(p) for p in frame_paths]
        bad = [str(p) for p, img in zip(frame_paths, images) if img is None or not getattr(img, "size", 1)]
        if bad:
            raise PipelineError(f"build-prompt: failed to read {len(bad)} frame(s), first: {bad[0]}")
        jpegs = [self._to_jpeg(self._overlay(img, ann_by_frame[idx])) for img, idx in zip(images, idxs)]
        self.progress.finish("normalize", f"{len(jpegs)} frames")

        st = self.llm.status()
        if not st.startswith("ready"):
            raise AdapterUnavailable(f"prompt-parser llm: {st}")
        self.progress.start("parse", detail=f"llm {self.llm.model}")
        raw = self.llm.complete(self._system_prompt, self._user_message(req.prompt, ann_by_frame, idxs), images=jpegs)
        data = _extract_json(raw)
        if data is None:
            raise PipelineError("build-prompt: model returned no JSON object")
        targets_json = data.get("targets") or []
        targets = targets_from_json(targets_json) if targets_json else []
        out_prompt = str(data.get("prompt") or "").strip()
        if not out_prompt:
            out_prompt = ", ".join(t.query for t in targets) or (req.prompt or "").strip()
        self.progress.finish("parse", ", ".join(t.query for t in targets) or "(no targets)")

        report.update(
            {
                "state": "COMPLETED",
                "prompt": out_prompt,
                "userPrompt": (req.prompt or "").strip(),
                "targets": [_t_json(t) for t in targets],
                "framesUsed": idxs,
                "annotatedFrames": sorted(ann_by_frame),
                "imagesSent": len(jpegs),
                "llmModel": self.llm.model,
                "parseMode": "interpret",
            }
        )
        paths.report_file.parent.mkdir(parents=True, exist_ok=True)
        paths.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.rmtree(paths.frames_dir, ignore_errors=True)
        return report

    def _annotations(self, req: BuildPromptRequest) -> dict[int, Path]:
        out: dict[int, Path] = {}
        for a in req.annotations or []:
            try:
                frame = int(a.get("frame"))
            except (TypeError, ValueError):
                continue
            mask = Path(str(a.get("mask") or ""))
            if frame >= 0 and str(mask):
                out[frame] = mask
        return out

    @staticmethod
    def _overlay(img: np.ndarray, mask_path: Path) -> np.ndarray:
        m = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return img
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = img.copy()
        out[m > 0] = (0.5 * out[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        return out

    @staticmethod
    def _user_message(prompt: str, ann_by_frame: dict[int, Path], idxs: list[int]) -> str:
        lines = [
            f"User request: {prompt.strip()}" if prompt.strip() else "User request: (none — the user only marked areas on frames)"
        ]
        for i, idx in enumerate(idxs):
            if idx in ann_by_frame:
                lines.append(f"Image {i + 1}: frame {idx}. The red overlay marks what the user wants removed.")
            else:
                lines.append(f"Image {i + 1}: frame {idx}.")
        return "\n".join(lines)


def _t_json(t: Target) -> dict:
    return {
        "kind": t.kind,
        "query": t.query,
        "where": t.where,
        "motion": t.motion,
        "ordinal": t.ordinal,
        "from_side": t.from_side,
        "frames": list(t.frames) if t.frames else None,
    }
```

Примечания: красный оверлей `[0, 0, 255]` — BGR (read_bgr возвращает BGR). `targets_from_json` поднимает PipelineError при пустом списке — обёрнуто вызовом только если `targets_json` непуст; модель может вернуть пустой targets → валидный отчёт без таргетов (пользователь правит промпт).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_build_prompt.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/use_cases/build_prompt.py tests/test_build_prompt.py
git commit -m "feat: BuildPrompt use case (prompt+masks -> VLM -> prompt+targets)"
```

---

### Task 8: composition + worker dispatch kind="prompt"

**Files:**
- Modify: `videoclean/composition.py` (`build_build_prompt`, `build_job_worker`)
- Modify: `videoclean/application/jobs/worker.py`
- Test: `tests/test_job_worker.py` (дополнить)

**Interfaces:**
- Consumes: `BuildPrompt` (Task 7), `interpret_system_prompt` (Task 6), `bgr_to_jpeg` (`videoclean/adapters/prompt/frames.py`).
- Produces: `build_build_prompt(cfg, progress, jobs=None, job_id=None)`; `JobWorker(..., build_prompt_runner=...)`; payload `kind="prompt"` → отчёт BuildPrompt.

- [ ] **Step 1: Write the failing test (дополнить tests/test_job_worker.py)**

Проверь существующие тесты файла и добавь в том же стиле (фабрика-раннер — заглушка):

```python
def test_worker_dispatches_prompt_kind(tmp_path):
    """kind=prompt jobs go to the prompt runner, not cleanup."""
    from pathlib import Path as _P

    jobs = JobIndex(tmp_path / "jobs.sqlite")
    seen = {"prompt": 0, "cleanup": 0}

    class FakeRunner:
        def __init__(self, *args):
            pass

        def execute(self, req, data_dir):
            return {"jobId": req.job_id, "state": "COMPLETED", "kind": "prompt"}

    def cleanup_factory(*a):
        seen["cleanup"] += 1
        return FakeRunner()

    def prompt_factory(*a):
        seen["prompt"] += 1
        return FakeRunner()

    worker = JobWorker(tmp_path, jobs, build_runner=cleanup_factory, build_prompt_runner=prompt_factory)
    jobs.upsert(
        "job-p", "QUEUED",
        input_path=str(tmp_path / "in.mp4"),
        request={"kind": "prompt", "prompt": "x", "annotations": [{"frame": 0, "mask": str(tmp_path / "m.png")}]},
        prompt="x",
    )
    worker.start()
    worker.stop(timeout=5)
    assert seen["prompt"] == 1 and seen["cleanup"] == 0
    assert jobs.get("job-p")["state"] == "COMPLETED"
```

(импорты JobIndex/JobWorker в файле уже есть — сверить с существующими тестами.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_job_worker.py::test_worker_dispatches_prompt_kind -q`
Expected: FAIL — видимо `TypeError: unexpected kwarg build_prompt_runner` либо промпт-джоб уходит в cleanup-фабрику

- [ ] **Step 3: Implement**

3a. `worker.py`: `__init__` — параметр `build_prompt_runner: Callable[..., Any] | None = None`, в теле `self.build_prompt_runner = build_prompt_runner`. В `_run_one` после чтения payload:

```python
        if payload.get("kind") == "preview":
            self._run_preview(job_id, row, payload)
            return
        if payload.get("kind") == "prompt":
            self._run_prompt(job_id, row, payload)
            return
```

Метод (зеркало `_run_preview`):

```python
    def _run_prompt(self, job_id: str, row, payload: dict) -> None:
        from videoclean.application.use_cases.build_prompt import BuildPromptRequest

        try:
            if self.build_prompt_runner is None:
                raise RuntimeError("prompt runner is not configured")
            config = self._config_from_payload(payload)
            input_path = Path(row["input_path"] or payload.get("input_path") or "")
            annotations = [
                {"frame": int(a["frame"]), "mask": Path(str(a.get("mask") or ""))}
                for a in (payload.get("annotations") or [])
                if str(a.get("mask") or "").strip()
            ]
            req = BuildPromptRequest(
                input_path=input_path,
                prompt=payload.get("prompt") or "",
                annotations=annotations,
                config=config,
                job_id=job_id,
            )
            runner = self.build_prompt_runner(config, None, self.jobs, job_id)
            report = runner.execute(req, self.data_dir)
        except JobCancelled:
            self.jobs.upsert(job_id, "CANCELLED", error="cancelled")
            return
        except Exception as exc:  # noqa: BLE001 — queue must isolate job failures
            current = self.jobs.get(job_id)
            if current is None:
                return
            if current["state"] in {"FAILED", "CANCELLED"}:
                return
            if self.jobs.is_cancel_requested(job_id):
                self.jobs.upsert(job_id, "CANCELLED", error=str(exc)[:500])
            else:
                self.jobs.upsert(job_id, "FAILED", error=str(exc)[:800])
            return
        current = self.jobs.get(job_id)
        if current is not None and current["state"] == "RUNNING":
            self.jobs.upsert(job_id, "COMPLETED", report=report)
```

3b. `composition.py` — `build_build_prompt` и проводка в `build_job_worker`:

```python
def build_build_prompt(
    cfg: PipelineConfig,
    progress: ProgressPort | None,
    jobs: JobIndex | None = None,
    job_id: str | None = None,
):
    from videoclean.adapters.prompt.frames import bgr_to_jpeg
    from videoclean.adapters.prompt.llm import interpret_system_prompt
    from videoclean.application.use_cases.build_prompt import BuildPrompt

    id_factory = (lambda: job_id) if job_id else new_job_id
    return BuildPrompt(
        media=FFmpegMedia(),
        llm=resolve_llm(cfg),
        system_prompt=interpret_system_prompt(cfg.prompt_templates),
        jobs=jobs or JobIndex(Path.home() / ".videoclean" / "jobs.sqlite"),
        progress=progress,
        new_job_id=id_factory,
        make_paths=PromptPathsFactory,
        read_image=read_bgr,
        write_image=write_bgr,
        to_jpeg=bgr_to_jpeg,
    )
```

`PromptPathsFactory` рядом с `PreviewPathsFactory`:

```python
def PromptPathsFactory(root: Path):
    from videoclean.application.use_cases.build_prompt import PromptPaths

    return PromptPaths.create(root)
```

(`BuildPrompt` не использует write_image — передаётся для симметрии; можно убрать из конструктора и из этой фабрики. Решение: НЕ передавать write_image — из Task 7 сигнатуры write_image нет. Итог: `build_build_prompt` без write_image.)

В `build_job_worker` — третий фаб:

```python
    def prompt_factory(cfg, progress, jobs, job_id):
        return build_build_prompt(
            cfg,
            progress or ProgressBridge(jobs, job_id),
            jobs,
            job_id=job_id,
        )

    return JobWorker(
        data_dir=data_dir, jobs=jobs,
        build_runner=factory, build_preview_runner=preview_factory,
        build_prompt_runner=prompt_factory,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_job_worker.py tests/test_cli_serve_helpers.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add videoclean/composition.py videoclean/application/jobs/worker.py tests/test_job_worker.py
git commit -m "feat: worker dispatches kind=prompt jobs to BuildPrompt"
```

---

### Task 9: Sources API — CRUD, video, кадры

**Files:**
- Modify: `server/app_state.py` (AppState.sources)
- Modify: `server/service.py` (`register_source`, `source_dict`, `source_frame_path`)
- Modify: `server/fastapi_app.py` (роуты)
- Test: `tests/test_api_sources.py` (новый)

**Interfaces:**
- Consumes: `SourceIndex` (Task 1), `FFmpegMedia.probe`.
- Produces: `POST /api/sources` (multipart `video`) → 201 `{id, name, createdAt, probe}`; `GET /api/sources` → список; `GET /api/sources/{id}` → dict|404; `DELETE /api/sources/{id}` → 409 при активных job'ах этого source_id, иначе удаляет строку и каталог; `GET /api/sources/{id}/video` → FileResponse; `GET /api/sources/{id}/frames/{n}` и `/frames/{n}.jpg` → JPEG кадра (кэш `sources/{id}/frames/{n:06d}.jpg`, ffmpeg `-ss n/fps`). `source_frame_path(state, source_row, n: int) -> Path | None` — None вне диапазона/при ошибке ffmpeg.

- [ ] **Step 1: Write the failing test**

Тест использует реальный ffmpeg-клип (как прод-путь):

```python
# tests/test_api_sources.py
"""Sources API: upload, list, delete, video stream, single-frame extraction."""
import base64
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_from_env


def _make_clip(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=60,
    )


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    token = auth_from_env()
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client, state, tmp_path


def _upload(client, tmp_path: Path, name="clip.mp4"):
    clip = tmp_path / name
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/sources", files={"video": (name, f, "video/mp4")})
    assert resp.status_code == 201, resp.text
    return resp.json(), clip


def test_upload_list_get_delete(client, tmp_path: Path):
    client, state, _ = client
    src, clip = _upload(client, tmp_path)
    assert src["name"] == "clip.mp4"
    assert src["probe"]["frame_count"] == 5
    assert src["probe"]["width"] == 64
    lst = client.get("/api/sources").json()
    assert [s["id"] for s in lst] == [src["id"]]
    got = client.get(f"/api/sources/{src['id']}").json()
    assert got["id"] == src["id"]
    assert client.get("/api/sources/s_missing").status_code == 404
    assert client.delete(f"/api/sources/{src['id']}").status_code == 200
    assert client.get(f"/api/sources/{src['id']}").status_code == 404
    assert not clip.exists(), "source dir removed with the row"


def test_rejects_non_video(client, tmp_path: Path):
    client, state, _ = client
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with bad.open("rb") as f:
        resp = client.post("/api/sources", files={"video": ("x.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_video_stream_and_frame(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    resp = client.get(f"/api/sources/{src['id']}/video")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/")
    for path in (f"/api/sources/{src['id']}/frames/2", f"/api/sources/{src['id']}/frames/2.jpg"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/sources/{src['id']}/frames/99").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_sources.py -q`
Expected: FAIL — 404 (роутов нет) / `AttributeError: AppState has no attribute 'sources'`

- [ ] **Step 3: Implement**

3a. `server/app_state.py`: импорт `from videoclean.store import JobIndex, SourceIndex`; в `AppState` поле `sources: SourceIndex | None = None`; в `build_app_state` — `sources = SourceIndex(data_dir / "jobs.sqlite")` и в `AppState(...)` — `sources=sources`.

3b. `server/service.py` — новые функции (импорты: `new_job_id, new_source_id` уже есть/добавить `new_source_id`):

```python
def source_dict(row) -> dict[str, Any]:
    probe = _as_dict(row["probe_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "createdAt": row["created_at"],
        "probe": probe,
        "video_url": f"/api/sources/{row['id']}/video",
        "annotations_url": f"/api/sources/{row['id']}/annotations",
    }


def register_source(state: AppState, tmp: Path, original_name: str) -> str:
    from videoclean.adapters.media.ffmpeg import FFmpegMedia

    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise PipelineError(f"unsupported video type {suffix}")
    source_id = new_source_id()
    dest_dir = Path(state.data_dir) / "sources" / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"input{suffix}"
    shutil.copy2(tmp, dest)
    try:
        m = FFmpegMedia().probe(dest)
    except Exception:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    probe = {
        "fps": m.fps, "duration_s": m.duration_s, "width": m.width, "height": m.height,
        "frame_count": m.frame_count, "has_audio": m.has_audio,
    }
    state.sources.register(source_id, Path(original_name).name or dest.name, str(dest), probe=probe)
    return source_id


def source_frame_path(state: AppState, source_row, n: int) -> Path | None:
    probe = _as_dict(source_row["probe_json"])
    fc = int(probe.get("frame_count") or 0)
    if n < 0 or (fc and n >= fc):
        return None
    src = Path(source_row["path"])
    out = src.parent / "frames" / f"{n:06d}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = float(probe.get("fps") or 0) or 25.0
    import subprocess

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{n / fps:.6f}", "-i", str(src),
        "-frames:v", "1", "-q:v", "2", str(out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        return None
    return out
```

`VIDEO_SUFFIXES` перенести из `fastapi_app.py` в `service.py` (или импортировать) — чтобы `register_source` его видел; fastapi_app импортирует оттуда.

3c. `server/fastapi_app.py` — роуты (в `create_app`):

```python
    @app.post("/api/sources")
    async def create_source(st: AppState = Depends(get_state), video: UploadFile = File(...)):
        if not (video.filename or "").strip():
            raise HTTPException(400, "video file is required")
        tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"src_{os.getpid()}_{video.filename}"
        try:
            await _save_upload(video, tmp)
            sid = register_source(st, tmp, video.filename)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)
        return JSONResponse(source_dict(st.sources.get(sid)), status_code=201)

    @app.get("/api/sources")
    def list_sources(st: AppState = Depends(get_state)):
        return [source_dict(row) for row in st.sources.list()]

    @app.get("/api/sources/{source_id}")
    def get_source(source_id: str, st: AppState = Depends(get_state)):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        return source_dict(row)

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id: str, st: AppState = Depends(get_state)):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        for job_row in st.jobs.list_jobs_full(limit=10_000):
            if (row["id"] if "source_id" not in job_row.keys() else job_row["source_id"]) == source_id \
                    and job_row["state"] in {"QUEUED", "RUNNING"}:
                raise HTTPException(409, "source has active jobs; cancel them first")
        st.sources.delete(source_id)
        shutil.rmtree(Path(st.data_dir) / "sources" / source_id, ignore_errors=True)
        return {"ok": True, "id": source_id}

    @app.get("/api/sources/{source_id}/video")
    def source_video(source_id: str, st: AppState = Depends(get_state)):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(404, "source file missing")
        media_type = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.get("/api/sources/{source_id}/frames/{n}.jpg")
    @app.get("/api/sources/{source_id}/frames/{n}")
    def source_frame(source_id: str, n: int, st: AppState = Depends(get_state)):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        path = source_frame_path(st, row, n)
        if path is None:
            raise HTTPException(404, f"frame {n} unavailable")
        return FileResponse(path, media_type="image/jpeg")
```

Импорты в fastapi_app: `register_source, source_dict, source_frame_path` из service; `shutil`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_sources.py tests/test_api_static.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add server/app_state.py server/service.py server/fastapi_app.py tests/test_api_sources.py
git commit -m "feat: sources API — upload, list, delete, video stream, frame extraction"
```

---

### Task 10: Masks API + annotations

**Files:**
- Modify: `server/service.py` (`save_mask`, `mask_path`, `delete_mask`, `annotations_payload`)
- Modify: `server/fastapi_app.py` (роуты)
- Test: `tests/test_api_sources.py` (дополнить)

**Interfaces:**
- Produces: `PUT /api/sources/{id}/masks/{n}` (multipart: `mask` PNG + `strokes` JSON-строка) → 201 `{ok, frame}`; `GET .../masks/{n}` → PNG; `DELETE .../masks/{n}` → `{ok}`; `GET /api/sources/{id}/annotations` → `{frames: [{frame, url, strokes, updatedAt}]}`. Файлы: `sources/{id}/masks/{n:06d}.png` + `.json`. Проверка диапазона по probe (400 вне диапазона); PNG-магия проверяется.

- [ ] **Step 1: Write the failing test (дополнить tests/test_api_sources.py)**

```python
def _png_bytes() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.zeros((64, 64), np.uint8))
    assert ok
    return buf.tobytes()


def test_mask_put_get_delete_and_annotations(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    sid = src["id"]
    png = _png_bytes()
    resp = client.put(
        f"/api/sources/{sid}/masks/3",
        files={"mask": ("m.png", png, "image/png")},
        data={"strokes": '[{"tool":"brush","points":[1,2,3],"size":40}]'},
    )
    assert resp.status_code == 201, resp.text
    got = client.get(f"/api/sources/{sid}/masks/3")
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"
    ann = client.get(f"/api/sources/{sid}/annotations").json()
    assert ann["frames"] == [{
        "frame": 3,
        "url": f"/api/sources/{sid}/masks/3",
        "strokes": [{"tool": "brush", "points": [1, 2, 3], "size": 40}],
    }]
    assert client.delete(f"/api/sources/{sid}/masks/3").status_code == 200
    assert client.get(f"/api/sources/{sid}/masks/3").status_code == 404
    assert client.get(f"/api/sources/{sid}/annotations").json()["frames"] == []


def test_mask_out_of_range(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    resp = client.put(
        f"/api/sources/{src['id']}/masks/99",
        files={"mask": ("m.png", _png_bytes(), "image/png")},
        data={"strokes": "[]"},
    )
    assert resp.status_code == 400


def test_mask_rejects_non_png(client, tmp_path: Path):
    client, state, _ = client
    src, _ = _upload(client, tmp_path)
    resp = client.put(
        f"/api/sources/{src['id']}/masks/0",
        files={"mask": ("m.png", b"not a png", "image/png")},
        data={"strokes": "[]"},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_sources.py -q`
Expected: новые тесты FAIL (404), старые passed

- [ ] **Step 3: Implement**

3a. `server/service.py`:

```python
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def source_masks_dir(state: AppState, source_row) -> Path:
    return Path(state.data_dir) / "sources" / source_row["id"] / "masks"


def save_mask(state: AppState, source_row, n: int, png: bytes, strokes: str) -> None:
    probe = _as_dict(source_row["probe_json"])
    fc = int(probe.get("frame_count") or 0)
    if fc and not 0 <= n < fc:
        raise PipelineError(f"frame {n} out of range (0..{fc - 1})")
    if not png.startswith(PNG_MAGIC):
        raise PipelineError("mask must be a PNG image")
    directory = source_masks_dir(state, source_row)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        strokes_data = json.loads(strokes) if strokes and strokes.strip() else []
    except json.JSONDecodeError as exc:
        raise PipelineError(f"strokes must be JSON: {exc}") from exc
    (directory / f"{n:06d}.png").write_bytes(png)
    (directory / f"{n:06d}.json").write_text(json.dumps(strokes_data, ensure_ascii=False), encoding="utf-8")


def mask_path(state: AppState, source_row, n: int) -> Path | None:
    p = source_masks_dir(state, source_row) / f"{n:06d}.png"
    return p if p.is_file() else None


def delete_mask(state: AppState, source_row, n: int) -> bool:
    directory = source_masks_dir(state, source_row)
    png = directory / f"{n:06d}.png"
    meta = directory / f"{n:06d}.json"
    existed = png.is_file()
    png.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    return existed


def annotations_payload(state: AppState, source_row) -> dict[str, Any]:
    out = []
    directory = source_masks_dir(state, source_row)
    for p in sorted(directory.glob("*.png")):
        try:
            frame = int(p.stem)
        except ValueError:
            continue
        meta = p.with_suffix(".json")
        try:
            strokes = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            strokes = []
        out.append({
            "frame": frame,
            "url": f"/api/sources/{source_row['id']}/masks/{frame}",
            "strokes": strokes if isinstance(strokes, list) else [],
            "updatedAt": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"frames": out}
```

3b. Роуты в `fastapi_app.py`:

```python
    def _source_or_404(st, source_id):
        row = st.sources.get(source_id)
        if row is None:
            raise HTTPException(404, f"unknown source {source_id}")
        return row

    @app.put("/api/sources/{source_id}/masks/{n}")
    async def put_mask(source_id: str, n: int, st: AppState = Depends(get_state),
                       mask: UploadFile = File(...), strokes: str = Form("")):
        row = _source_or_404(st, source_id)
        data = await mask.read()
        try:
            save_mask(st, row, n, data, strokes)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse({"ok": True, "frame": n}, status_code=201)

    @app.get("/api/sources/{source_id}/masks/{n}")
    def get_mask(source_id: str, n: int, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        path = mask_path(st, row, n)
        if path is None:
            raise HTTPException(404, f"mask for frame {n} not found")
        return FileResponse(path, media_type="image/png")

    @app.delete("/api/sources/{source_id}/masks/{n}")
    def remove_mask(source_id: str, n: int, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        if not delete_mask(st, row, n):
            raise HTTPException(404, f"mask for frame {n} not found")
        return {"ok": True, "frame": n}

    @app.get("/api/sources/{source_id}/annotations")
    def source_annotations(source_id: str, st: AppState = Depends(get_state)):
        row = _source_or_404(st, source_id)
        return annotations_payload(st, row)
```

(`datetime`, `timezone` уже импортированы в service.py.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_sources.py -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add server/service.py server/fastapi_app.py tests/test_api_sources.py
git commit -m "feat: per-frame mask upload/read/delete + annotations listing"
```

---

### Task 11: POST /api/jobs — kind, source_id, overrides

**Files:**
- Modify: `server/service.py` (`queue_source_run`, `queue_source_preview`, `queue_source_prompt`)
- Modify: `server/fastapi_app.py` (`create_job`)
- Test: `tests/test_api_jobs_kinds.py` (новый)

**Interfaces:**
- Consumes: `serialize_clean_form` (Task 5), `ManageJobs.submit(source_id=...)` (Task 2).
- Produces: `POST /api/jobs` multipart c новыми полями: `kind` (`run|preview|prompt`, дефолт run), `source_id`, `targets` (JSON list → `targets_override`), `tracks` (JSON list → `tracks_override`), `masks` (список int через запятую → пути масок source → `masks_override`), `start/count/stride/indices/mode` для preview. Upload-путь (kind=run c файлом) регистрирует source и привязывает job к нему. Взаимоисключаемость targets/tracks/masks → 400. Ответ 201 job_dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_jobs_kinds.py
"""POST /api/jobs: kinds run/preview/prompt, source_id, manual overrides."""
import base64
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_from_env


def _make_clip(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=60,
    )


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client, state, tmp_path


def _source(client, tmp_path: Path) -> dict:
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/sources", files={"video": ("clip.mp4", f, "video/mp4")})
    assert resp.status_code == 201
    return resp.json()


def _payload(row) -> dict:
    return json.loads(row["request_json"])


def test_run_from_source_with_tracks_override(client):
    client, state, _ = client
    src = _source(client, tmp_path=None) if False else _source(client, state.data_dir.parent)
    resp = client.post(
        "/api/jobs",
        data={
            "kind": "run", "source_id": src["id"],
            "tracks": json.dumps([{"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4]]}]),
            "prompt": "",
        },
    )
    assert resp.status_code == 201, resp.text
    row = state.jobs.get(resp.json()["id"])
    payload = _payload(row)
    assert payload["kind"] == "run"
    assert payload["tracks_override"] == [{"id": 0, "label": "logo", "motion": "static", "boxes": [[1, 1, 4, 4]]}]
    assert row["source_id"] == src["id"]
    assert Path(payload["input_path"]).parent.parent.name == src["id"], "input lives under the source dir"


def test_overrides_are_mutually_exclusive(client):
    client, state, _ = client
    src = _source(client, state.data_dir.parent)
    resp = client.post(
        "/api/jobs",
        data={
            "kind": "run", "source_id": src["id"],
            "targets": json.dumps([{"kind": "object", "query": "mug"}]),
            "tracks": json.dumps([{"id": 0, "label": "x", "boxes": [[1, 1, 4, 4]]}]),
            "prompt": "p",
        },
    )
    assert resp.status_code == 400
    assert "mutually exclusive" in resp.json()["detail"]


def test_preview_from_source_all_frames(client):
    client, state, _ = client
    src = _source(client, state.data_dir.parent)
    resp = client.post(
        "/api/jobs",
        data={"kind": "preview", "source_id": src["id"], "prompt": "убери", "all": "1"},
    )
    assert resp.status_code == 201, resp.text
    payload = _payload(state.jobs.get(resp.json()["id"]))
    assert payload["kind"] == "preview"
    assert payload["start"] == 0 and payload["count"] == 5, "all-frames expanded"
    assert payload["segmenter"] == "sam2", "detect stage forces the per-frame segmenter"


def test_prompt_job_collects_source_masks(client):
    client, state, tmp_path = client
    src = _source(client, state.data_dir.parent)
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.zeros((64, 64), np.uint8))
    assert ok
    masks_dir = state.data_dir / "sources" / src["id"] / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / "000002.png").write_bytes(buf.tobytes())
    resp = client.post("/api/jobs", data={"kind": "prompt", "source_id": src["id"], "prompt": ""})
    assert resp.status_code == 201, resp.text
    payload = _payload(state.jobs.get(resp.json()["id"]))
    assert payload["kind"] == "prompt"
    assert payload["annotations"] == [{
        "frame": 2, "mask": str(masks_dir / "000002.png"),
    }]


def test_run_upload_registers_source(client):
    client, state, tmp_path = client
    clip = tmp_path / "up.mp4"
    _make_clip(clip)
    with clip.open("rb") as f:
        resp = client.post("/api/jobs", files={"video": ("up.mp4", f, "video/mp4")}, data={"prompt": "убери логотип"})
    assert resp.status_code == 201, resp.text
    job = resp.json()
    row = state.jobs.get(job["id"])
    assert row["source_id"], "upload created an implicit source"
    assert state.sources.get(row["source_id"]) is not None
```

Примечание: `_source(client, ...)` сигнатура — упростить до `_source(client, tmp_path)`; в тестах выше передавать реальный tmp_path (исправь при написании файла — вызовы `_source(client, state.data_dir.parent)` заменить на передачу `tmp_path` из фикстуры).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_jobs_kinds.py -q`
Expected: FAIL — kind/source_id не обрабатываются (payload без tracks_override и т.п.)

- [ ] **Step 3: Implement**

3a. `server/service.py`:

```python
def queue_source_run(state: AppState, source_row, prompt: str, request: dict[str, Any]) -> str:
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    prompt = (prompt or "").strip()
    if not prompt and not request.get("targets_override") and not request.get("tracks_override") and not request.get("masks"):
        raise PipelineError("prompt is required (or targets/tracks/masks)")
    job_id = new_job_id()
    suffix = src.suffix.lower() or ".mp4"
    output_path = Path(state.data_dir) / "jobs" / job_id / "output" / f"cleaned{suffix}"
    payload = dict(request or {})
    payload["kind"] = "run"
    payload["allow_download"] = False
    payload["input_path"] = str(src)
    payload["output_path"] = str(output_path)
    payload["prompt"] = prompt
    payload["source_id"] = source_row["id"]
    frames = payload.pop("masks", None)
    if frames:
        masks_dir = Path(state.data_dir) / "sources" / source_row["id"] / "masks"
        paths = [masks_dir / f"{int(n):06d}.png" for n in frames]
        missing = next((p for p in paths if not p.is_file()), None)
        if missing is not None:
            raise PipelineError(f"mask for frame {missing.stem} not found on source")
        payload["masks_override"] = [str(p) for p in paths]
    return state.manage.submit(payload, src, output_path, prompt, job_id=job_id, source_id=source_row["id"])


def queue_source_preview(state: AppState, source_row, prompt: str, request: dict[str, Any]) -> str:
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    prompt = (prompt or "").strip()
    mode = str((request or {}).get("mode") or "parse")
    if mode not in {"parse", "detect"}:
        raise PipelineError("preview mode must be parse | detect")
    if mode == "parse" and not prompt:
        raise PipelineError("prompt is required for mode=parse")
    if mode == "detect" and not request.get("targets"):
        raise PipelineError("targets are required for mode=detect")
    job_id = new_job_id()
    payload = dict(request or {})
    payload["kind"] = "preview"
    payload["input_path"] = str(src)
    payload["prompt"] = prompt
    payload["source_id"] = source_row["id"]
    output_dir = Path(state.data_dir) / "jobs" / job_id / "output"
    payload["output_path"] = str(output_dir / "preview")
    if payload.pop("all", None):
        probe = _as_dict(source_row["probe_json"])
        payload["start"] = 0
        payload["count"] = int(probe.get("frame_count") or 0)
    payload["segmenter"] = "sam2"  # spec §9: sam2-video is not available in preview jobs
    return state.manage.submit(payload, src, output_dir, prompt, job_id=job_id, source_id=source_row["id"])


def queue_source_prompt(state: AppState, source_row, prompt: str, request: dict[str, Any]) -> str:
    src = Path(source_row["path"])
    if not src.is_file():
        raise PipelineError("source file missing")
    masks_dir = Path(state.data_dir) / "sources" / source_row["id"] / "masks"
    annotations = [{"frame": int(p.stem), "mask": str(p)} for p in sorted(masks_dir.glob("*.png"))]
    prompt = (prompt or "").strip()
    if not prompt and not annotations:
        raise PipelineError("provide a text prompt or draw at least one mask on the source")
    job_id = new_job_id()
    payload = dict(request or {})
    payload["kind"] = "prompt"
    payload["input_path"] = str(src)
    payload["prompt"] = prompt
    payload["annotations"] = annotations
    payload["source_id"] = source_row["id"]
    output_dir = Path(state.data_dir) / "jobs" / job_id / "output"
    payload["output_path"] = str(output_dir)
    return state.manage.submit(payload, src, output_dir, prompt, job_id=job_id, source_id=source_row["id"])
```

3b. `server/fastapi_app.py` — переписать `create_job`. Новые Form-параметры: `kind: str = Form("run")`, `source_id: str = Form("")`, `targets: str = Form("")`, `tracks: str = Form("")`, `masks: str = Form("")`, `mode: str = Form("")`, `start/count/stride: str = Form("")`, `indices: str = Form("")`, `all: str = Form("")` (плюс существующие конфиг-поля и новые `llm_base_url`, `llm_api_key`, `keep_workdir`, `min_mask_coverage`, `verify_max_coverage`, `formats` из Task 5). Тело:

```python
        kind = (kind or "run").strip().lower() or "run"
        if kind not in {"run", "preview", "prompt"}:
            raise HTTPException(400, "kind must be run | preview | prompt")
        fields = {...}  # все существующие + новые строковые поля
        payload = serialize_clean_form({k: v for k, v in fields.items() if v != ""})
        for raw, key in ((targets, "targets_override"), (tracks, "tracks_override")):
            raw = (raw or "").strip()
            if raw:
                try:
                    payload[key] = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise HTTPException(400, f"{key} must be JSON: {exc}") from exc
        if (masks or "").strip():
            try:
                payload["masks"] = [int(x) for x in masks.split(",") if x.strip()]
            except ValueError as exc:
                raise HTTPException(400, f"masks must be comma-separated frame numbers: {exc}") from exc
        if mode.strip():
            payload["mode"] = mode.strip()
        for name in ("start", "count", "stride"):
            val = {"start": start, "count": count, "stride": stride}[name]
            if val.strip():
                payload[name] = int(val)
        if indices.strip():
            payload["indices"] = [int(i) for i in indices.split(",") if i.strip()]
        if all_.strip():
            payload["all"] = True
        if sum(1 for k in ("targets_override", "tracks_override", "masks") if payload.get(k)) > 1:
            raise HTTPException(400, "targets, tracks and masks are mutually exclusive")

        source_row = None
        if source_id.strip():
            source_row = st.sources.get(source_id.strip())
            if source_row is None:
                raise HTTPException(404, f"unknown source {source_id.strip()}")
        try:
            if kind == "run":
                if source_row is not None:
                    job_id = queue_source_run(st, source_row, prompt, payload)
                elif video is not None and (video.filename or "").strip():
                    suffix = Path(video.filename).suffix.lower()
                    if suffix not in VIDEO_SUFFIXES:
                        raise HTTPException(400, f"unsupported video type {suffix}")
                    tmp_dir = Path(st.data_dir) / "uploads" / "_incoming"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp = tmp_dir / f"up_{os.getpid()}_{video.filename}"
                    try:
                        await _save_upload(video, tmp)
                        sid = register_source(st, tmp, video.filename)
                    finally:
                        tmp.unlink(missing_ok=True)
                    job_id = queue_source_run(st, st.sources.get(sid), prompt, payload)
                else:
                    raise HTTPException(400, "provide a video file or source_id")
            elif kind == "preview":
                if source_row is None:
                    raise HTTPException(400, "preview from the editor requires source_id")
                job_id = queue_source_preview(st, source_row, prompt, payload)
            else:
                if source_row is None:
                    raise HTTPException(400, "prompt interpretation requires source_id")
                job_id = queue_source_prompt(st, source_row, prompt, payload)
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        row = st.jobs.get(job_id)
        body = job_dict(st, row) if row is not None else {"id": job_id, "state": "QUEUED"}
        body["poll"] = f"/api/jobs/{job_id}"
        body["download"] = f"/api/jobs/{job_id}/output" if kind == "run" else None
        return JSONResponse(body, status_code=201)
```

(HTTPException внутри try/except PipelineError: HTTPException не наследует PipelineError — пробросится корректно.)

Старая функция `queue_clean_job` и route `/api/preview` остаются как есть (обратная совместимость).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_jobs_kinds.py tests/test_api_preview.py tests/test_api_probe.py -q`
Expected: все passed (если старые тесты завязаны на `uploads/{job}/input` — поправить ожидания на source-based input_path; сообщить в ревью)

- [ ] **Step 5: Commit**

```bash
git add server/service.py server/fastapi_app.py tests/test_api_jobs_kinds.py
git commit -m "feat: unified POST /api/jobs with kind, source_id and manual overrides"
```

---

### Task 12: Пресеты

**Files:**
- Modify: `server/service.py` (`list_presets`, `save_preset`, `delete_preset`)
- Modify: `server/fastapi_app.py` (роуты)
- Test: `tests/test_api_presets.py` (новый)

**Interfaces:**
- Produces: `GET /api/presets` → `[{id, name, payload, createdAt}]`; `POST /api/presets` JSON `{name, payload}` → 201 item (payload — dict полей serialize_clean_form); `DELETE /api/presets/{id}` → `{ok}` | 404. Хранение `data_dir/presets.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_presets.py
"""Presets CRUD backed by data_dir/presets.json."""
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import fastapi_app as fa
from server.service import auth_from_env


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VIDEOCLEAN_UI_USER", "admin")
    monkeypatch.setenv("VIDEOCLEAN_UI_PASSWORD", "pw")
    from server.app_state import build_app_state

    state = build_app_state(tmp_path, worker=False, downloader=False)
    app = fa.create_app(state)
    client = TestClient(app)
    client.headers.update({"Authorization": "Basic " + base64.b64encode(b"admin:pw").decode()})
    yield client, state, tmp_path


def test_preset_crud(client):
    client, state, tmp_path = client
    created = client.post("/api/presets", json={"name": "Логотип", "payload": {"inpainter": "lama", "mask_dilate_px": 5}})
    assert created.status_code == 201, created.text
    item = created.json()
    assert item["name"] == "Логотип" and item["payload"]["inpainter"] == "lama"
    lst = client.get("/api/presets").json()
    assert [p["id"] for p in lst] == [item["id"]]
    assert (tmp_path / "presets.json").is_file()
    assert client.delete(f"/api/presets/{item['id']}").status_code == 200
    assert client.get("/api/presets").json() == []
    assert client.delete(f"/api/presets/{item['id']}").status_code == 404


def test_preset_validation(client):
    client, state, tmp_path = client
    assert client.post("/api/presets", json={"name": " ", "payload": {}}).status_code == 400
    assert client.post("/api/presets", json={"name": "x", "payload": [1, 2]}).status_code == 400
    assert (tmp_path / "presets.json").exists() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_presets.py -q`
Expected: FAIL — 404/405 (роутов нет)

- [ ] **Step 3: Implement**

3a. `server/service.py`:

```python
def _presets_file(data_dir: Path) -> Path:
    return Path(data_dir) / "presets.json"


def list_presets(data_dir: Path) -> list[dict[str, Any]]:
    f = _presets_file(data_dir)
    if not f.is_file():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_preset(data_dir: Path, name: str, payload: Any) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise PipelineError("preset name is required")
    if not isinstance(payload, dict):
        raise PipelineError("preset payload must be an object of pipeline fields")
    presets = list_presets(data_dir)
    item = {
        "id": f"p_{uuid.uuid4().hex[:8]}",
        "name": name,
        "payload": payload,
        "createdAt": utc_now().isoformat(),
    }
    presets.append(item)
    _presets_file(data_dir).write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def delete_preset(data_dir: Path, preset_id: str) -> bool:
    presets = list_presets(data_dir)
    rest = [p for p in presets if p.get("id") != preset_id]
    if len(rest) == len(presets):
        return False
    _presets_file(data_dir).write_text(json.dumps(rest, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
```

(импорты: `uuid`, `utc_now` добавить в service.py.)

3b. Роуты:

```python
    @app.get("/api/presets")
    def presets(st: AppState = Depends(get_state)):
        return list_presets(st.data_dir)

    @app.post("/api/presets")
    def create_preset(body: dict[str, Any], st: AppState = Depends(get_state)):
        data = body or {}
        try:
            item = save_preset(st.data_dir, str(data.get("name") or ""), data.get("payload"))
        except PipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(item, status_code=201)

    @app.delete("/api/presets/{preset_id}")
    def remove_preset(preset_id: str, st: AppState = Depends(get_state)):
        if not delete_preset(st.data_dir, preset_id):
            raise HTTPException(404, f"unknown preset {preset_id}")
        return {"ok": True, "id": preset_id}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_api_presets.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add server/service.py server/fastapi_app.py tests/test_api_presets.py
git commit -m "feat: server-side pipeline presets CRUD"
```

---

### Task 13: job_dict source-поля + документация + полный прогон

**Files:**
- Modify: `server/service.py` (`job_dict`)
- Modify: `server/fastapi_app.py` (`api_index` — описания новых роутов)
- Modify: `docs/PARAMS.md`
- Test: `tests/test_api_jobs_kinds.py` (дополнить)

**Interfaces:**
- Produces: job_dict добавляет `source_id`, `source_name`; `kind` уже динамический (`run|preview|prompt`).

- [ ] **Step 1: Write the failing test (дополнить tests/test_api_jobs_kinds.py)**

```python
def test_job_dict_carries_source_fields(client):
    client, state, tmp_path = client
    src = _source(client, tmp_path)
    resp = client.post("/api/jobs", data={"kind": "run", "source_id": src["id"], "prompt": "p"})
    assert resp.status_code == 201
    job = resp.json()
    assert job["source_id"] == src["id"]
    assert job["source_name"] == "clip.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_jobs_kinds.py::test_job_dict_carries_source_fields -q`
Expected: FAIL — ключей нет (KeyError/AssertionError)

- [ ] **Step 3: Implement**

3a. `server/service.py`, `job_dict` — после `"kind": ...`:

```python
        "source_id": (row["source_id"] if "source_id" in row.keys() else None) or request.get("source_id"),
```

и перед `return`:

```python
    sid = out["source_id"]  # если структура позволяет; иначе вычислить до return
    source_name = None
    if sid:
        srow = state.sources.get(sid) if state.sources is not None else None
        if srow is not None:
            source_name = srow["name"]
    out["source_name"] = source_name
    return out
```
(аккуратно встроить в существующую функцию: собрать `out = {...}`, затем source_name, затем `return out`.)

3b. `api_index` — дополнить секции:

```python
            "sources": {
                "POST /api/sources": "multipart video upload",
                "GET /api/sources": "list",
                "GET /api/sources/{id}": "detail",
                "DELETE /api/sources/{id}": "delete (409 if active jobs)",
                "GET /api/sources/{id}/video": "stream source video",
                "GET /api/sources/{id}/frames/{n}": "extracted frame jpeg",
                "PUT/GET/DELETE /api/sources/{id}/masks/{n}": "per-frame annotation mask",
                "GET /api/sources/{id}/annotations": "list annotated frames",
            },
            "jobs": {
                "POST /api/jobs": "multipart: kind=run|preview|prompt, source_id or video, prompt, pipeline fields, targets/tracks/masks overrides",
                ...existing...
            },
            "presets": {
                "GET/POST /api/presets": "pipeline presets",
                "DELETE /api/presets/{id}": "",
            },
```

3c. `docs/PARAMS.md` — добавить раздел «WebUI/API-поля» с таблицей новых полей: kind, source_id, targets, tracks, masks, llm_base_url, llm_api_key, keep_workdir, min_mask_coverage, verify_max_coverage, formats. Кратко, по образцу существующих разделов.

- [ ] **Step 4: Полный прогон**

Run: `uv run pytest -q`
Expected: все passed; `uv run python -c "import server.fastapi_app"` без ошибок.

- [ ] **Step 5: Commit**

```bash
git add server/service.py server/fastapi_app.py docs/PARAMS.md tests/test_api_jobs_kinds.py
git commit -m "feat: job source fields, API index and PARAMS docs for editor backend"
```

---

## Самопроверка плана (выполнена)

1. **Покрытие спеки §4–6:** BuildPrompt (Task 7-8), tracks/masks/targets overrides (4-5), SourceIndex (1), sources API (9-10), frames (9), masks+annotations (10), presets (12), единый POST /api/jobs (11), job_dict source (13), preview count-развёртка + форс sam2 (11). Не покрывает §7 (фронтенд) — это План B.
2. **Плейсхолдеры:** Task 4 Step 3 содержит «существующий код … без изменений» — осознанное указание зоны рефактора, не плейсхолдер: исполнитель двигает существующий блок в else-ветку.
3. **Типы:** `masks_override` — `list[str]` в request, `list[Path]` после cleanup_request_from_row; `tracks_override` — `list[dict]`; согласовано между Task 4/5/11.
