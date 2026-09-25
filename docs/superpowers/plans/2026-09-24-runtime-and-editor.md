# Полная загрузка машины и обычный редактор — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** После семи задач собирается один образ, который на тёплом 24 ГБ прогоне balanced минутного 1080p укладывается в 120 с и не оставляет карту на простое в единицы гигабайт, а редактор ведёт один прогон до MP4.

**Architecture:** Один `PipelineConfig`. Сборка — `merge_run_config`: дефолты, затем пресет, затем ключи рецепта, затем только те поля, которые форма или CLI реально прислали. Движок дырки считает кроп под потолок VRAM. Пресет — плоский набор тех же полей. Редактор шлёт `profile` или `preset` и разошедшиеся ручки, не весь формуляр.

**Tech Stack:** Python 3.11, FastAPI, pytest, React + Vite + bun, ffmpeg, torch, sam2 pin `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

**Spec:** `docs/superpowers/specs/2026-09-24-runtime-and-editor.md`

## Global Constraints

- Пустой потолок = вся машина. `max_vram_mb`, `cpu_threads`, `inpaint_max_side` — `int | None`. `0` у этих трёх — ошибка. `inpaint_workers=0` остаётся «авто».
- Резерв VRAM ровно `1536 * 1024 * 1024` байт, вычитается один раз внутри `vram_budget_bytes`.
- `vramPeakInpaintMb` — дельта активаций заливки: `max_memory_allocated()` после инпейнта минус `memory_allocated()` сразу после `reset_peak_memory_stats`. Чекер сравнивает эту дельту с `vramBudgetMb`, полоса 60–100%.
- Кандидат батча: `min(last * 2, range_len)`. Не влез или CUDA OOM — один раз середина `(last + nxt) // 2`, не половина степени двойки. `limitedBy=frames` только если финальный батч равен длине диапазона.
- RAFT остаётся fp32, `propainter_raft_iter` остаётся 20. `.half()` только у `flow_complete` и `InpaintGenerator`, один раз при загрузке.
- LaMa fp16 — только если проба CUDA 64×64 батчем 2 прошла. Исключение или нефинитный выход запирает процесс на fp32. CPU и MPS half не пробуют.
- sam2 pin `2b90b9f5ceec907a1c18123530e92e794ad901a4`. State — копия `init_state` этого коммита. `reset_state` для сборки не вызывается.
- Нет группы «Скорость». `fast` / `balanced` / `quality` — рецепты картинки. `balanced` не переводится на `sam2-video`.
- `device=auto` включается в том же коммите, что и кроп. До этого отсутствующее устройство остаётся `cpu` и не записывается в explicit как строка `cpu`.
- Образ собирается после задачи 7. `VIDEOCLEAN_DATA_DIR=/root/.videoclean`, `HF_HOME=/workspace/.cache/huggingface`.
- UI-тексты русские. `shared/ui/*` не менять. SSE остаётся. Тесты — контракт, не браузер.
- Неизвестный пресет: HTTP 404, `detail` ровно `unknown preset`. Не 400.

Порядок задач: 1 → 2 → 3 → 5 → 4 → 6 → 7. Задача 4 не мержится в сегодняшний `main` без 2, 3 и 5. Задача 6 не мержится без 5. Номера задач совпадают с номерами PR в спеке, поэтому 5 идёт раньше 4.

---

### Task 1: Отчёт стадий и снимок машины

**Files:**
- Modify: `videoclean/adapters/progress/job_store.py`
- Modify: `videoclean/application/use_cases/run_cleanup.py`
- Modify: `videoclean/composition.py`
- Modify: `videoclean/cli.py`
- Test: `tests/test_job_timings.py`

**Interfaces:**
- Consumes: `STAGES` в `videoclean/progress.py`
- Produces: `report["timings"]` с ключами `load`, `decode`, `parse`, `detect`, `track`, `segment`, `inpaint`, `verify`, `encode`, `package`. Честные секунды у `decode`, `parse`, `inpaint`, `verify`, `encode`, `package`. `detect`, `track`, `segment` равны `null`. `report["budget"]` только для чтения: устройство как в конфиге, VRAM total/free если CUDA есть, `cpuCount`, `nvenc`. Верхний `inpaintWorkers` не трогать.

- [ ] **Step 1: Write the failing test**

```python
def test_finished_report_has_honest_timings(tmp_path, monkeypatch):
    report = _run_cleanup_report(tmp_path, monkeypatch)
    timings = report["timings"]
    assert isinstance(timings["decode"], (int, float))
    assert isinstance(timings["encode"], (int, float))
    assert timings["segment"] is None
    assert timings["track"] is None
    assert timings["detect"] is None
    assert report["budget"]["cpuCount"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_job_timings.py::test_finished_report_has_honest_timings -q`
Expected: FAIL, ключа `timings` нет.

- [ ] **Step 3: Write minimal implementation**

`ProgressBridge` запоминает секунды между `start` и `finish` и кладёт `stageTitle` из `STAGES`. В SSE уходят `stage`, `stageTitle`, `fraction`, `detail`. Алгоритмы, дефолт устройства и профили не меняются. Ноль вместо `null` у `segment` — провал теста.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_job_timings.py tests/test_profiles_verify_inpaint.py -q`
Expected: PASS. Кадр результата не меняется.

- [ ] **Step 5: Commit**

```bash
git add videoclean/adapters/progress/job_store.py videoclean/application/use_cases/run_cleanup.py videoclean/composition.py videoclean/cli.py tests/test_job_timings.py
git commit -m "feat: record stage timings and a read-only machine snapshot"
```

### Task 2: Потолок ресурсов и кэш весов

**Files:**
- Create: `videoclean/application/budget.py`
- Create: `videoclean/adapters/models/weights_cache.py`
- Modify: `videoclean/application/config.py`
- Modify: `videoclean/adapters/inpainters/propainter.py`
- Modify: `pyproject.toml`
- Modify: `server/fastapi_app.py`
- Test: `tests/test_budget.py`
- Test: `tests/test_weights_cache.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `torch.cuda.mem_get_info`
- Produces: `Budget` и `resolve_budget(cfg, *, probe) -> Budget` как в спеке, раздел «Устройство и потолок». `get_or_load(key, loader)`.

- [ ] **Step 1: Write the failing test**

```python
def test_empty_ceiling_is_free_minus_reserve():
    free = 20 * 1024**3
    budget = resolve_budget(cfg_with(max_vram_mb=None), probe=probe(free=free, resident=True))
    assert budget.vram_budget_bytes == free - 1536 * 1024**2

def test_second_get_or_load_does_not_call_loader():
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return object()
    get_or_load(("lama", "big-lama", "cpu", "fp32"), loader)
    get_or_load(("lama", "big-lama", "cpu", "fp32"), loader)
    assert calls["n"] == 1
```

Плюс тест: исходник `_run` в `propainter.py` не содержит `.half()`, а загрузка кастует только `flow_complete` и `InpaintGenerator`. Явный `device=cpu` остаётся cpu. Прогон без `device` остаётся cpu. `resolve_workers` на GPU при `inpaint_workers=0` остаётся 1.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_budget.py tests/test_weights_cache.py -q`
Expected: FAIL, модулей нет.

- [ ] **Step 3: Write minimal implementation**

`psutil` в dependencies. MPS: пустой потолок = 60% `virtual_memory().total`. Ниже 512 МБ после резерва — `PipelineError`. Потоки CPU ставятся на job и восстанавливаются в `finally`. LaMa в этом таске не кастуется. `inpaint_max_side` в отчёт попадает, кадр им ещё не режется. Строка `auto` CUDA-ветку `quality` не выбирает.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_budget.py tests/test_weights_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/budget.py videoclean/adapters/models/weights_cache.py videoclean/application/config.py videoclean/adapters/inpainters/propainter.py pyproject.toml server/fastapi_app.py tests/test_budget.py tests/test_weights_cache.py
git commit -m "feat: add resource ceilings and a process weight cache"
```

### Task 3: Raw-кадры и pipe-энкод

**Files:**
- Create: `videoclean/adapters/media/raw_store.py`
- Modify: `videoclean/adapters/media/ffmpeg.py`
- Modify: `videoclean/application/use_cases/run_cleanup.py`
- Test: `tests/test_raw_store.py`
- Test: `tests/test_encode_choice.py`

**Interfaces:**
- Produces: `FrameStore` с записью и чтением кадра `bgr24`. Энкод мезонина читает raw stdin, не `frame_%06d.jpg`.

- [ ] **Step 1: Write the failing test**

```python
def test_nvenc_argv_has_no_crf():
    argv = encoder_argv(nvenc=True)
    assert argv[:6] == ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr"]
    assert "-cq" in argv and "18" in argv
    assert "-crf" not in argv

def test_missing_nvenc_uses_veryfast():
    argv = encoder_argv(nvenc=False)
    assert "libx264" in argv and "veryfast" in argv and "-crf" in argv
```

Плюс: `FrameStore` roundtrip одного кадра. Прогон с `keep_workdir=false` не оставляет `frame_*.jpg`. Data dir на сетевой ФС — отказ до mmap.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_encode_choice.py tests/test_raw_store.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Пробник NVENC: `ffmpeg -encoders` ищет `h264_nvenc`, затем пробный 16×16 в `-f null` теми же аргументами. Кэш пробника на процесс. Второй полный `list[ndarray]` заливки не держится. Превью-JPEG не трогать. Заливка в этом таске всё ещё полный кадр.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_encode_choice.py tests/test_raw_store.py -q`
Expected: PASS. MP4 сохраняет число кадров и аудио.

- [ ] **Step 5: Commit**

```bash
git add videoclean/adapters/media/raw_store.py videoclean/adapters/media/ffmpeg.py videoclean/application/use_cases/run_cleanup.py tests/test_raw_store.py tests/test_encode_choice.py
git commit -m "feat: decode and encode the hot path without JPEG"
```

### Task 5: Плоский пресет и merge

Делается до движка. Дефолт устройства остаётся `cpu`.

**Files:**
- Modify: `videoclean/application/profiles.py`
- Modify: `videoclean/composition.py`
- Modify: `videoclean/cli.py`
- Modify: `server/service.py`
- Modify: `server/fastapi_app.py`
- Modify: `server/api_schema.py`
- Modify: `webui/src/widgets/stage-rail/params.ts`
- Modify: `webui/src/entities/job/api.ts`
- Test: `tests/test_preset_merge.py`
- Test: `tests/test_api_presets.py`

**Interfaces:**
- Produces: `explicit_from_form` и `merge_run_config` с сигнатурами из раздела «Слияние» спеки.

- [ ] **Step 1: Write the failing test**

```python
def test_merge_order_dilate():
    defaults = {"mask_dilate_px": 3, "profile": "custom", "device": "cpu"}
    assert merge(defaults, {"mask_dilate_px": 8}, None, {})["mask_dilate_px"] == 8
    assert merge(defaults, {"mask_dilate_px": 8}, "quality", {})["mask_dilate_px"] == 5
    assert merge(defaults, {"mask_dilate_px": 8}, "quality", {"mask_dilate_px": 9})["mask_dilate_px"] == 9

def test_preset_profile_quality_does_not_keep_its_own_dilate():
    out = merge(defaults, {"profile": "quality", "mask_dilate_px": 8}, None, {})
    assert out["mask_dilate_px"] == 5

def test_unknown_preset_is_404():
    res = client.post("/api/jobs", data={"preset": "p_missing", "prompt": "logo", "source_id": sid})
    assert res.status_code == 404
    assert res.json()["detail"] == "unknown preset"
```

Плюс тесты из приёмки PR 5 спеки: отсутствующий `device` не становится explicit `"cpu"`; `explicitFields(DEFAULT_PARAMS)` не содержит `device`, `segmenter`, `inpainter`, `mask_dilate_px`; старый payload с `detect.stride` после чтения без `stride`; `PUT` сохраняет id; дубликат имени без `replace` — 409; непереданный CLI `--detector-max-box-area` даёт `0.45`; неизвестный профиль — 400; `custom` не вызывает `profile_defaults`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preset_merge.py tests/test_api_presets.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`serialize_clean_form` на job-пути дыры не заполняет. `config_from_flags` собирает explicit только из `ParameterSource.COMMANDLINE`. `apply_profile` больше не вызывается из `pipeline_config_from_dict`. Переписать `test_apply_profile_overwrites_owned_keys`: явный `inpainter` побеждает `fast`. `verify_redetect` в allow-list пресета в этом таске не добавлять. `GET /api/presets` отдаёт плоский payload, и `applyPreset` читает его в том же коммите. `PUBLIC_OPS` включает CRUD пресетов.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preset_merge.py tests/test_api_presets.py tests/test_profiles_verify_inpaint.py -q`
Expected: PASS.
Run: `cd webui && bun run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/profiles.py videoclean/composition.py videoclean/cli.py server/service.py server/fastapi_app.py server/api_schema.py webui/src/widgets/stage-rail/params.ts webui/src/entities/job/api.ts tests/test_preset_merge.py tests/test_api_presets.py tests/test_profiles_verify_inpaint.py
git commit -m "feat: merge presets after explicit fields and before profile keys"
```

### Task 4: Движок дырки

**Files:**
- Create: `videoclean/application/hole_policy.py`
- Modify: `videoclean/application/ports/inpainter.py`
- Modify: `videoclean/application/inpaint_runtime.py`
- Modify: `videoclean/adapters/inpainters/lama.py`
- Modify: `videoclean/adapters/inpainters/propainter.py`
- Modify: `videoclean/adapters/segmenters/sam2.py`
- Modify: `videoclean/adapters/segmenters/sam2_video.py`
- Modify: `videoclean/adapters/detectors/grounding_dino.py`
- Modify: `videoclean/application/profiles.py`
- Modify: `videoclean/application/config.py`
- Modify: `videoclean/application/use_cases/run_cleanup.py`
- Modify: `Dockerfile`
- Modify: `Dockerfile.api`
- Test: `tests/test_hole_policy.py`
- Create: `tests/fixtures/sla/generate.py`
- Create: `tests/fixtures/sla/sla_1080p30_60s.sha256` (после первого запуска генератора)

**Interfaces:**
- Consumes: `Budget`, `merge_run_config`, `FrameStore`
- Produces: `plan_holes(...) -> list[RangePlan]`. `inpaint_masked(frames, masks, plan)` без дефолтной реализации. `verify_redetect: bool = False` на `PipelineConfig`.

- [ ] **Step 1: Write the failing test**

```python
def test_small_mask_plans_a_crop_under_the_ceiling():
    plans = plan_holes(frame_hw=(1080, 1920), mask=box_mask(80, 40), budget_side=256, budget_batch=2)
    assert len(plans) >= 1
    assert plans[0].side <= 256 and plans[0].side % 8 == 0
    assert plans[0].batch <= 2
    assert plans[0].policy == "lama-crop"

def test_lama_family_stays_lama_on_a_large_hole():
    plans = plan_holes(frame_hw=(1080, 1920), mask=box_mask(900, 900), family="lama", budget_side=512, budget_batch=1)
    assert plans[0].policy == "lama-crop"
    assert plans[0].side <= 512

def test_half_probe_failure_locks_fp32():
    assert probe_lama_dtype(raise_on_half=True) == "fp32"
```

Плюс: покрытие считается по `count_nonzero(mask) / mask.size`, не по площади бокса. `verify_redetect=false` не зовёт детектор. `masks_override` пропускает verify. Грязный диапазон идёт в `inpaint_masked`. Нет метода — `AdapterUnavailable` до кадров. State sam2 содержит ключи `init_state` коммита `2b90b9f5`, `images[0]` терпит `.to().float().unsqueeze(0)`, `reset_state` не вызывается. `{profile: "quality"}` на CUDA даёт `sam2-video` + `propainter` и вызов кропа, не полного кадра. Рецепт `quality` на CUDA имеет `inpaint_workers=0`. `balanced` остаётся `segmenter=sam2`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hole_policy.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

В том же коммите: серверный дефолт устройства становится `auto`, CLI `--device` становится `auto`, кроп включается. `resolve_budget` после резидентных весов. Пороги дырки — константы из Key Decision 8 спеки. Батч от 1 по аллокатору, кандидат `min(2×, range_len)`, середина при отказе. `re_inpaint_ranges` использует те же кропы. sam2 pin в обоих Dockerfile. Верхний `inpaintWorkers` больше не пишется. `timings.detect`, `track`, `segment` заполняются раздельно. Docstring `profiles.py` больше не говорит speed. Запустить генератор один раз и закоммитить только sha256, не mp4.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_hole_policy.py tests/test_preset_merge.py tests/test_budget.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add videoclean/application/hole_policy.py videoclean/application/ports/inpainter.py videoclean/application/inpaint_runtime.py videoclean/adapters/inpainters/lama.py videoclean/adapters/inpainters/propainter.py videoclean/adapters/segmenters/sam2.py videoclean/adapters/segmenters/sam2_video.py videoclean/adapters/detectors/grounding_dino.py videoclean/application/profiles.py videoclean/application/config.py videoclean/application/use_cases/run_cleanup.py Dockerfile Dockerfile.api tests/test_hole_policy.py tests/fixtures/sla/generate.py tests/fixtures/sla/sla_1080p30_60s.sha256
git commit -m "feat: inpaint holes inside the VRAM budget"
```

### Task 6: Редактор

**Files:**
- Modify: `webui/src/widgets/stage-rail/stage-rail.tsx`
- Modify: `webui/src/widgets/stage-rail/stage-parts.tsx`
- Modify: `webui/src/widgets/stage-rail/params.ts`
- Modify: `webui/src/widgets/stage-rail/param-meta.ts`
- Modify: `webui/src/pages/workspace/index.tsx`
- Modify: `webui/src/widgets/library/index.tsx`
- Modify: `webui/src/widgets/library/upload.tsx`
- Modify: `webui/src/widgets/editor-viewer/index.tsx`
- Modify: `webui/src/widgets/timeline/index.tsx`
- Modify: `webui/src/entities/preset/api.ts`
- Modify: `webui/src/features/inpaint-run/index.ts`
- Modify: `webui/src/features/interpret/index.ts`

**Interfaces:**
- Consumes: `explicitFields`, `PUT /api/presets/{id}`, `stageTitle` из SSE
- Produces: обычный режим и эксперт, как раздел «Обычный редактор и эксперт» спеки. `shared/ui/*` не менять.

- [ ] **Step 1: Write the failing test**

Контракт клиента уже покрыт `explicitFields` в задаче 5. Здесь добавить проверку предиката, если он вынесен в чистую функцию `webui/src/widgets/stage-rail/run-choice.ts`:

```ts
expect(chooseRun({ strokes: true, tracksFull: true, manualTargets: true })).toEqual({
  path: "masks",
  maskPolicy: "propagate",
})
expect(chooseRun({ strokes: false, tracksFull: true, useTracks: true })).toEqual({ path: "tracks" })
expect(chooseRun({ strokes: false, tracksFull: false, manualTargets: true })).toEqual({ path: "targets" })
expect(chooseRun({})).toEqual({ path: "prompt" })
```

Ручной запрос: `source === "manual"` или `query` отличается от последнего interpret. Голая фраза не шлёт `stride`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd webui && bun test src/widgets/stage-rail/run-choice.test.ts`
Expected: FAIL, модуля нет. Если в проекте нет bun test, положить тот же модуль и прогнать его через `bunx vitest run` одной командой, не поднимая браузер.

- [ ] **Step 3: Write minimal implementation**

Удалить «Запустить всё», `PresetsPopover` и запись `interpret.result.prompt` в поле фразы. Эксперт: `localStorage`, дефолт выключен. `keep_workdir` в дефолте редактора выключен. Результат начинается с «Скачать MP4», остальные форматы свёрнуты. Во время прогона правая колонка заблокирована, центр живой. Статус: `stageTitle`, detail, проценты, ETA, «Стоп». Загрузка выбирает новый источник. Смена источника и кроп спрашивают при непустой фразе или несохранённых рамках. Удаление job спрашивает. Проваленный job показывает `error`. Кадры на экране с 1. Пустой кадр разметки без «Маска сохранена». Станции эксперта: Детектор, Трекинг, Сегментация, Заливка, Проверка, Разбор фразы, Устройство. Группы «Скорость» нет.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd webui && bun run lint && bun run typecheck`
Expected: PASS. Ручной проход — чеклист приёмки PR 6 в спеке, его делает владелец на собранном фронте кандидата, не эта задача отдельно.

- [ ] **Step 5: Commit**

```bash
git add webui/src/widgets/stage-rail webui/src/pages/workspace/index.tsx webui/src/widgets/library webui/src/widgets/editor-viewer/index.tsx webui/src/widgets/timeline/index.tsx webui/src/entities/preset/api.ts webui/src/features/inpaint-run/index.ts webui/src/features/interpret/index.ts
git commit -m "feat: replace the pipeline rail with a scenario and an expert toggle"
```

### Task 7: Документы и чекер SLA

**Files:**
- Modify: `docs/PARAMS.md`
- Create: `scripts/check_sla_report.py`
- Test: `tests/test_sla_checker.py`

**Interfaces:**
- Consumes: поля `budget`, `hole`, `timings`, `finishedAt`, `startedAt` из отчёта задачи 4
- Produces: код выхода 0 только когда все условия раздела SLA спеки выполнены. Нет файла хеша — код 2. Хеш клипа не совпал — код 3.

- [ ] **Step 1: Write the failing test**

```python
def test_checker_rejects_weight_peak_as_if_it_were_activations(tmp_path):
    report = sla_report(vram_budget_mb=20480, vram_peak_inpaint_mb=4000, limited_by="ceiling")
    assert check(report, max_wall_s=120) != 0

def test_checker_accepts_activation_delta_inside_the_band(tmp_path):
    report = sla_report(
        vram_budget_mb=20480,
        vram_peak_inpaint_mb=14000,
        limited_by="ceiling",
        warm=True,
        device="cuda",
        policy="lama-crop",
        wall_s=90,
    )
    assert check(report, max_wall_s=120) == 0
```

Плюс: `limitedBy=frames` проходит только при `batch == frameCount`. `budget.inpaintMaxSide is not None` или `hole.limitedBy == "inpaint_max_side"` — провал на воротах 24 ГБ. Расхождение стены и суммы `timings` больше 5 с — провал.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sla_checker.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`PARAMS.md`: порядок merge, `device=auto`, потолки, `inpaint_max_side`, `verify_redetect`, политика дырки. Удалить фразы «никаких профилей» и «дефолт device cpu». data dir `/root/.videoclean`, HF на `/workspace/.cache/huggingface`. Абзац, где data dir лежит на `/workspace/.videoclean`, удалить. В тексте нет группы «Скорость» и нет обещания 120 с для 4K, CPU и дырки на весь кадр.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sla_checker.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/PARAMS.md scripts/check_sla_report.py tests/test_sla_checker.py
git commit -m "docs: pin paths and the 120s checker"
```

## Self-review

Покрытие спеки: наблюдаемость — задача 1; потолки и кэш — 2; JPEG и энкод — 3; merge и пресет — 5; дырка, sam2, verify, `device=auto` — 4; редактор — 6; пути пода и чекер — 7. Задача 7 закрывает пути и чекер поверх уже влитых 1–6. Промежуточные задачи на под не ставятся.

Чекер не принимает пик, в который входят веса. Рецепт `quality` на CUDA не остаётся полным кадром. `balanced` не становится `sam2-video`.
