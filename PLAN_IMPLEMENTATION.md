# План реализации (детальный). Overlay Eraser: статичный пайплайн + адаптеры

> Сопроводительные документы: `PLAN_OVERLAY_ERASER.md` — архитектура и продукт; `REPORT_1`/`REPORT_2` — исследования. Этот файл — пошаговая реализация **по фактам текущей кодовой базы** (каждый пункт привязан к файлу и строке; ничего не «по памяти»).
>
> Правила версии 2 (решение владельца продукта):
> 1. **API ломать можно.** Алиасы, двойные имена и миграции запрещены — пользователей нет, код должен быть чистым.
> 2. **Ollama выпиливается из приложения полностью.** Единственный LLM-контракт — openai-compatible провайдеры, добавляемые через форму (`POST /api/providers`). Ollama — просто такой провайдер (`base_url=http://host:11434/v1`), который пользователь добавляет руками.
> 3. **Ничего не выбрасываем из реализаций** — существующие механизмы становятся адаптерами за портами статичного пайплайна (см. `PLAN_OVERLAY_ERASER.md` §2–3).
> 4. Всё, что не удалось проверить по коду/логу, помечено «⚠️ требует проверки» или вынесено в §11 «Помеченные решения».

---

## 1. Фактическая база (что проверено и на что опираемся)

### 1.1 Порты уже есть — `videoclean/application/ports/` (10 Protocol-классов)

| Порт | Файл | Контракт (факт) |
|---|---|---|
| `Detector` | `ports/detector.py:14` | `status()`, `discover(images, queries, on_progress)` |
| `Segmenter` | `ports/segmenter.py:14` | `masks(frames, tracks, on_progress)` |
| `Inpainter` | `ports/inpainter.py:8` | `inpaint`, `inpaint_clip`, `inpaint_masked` |
| `PromptParser` | `ports/prompt.py:10` | `parse(prompt, frames, frame_indices, parse_chunk_frames)` |
| `MediaGateway` | `ports/media.py:9` | `probe/extract_frames/extract_frames_subset/encode_mezzanine/package` |
| `LlmClient` | `ports/llm.py:6` | `status()`, `complete(system, user, image_jpeg, images)` |
| `ModelCatalog` | `ports/model_catalog.py:25` | `list_status()`, `is_ready(component_id)` |
| `ProgressPort` | `ports/progress.py:6` | `start/tick/finish` (+`stage_seconds`) |
| `JobStore` / `JobControl` | `ports/jobs.py:6`, `ports/job_control.py:7` | upsert / submit·cancel·retry·delete·mark_failed·recover_orphans |

### 1.2 Оркестратор — `application/use_cases/run_cleanup.py` (текущий порядок стадий)

`execute`:69 → `_run`:140:
1. `_ensure_ready`:893 — хард-фейл, если `parser.status()` не «ready» (строки 904–907) ← следствие зависимости от LLM;
2. validate + `input_manifest.json`:141–161;
3. decode: `_open_frames`:530 → `FrameStore` (`adapters/media/raw_store.py`) при наличии `decode_bgr`, иначе `LazyFrames` из JPEG;
4. `_prepare_runtime`:558 — эйгр-`ensure()` детекторов и сегментера (566–575; **inpainter убран из эйгр-цикла** коммитом `5f90cb8`), бюджет/теневые потоки (578–579), проверка места под `sam2.f32` при `segmenter == "sam2-video"` (589–595), выбор энкодера nvenc/libx264 (596–599);
5. три входа: `masks_override` (176–207, policy `static|propagate`), `tracks_override` (208–231), промпт-путь (232–317): `parser.parse`:248 → `_discover`:858 (цепочка детекторов) → `select_tracks` (`application/select.py`) → `_segment_masks`:845;
6. покрытие масок `min_mask_coverage`:326–332;
7. ERASE: `_paint_holes`:629 — `_ready_inpainter`:601 (ленивая загрузка весов), `plan_holes` (`application/hole_policy.py:123`), окна по 16 кадров / `propainter-crop`-диапазоны, `inpaint_masked`:660–674;
8. VERIFY: 349–415 — `residual_unchanged_mask` **на каждый кадр по всему кадру** (`application/verify_quality.py:9–40`, absdiff строки 26–27), опц. `verify_redetect` (363–375) = повторный `_discover`+`_segment_masks` на cleaned, `_repaint_ranges`:704;
9. encode (417–446, `encode_from_store` c флагом nvenc:428) → package (448–477) → report (479–510) → cleanup (512–527, удаляет `frames.bgr`, `cleaned.bgr`, `sam2.f32`).

Строковая связка «какой инпейнтёр → семейство политики»: `run_cleanup.py:643` и `:708` (`family = "propainter" if cfg.inpainter == "propainter" else "lama"`) — убрать в пользу метаданных реестра.

### 1.3 Конфиг — `application/config.py`

`PipelineConfig`:137–205: `device`, `detectors`, `detector_*`, `segmenter`, `segmenter_model`, `inpainter`, `inpainter_model`, **`llm_place` (149)**, `llm_model`, `llm_base_url`, `llm_api_key`, `formats`, `webm_crf`, `segment_seconds`, `allow_download`, `verify`, `mask_dilate_px`, `min_mask_coverage`, `verify_max_coverage`, **`profile` (163)**, `device_requested`, `max_vram_mb`, `cpu_threads`, `inpaint_max_side`, `verify_redetect`, **`preset_id` (172)**, `verify_max_passes` (174), `inpaint_workers`, `inpaint_chunk_overlap`, `prompt_frame_stride/max`, `parse_chunk_frames`, `vision_batch`, `detector_keyframes/nms_iou/max_box_area`, `select_relax`, `tracker_*`, `propainter_*` (**`propainter_raft_iter` = 20**, строка 203), `prompt_templates`. Константы: `LLM_PLACES = ("auto","local","cloud")` (15), `INPAINTERS = ("lama","propainter")` (14), `SEGMENTERS = ("sam2","sam2-video")` (13).
`RunCleanupRequest`:286–299: `targets_override` / `tracks_override` / `masks_override` / `mask_policy` — ручные входы уже первоклассные.

### 1.4 Профили — `application/profiles.py` (выпиливается целиком)

`PROFILES = ("fast","balanced","quality")`:15, `profile_defaults`:18–61, `PROFILE_KEYS` (9 ключей):65–77, `apply_profile`:80–96, `profiles_payload`:99–123 (метаданные UI), `explicit_from_form`:126–136, `merge_run_config`:142–202 (defaults → preset → PROFILE_KEYS → explicit).

### 1.5 LLM-слой (выпиливается/упрощается)

- `adapters/llm/resolve.py` — `resolve_llm(cfg)`:69–123: логика `llm_place` auto|local|cloud, дефолты **`DEFAULT_LOCAL_URL = "http://127.0.0.1:11434/v1"`** (10, это Ollama!), GGUF → `LlamaCppLlm`, `_provider_credentials`:50–66 (подстановка из `catalog.load_providers` — **оставляем, это и есть «форма провайдеров»**).
- `adapters/llm/openai_compat.py` — `OpenAiCompatLlm`:11–105 (urllib, `/v1/chat/completions`, vision) — **остаётся единственным клиентом**.
- `adapters/llm/llama_cpp.py` — `LlamaCppLlm` (текстовый GGUF-раннер) — **удаляется** (§11, D1).
- `adapters/llm/ollama_setup.py` (183 строки) — установка/запуск Ollama — **удаляется целиком**.

### 1.6 Ollama-следы (полный список для выпиливания)

`catalog.py`: `OLLAMA_API`:22, `OLLAMA_TAGS_TIMEOUT_S`/`OLLAMA_NEG_TTL_S`:23–24, `_ollama_neg_*`:26–27, провайдер-запись `{"id":"ollama",...}`:125–127, вызовы `ollama_tags()` в `list_infos/list_status`:213,230, статус `"start ollama"`:262, `ollama_tags`:286–288, `ollama_model_names`:297–324, валидация `ref_kind=="ollama"`:415–418, `backend or "ollama"`:439, `return "ollama"`:489, `resolve_component` `llm:ollama:tag`:505–517. `downloaders.py`: импорт `OLLAMA_API`:14, диспатч:50–52, `download_ollama`:143–191. `server/service.py`: импорт:19, `provider_id "ollama"`:1259, фильтр:1288, `start_ollama_install`:1376–1407, `start_ollama_serve`:1410–1418, `ollama_fn` в meta/poll:1453–1472. `server/fastapi_app.py`: импорты:28,32,78–79, `ollama_fn=`:477,492,505,527,531, `"ollama"` в `GET /api/providers`:1122, `GET /api/ollama`:1175–1177, `POST /api/ollama/install|start`:1179–1193, `_ollama_payload`:1216–1226. CLI-тексты: `cli.py:195,251,862`; `config.py:180`; `ports/llm.py:7`. Доки: `docs/MODELS.md:48`, `docs/PARAMS.md:36`, `README.md:53,97,100`, `.env.example:13` (`VIDEOCLEAN_OLLAMA_URL`).

### 1.7 Сервер и фронт (что задето)

- `fastapi_app.py` — `POST /api/jobs` (568): форм-поля 577–655, среди них **`llm_place`:621, `llm_model`:622, `llm_base_url`:598, `llm_api_key`:599, `profile`:645, `preset`:651**; `GET /api/options`:1167 → `options_payload` (`service.py:1282–1303`): `llm_places`, `llm_models`, `llm_model_options`, `profiles`, `families`, `providers`.
- `service.py` — `serialize_clean_form` (`"profile"`:156), `flatten_preset_payload`:427–458 (слой рецепта для fast/balanced/quality), `_reject_unknown_preset_keys` (`"profile"`:559), `assemble_run_config`:588–627 (defaults → preset → PROFILE_KEYS → explicit через `merge_run_config`).
- webui — `widgets/stage-rail/params.ts`: `PROFILE_KEYS`-зеркало:71–81, `builtInProfileDefaults`:191–246, `DEFAULT_PARAMS = cleanBuiltinParams("balanced", "cpu")`:371, `explicitFields`:524–552 (что уходит на wire), `SAVE_KEYS`:136–179 (41 поле); `stage-rail.tsx`: `SCENARIOS`:56–60, селектор:509–547, CRUD пресетов:412–440; `expert-stations.tsx` — pro-станции (фраза/поиск рамок/сегментация/заливка/проверка/устройство) — **уже готовый pro-раздел**; `pages/config/index.tsx` — Ollama-секции (548–570, 599–605, 683–785, 845), `AddProviderDialog`:365–457 (форма «OpenAI-compatible провайдер»: title/base_url/api_key/models → `POST /api/providers`:521–534) — **уже готовая нужная форма**; `shared/events/*` — `ollama` в снапшоте (`types.ts:10`, `sse.ts:9`, `EventsProvider.tsx:66,74`); `stage-parts.tsx` `LlmChip`:301–371 (ollama-фолбэк:314–323).
- Тесты (16 файлов задеты): `test_api_ollama_setup.py` (удалить), `test_model_catalog.py` (100–139, 205–208, 253–279), `test_llm_resolve.py`, `test_cli_serve_helpers.py` (фикстура:22–23), `test_config.py`, `test_cleanup_payload.py`, `test_cli_preview.py`, `test_build_prompt.py`, `test_llm_parser.py`, `test_prompt_*`, `test_tunables.py`, `test_profiles_verify_inpaint.py`, `test_preset_merge.py`, `test_api_presets.py`, `test_hole_policy.py` (45, 49).

### 1.8 SLA-гейт — `scripts/check_sla_report.py`

`check`:26–77 ждёт из `report.json`: `budget{warm,device,vramCeilingMb,inpaintMaxSide,cpuThreads,cpuCount,vramBudgetMb,vramPeakInpaintMb}`, `hole{policy,limitedBy,batch,frameCount}`, `timings{load,decode,parse,detect,track,segment,inpaint,verify,encode,package}` (TIMING_KEYS:12–23), wall ≤ `--max-wall-s`, `|wall − Σtimings| ≤ 5`. Производитель — `run_cleanup.py:488–506`. **Контракт полей сохраняем** (добавления допустимы, переименования — нет).

---

## 2. Форензика лога (исправленная; детали — `REPORT_2`, Приложение A)

Доказано по `temp/logs.txt` + `git show`:
- В логе работали **три версии кода**: `8cf076c` (12:20) → `434b297` (13:41) → `58fa32d` (14:01) — замеры джоб несопоставимы между собой.
- Пять джоб → пять tqdm-баров пропагации. Джобы 3 и 4 **прерваны удалением** (останов на 499/672 и 305/672). У джобы 5 два бара: **OOM на 581/672 → полный повтор** с `force_offload=True` (retry из `c9018ed`), каждый повтор пересоздаёт state и 8.4 ГБ тензор `sam2.f32`.
- Прежнее утверждение «propagate отдельно на каждый трек» **отозвано**: обе версии кода зовут `propagate_in_video` один раз на все треки (`sam2_video.py:195–235` / старая 81–142).
- `nvenc=False`, `sam2._C` не собран, Ollama недоступен.

Следствие для плана: замеры SLA — только на фиксированном коммите (§7.9), а «повтор пропагации с нуля» — главный враг бюджета (§7.1).

---

## 3. Целевой скелет (кратко; контракты — `PLAN_OVERLAY_ERASER.md` §2)

Статичный пайплайн: `INTENT → DETECT → TRACK → MASK → ROUTE → ERASE → TEMPORAL → VERIFY → PACKAGE`. Существующие порты переиспользуются; добавляются три новых:

```python
# application/ports/intent.py
class IntentResolver(Protocol):
    def status(self) -> str: ...
    def resolve(self, prompt: str, *, targets_override, frames, frame_indices, parse_chunk_frames: int) -> Intent: ...

# application/ports/temporal.py
class TemporalConsistency(Protocol):
    def stabilize(self, images, masks, cleaned) -> None: ...   # правит cleaned на месте

# application/ports/verifier.py
class Verifier(Protocol):
    def check(self, images, masks, cleaned, *, cap: float) -> VerifyResult: ...  # dirty flags + note
```

Реестр — `adapters/registry.py`:

```python
@dataclass(frozen=True)
class AdapterInfo:
    id: str            # "erase.lama"
    stage: str         # intent | detect | mask | erase | temporal | verify | route
    factory: Callable[..., Any]
    scope: str         # overlay | object | any
    cost: str          # s | m | xl
    experimental: bool = False
    models: tuple[str, ...] = ()   # id из ModelCatalog
    note: str = ""     # лицензии/ограничения
```

Ключи конфига (один на стадию; никаких алиасов): `preset`, `intent_adapter`, `detect_adapters` (список), `mask_adapter`, `erase_adapter`, `erase_strategy` (`per_frame|keyframe_warp`), `temporal_adapter`, `verify_adapter`, `route_adapter`.

---

## 4. Фаза 1 — выпиливание Ollama и упрощение LLM-контракта

### 4.1 Удалить файлы
- `videoclean/adapters/llm/ollama_setup.py` (целиком);
- `videoclean/adapters/llm/llama_cpp.py` (целиком) — решение D1 подтверждено;
- `tests/test_api_ollama_setup.py` (целиком).

### 4.2 `adapters/llm/resolve.py` — переписать (контракт «только openai-compatible»)
Оставить: `UnconfiguredLlm`, `_provider_credentials` (форма провайдеров `catalog.load_providers`). Удалить: `DEFAULT_LOCAL_URL` (Ollama-эндпоинт!), `DEFAULT_LOCAL_MODEL`, `DEFAULT_OPENAI_URL`, `DEFAULT_OPENAI_MODEL`, `is_local_url`, `_looks_gguf`, логику `place` auto|local|cloud. Новый `resolve_llm(cfg)`:
1. если `cfg.llm_model`/`cfg.llm_base_url` — найти провайдера в `load_providers()` (по base_url или списку моделей) и взять `api_key`;
2. иначе явные `cfg.llm_base_url` + `cfg.llm_api_key` + `cfg.llm_model`;
3. иначе `UnconfiguredLlm("no LLM provider configured — add one in Settings or pass --llm-*")`.
Всегда `OpenAiCompatLlm` (base_url обязателен). env-переменные `VIDEOCLEAN_LLM_*`/`OPENAI_*` — оставить как есть.

### 4.3 `application/config.py`
Удалить `llm_place` (149) и `LLM_PLACES` (15); правки docstring-комментария (180). `llm_model`/`llm_base_url`/`llm_api_key` остаются.

### 4.4 `catalog.py` / `downloaders.py` — выпилить §1.6 (строки перечислены там).
Оставить `load_providers/save_providers` (526+) — это и есть единственный LLM-контракт. Семейство `ref_kind=="ollama"` и backend `"ollama"` удаляются; LLM-extra-компоненты (`extra:llm:*`) остаются только с HF-бэкендом, если нужны — ⚠️ проверить `family_backends("llm")`:386 при реализации.

### 4.5 Сервер
- `fastapi_app.py`: удалить §1.6 (импорты, `ollama_fn=`, `GET /api/ollama`, `POST /api/ollama/install|start`, `_ollama_payload`, `"ollama"` из `GET /api/providers`:1122); форм-поле `llm_place`:621 удалить.
- `service.py`: удалить §1.6; в `options_payload` убрать `llm_places`; `llm_model_options` строить только из провайдеров и LLM-extra (без ollama-ветки:1259, 1288).

### 4.6 CLI
`cli.py`: удалить `--llm-place` (и связанные места), поправить help-тексты 195/251/862; `backends`-команда — убрать ollama-строки.

### 4.7 webui
- `pages/config/index.tsx`: удалить Ollama-секцию (548–570, 599–605, 683–785, 845) и ollama-ветку `AddModelDialog` (272–282, 321–342); карточка «Ollama» заменяется обычным провайдером из `AddProviderDialog` (365–457 — уже готово).
- `shared/events/types.ts:10`, `sse.ts:9`, `EventsProvider.tsx:66,74` — удалить `ollama` из снапшота.
- `stage-parts.tsx:314–323` — убрать ollama-фолбэк `LlmChip`; `expert-stations.tsx:137–147,189,252–254` и `stage-rail.tsx:244–245` — убрать `snap.ollama.models` из авто-выбора.

### 4.8 Доки
`docs/MODELS.md:48`, `docs/PARAMS.md:31–40,108–124,139–140` (LLM-разделы переписать под «openai-compatible провайдер»), `README.md:53,97,100`, `.env.example:13`, `AGENTS.md` — убрать ollama.

**Критерий готовности фазы:** `grep -ri ollama` по репозиторию = 0 (кроме этого плана и REPORT'ов); `uv run pytest -q` зелёный; форма «Добавить провайдера» — единственный путь подключения LLM (в т.ч. self-hosted Ollama через `http://…:11434/v1`).

---

## 5. Фаза 2 — удаление алиасов профилей, новый `presets.py`

### 5.1 `application/presets.py` (новый; заменяет `profiles.py` целиком)

```python
PRESETS: dict[str, Preset] = {
  "overlay-fast":     Preset(...),   # дефолт
  "overlay-quality": Preset(...),
  "classic-quality": Preset(..., experimental=False, legacy adapters),
  "objects-experimental": Preset(..., experimental=True),
}
def presets_payload(device: str) -> list[dict]: ...          # для /api/options
def merge_run_config(defaults, preset_flat: Mapping | None, explicit: Mapping, *, resolve_device) -> dict: ...
```

- `Preset` = `{id, label, hint, config: dict (плоский diff конфига), experimental: bool}`.
- **Состав пресетов на старте фазы** (пока новые адаптеры не сделаны) — на текущих адаптерах; по мере фазы 4–5 состав меняется **только данными**: `overlay-fast` = `{segmenter: sam2, inpainter: lama, verify: false, ...}`, `overlay-quality` = `{sam2, lama, verify: true, ...}`, `classic-quality` = `{sam2-video, propainter, ...}`, `objects-experimental` = тот же состав + флаг. Имена не меняются никогда.
- `merge_run_config`: defaults → preset → explicit (слой `PROFILE_KEYS` исчезает; явные поля всегда побеждают).

### 5.2 `application/config.py`
Удалить `profile` (163) и его ветку в `validate()` (255–258). `preset_id` (172) → **`preset: str = "overlay-fast"`** (одно поле: имя встроенного пресета или id сохранённого; сервер разрешает сохранённый в плоский diff).

### 5.3 Сервер
- `service.py`: `serialize_clean_form` (`"profile"`:156 → `preset`), `flatten_preset_payload`:427–458 (**слой рецепта удаляется** — payload сохранённого пресета = полный плоский diff), `_reject_unknown_preset_keys` (559), `assemble_run_config`:588–627 → `merge_run_config(defaults, preset_flat, explicit)`, `options_payload`: `"profiles"` → `"presets"` (1302, 41–44 → `presets_payload`).
- `fastapi_app.py:645`: форм-поле `profile` удалить; `preset`:651 — расширить: «имя встроенного или id сохранённого».
- `api_schema.py:31,35` — обновить тексты.

### 5.4 CLI
`cli.py`: `--profile` (702–706) → `--preset`; `config_from_flags` (`composition.py:73,83,124,136,177`) — убрать `merge_run_config`-вызов профилей, передавать `preset`.

### 5.5 webui
- `params.ts`: удалить `PROFILE_KEYS` (71–81), `builtInProfileDefaults` (191–246), `inferScenario`/`isProfile`-ветки (484–501, 606–608, 678–680, 697–709); `Scenario` (52–55) → `{kind:"preset", id}`; `explicitFields`:524–552 — шлёт **полный diff + `preset`**; `DEFAULT_PARAMS` (371) = overlay-fast; `presetPayload`:554–565 — плоский diff (без `profile`).
- `stage-rail.tsx`: `SCENARIOS` (56–60) → список из `/api/options.presets` (label+hint отдаёт сервер); селектор:509–547 упрощается (пресеты включая сохранённые); CRUD:412–440 без изменений по API.
- `expert-stations.tsx` — остаётся pro-разделом как есть (это уже «выпадашки по стадиям»).

### 5.6 Тесты
Переписать: `test_profiles_verify_inpaint.py`, `test_preset_merge.py`, `test_api_presets.py`, `test_hole_policy.py:45,49`, `test_config.py:91`; новые: `test_presets.py` (составы, приоритет explicit, unknown preset → ошибка).

**Критерий готовности:** `grep -rn "fast\|balanced\|quality" videoclean/ server/ webui/src/ --include="*.py" --include="*.ts" --include="*.tsx"` не находит профильные имена; UI показывает 4 пресета с русскими label/hint.

---

## 6. Фаза 3 — скелет пайплайна: реестр, порты, разбор `run_cleanup`

### 6.1 `adapters/registry.py` (новый)
`AdapterInfo` + `REGISTRY` со всеми текущими реализациями (11 штук из `PLAN_OVERLAY_ERASER.md` §3.2), `get(stage, id)`, `list_stage(stage)`, `experimental`-флаги. `composition.py::make_detector/make_segmenter/make_inpainter/make_parser` (219–276) становятся тонкими обёртками над реестром (сигнатуры сохранить — их зовут `build_run_cleanup`:296 и `build_job_worker`:384+).

### 6.2 Новые порты (§3) + обёртки над текущим кодом
- `intent.llm` = `LlmPromptParser` (`adapters/prompt/llm.py:74`) — как есть.
- `intent.rules` (новый `adapters/prompt/rules.py`) — ядро из `refine.py` (`refine_intent`:116, `_TEXT_QUERIES`:12, `_MARK_QUERIES`:26, `where_from_prompt`:100) + `locations.py` (`mask_regions`:31, `normalize_where`:56) + чипы типов (`KINDS` `llm.py:15` расширить `logo|badge|label|timestamp`). **Дефолт INTENT.** Детерминированно, без LLM, мгновенно.
- `verify.residual` = обёртка `verify_quality.py` (существующая логика + ROI-фикс §7.6).
- `temporal.none` = заглушка.

### 6.3 Разбор `run_cleanup._run` на стадии (behavior-preserving)
`RunCleanup.__init__`:40–67 принимает новые порты (`intent`, `verifier`, `temporal`, `router`). Методы `_run`:140+ разбиваются на `_stage_intent / _stage_detect / _stage_track / _stage_mask / _stage_route / _stage_erase / _stage_temporal / _stage_verify / _stage_package` — логика переносится 1-в-1, порядок фиксируется и больше не меняется. Точки, где код «знает» про конкретные адаптеры, устраняются:
- `family = "propainter" if cfg.inpainter == ...` (643, 708) → `registry.get("erase", cfg.erase_adapter).family`;
- `_ready_inpainter`:601 / `_prepare_runtime`:558 — через `AdapterInfo.models` + `weights_cache.get_or_load` (см. §7.3);
- ручные входы (`masks_override`/`tracks_override`/`targets_override`) остаются ветками **ввода** (это часть контракта ввода, не адаптеры).

### 6.4 Поведенческий гейт
До/после рефакторинга прогон эталонного ролика: `report.json` (targets, detections, hole.policy, verifyNote, meanMaskCoverage) обязан совпасть; `timings` — в пределах шума. Иначе рефакторинг не принимается.

---

## 7. Фаза 4 — P0-фиксы скорости (по фактам лога)

| # | Что | Где (факт) | Изменение | Ожидание |
|---|---|---|---|---|
| 7.1 | OOM-retry делает **полный** повтор пропагации и пересоздаёт state+тензор | `sam2_video.py:172–184` (retry), `:264–322` (`_init_state_from_frames`), `:324–346` (`_image_tensor`) | резюмируемый retry: сохранять готовые `acc`-кадры, перезапускать с последнего завершённого кадра; тензор `sam2.f32` строить **один раз** (файловый кэш уже предусмотрен `tensor_path` — переиспользовать при повторе) | −8 мин на OOM-случае (как в джобе 5) |
| 7.2 | `sam2.f32` ≈ 8.4 ГБ + per-frame `_normalize_frame` в Python | `sam2_video.py:324–346`, `image_size=1024` из предиктора (`:270`) | батчевый resize/normalize (cv2/torch, GPU где возможно); для оверлей-класса `image_size=512` (деталь D3) | −2–4 мин преамбулы, −8.4 ГБ диска |
| 7.3 | Веса пересоздаются на каждую джобу (эйгр-`ensure` в `_prepare_runtime`) | `weights_cache.py` есть (`get_or_load`:33), но используется **только** `lama.py:280–296`; `sam2.py:114–125`, `sam2_video.py:165`, `grounding_dino.py:180`, `propainter` — свои `from_pretrained` | перевести все фабрики на `get_or_load` (ключ `WeightKey(kind, model_id, device, dtype)`) + warm-воркер между jobs | −1–2 мин на джобу |
| 7.4 | `propainter_raft_iter` = 20 | `config.py:203`, ctor `propainter.py:35` | дефолт → 5 (значение из статьи ProPainter) | ×~4 на flow-этапе |
| 7.5 | ProPainter `_transformer`: CPU-композитинг + осреднение окон 0.5 | `propainter.py:325–369` (`pred_img.cpu()...`, `binary_masks.cpu()`, np-смесь, `*0.5` blend) | композитинг на GPU, overlap-блендинг вместо усреднения | ×~2 на видео-инпейнте |
| 7.6 | Verify считает `absdiff` по всему кадру на каждый кадр | `verify_quality.py:26–27`, вызов `run_cleanup.py:359–362` | считать только по bbox масок (ROI); `verify_redetect` оставить опцией | −1–2 полных прохода |
| 7.7 | LaMa fp32 (ограничение cuFFT JIT-FFC) | `lama.py:286–293` (комментарий + `model.float()`), dtype в `hole_policy.probe_lama_dtype`:256 | fp16 через pad входа до степени двойки | ×1.5–2 |
| 7.8 | `nvenc=False` в контейнере; `sam2._C` не собран | лог (Приложение A); выбор энкодера `run_cleanup.py:596–599` уже поддерживает nvenc | образ с ffmpeg+h264_nvenc; собрать `sam2._C` (либо перейти на transformers-бэкенд `sam2.py:114–115`) | encode → секунды; +качество масок |
| 7.9 | Замеры шли на трёх версиях кода | §2 | SLA-гейт (`scripts/check_sla_report.py`) запускать на фиксированном коммите; `--require-warm` обязателен | честная базовая линия |

**Целевая арифметика (расчёт, не замер):** джоба 5 = 32 мин. Вычитаем: повтор пропагации (7.1) −8, преамбула (7.2+7.3) −4, ProPainter (7.4+7.5) −5, verify (7.6) −1 → **~14 мин** на `classic-quality`; оверлей-путь (фаза 5) выводит типовой случай на 1–2 мин.

---

## 8. Фаза 5 — оверлей-адаптеры (новые реализации; каждый — отдельный файл, ноль правок пайплайна)

| Адаптер | Файл (новый) | Что делает / на чём основан | Цена |
|---|---|---|---|
| `intent.rules` | `adapters/prompt/rules.py` | чипы типов + эвристики `refine.py`/`locations.py` (текущие) | мгновенно |
| `detect.ocr` | `adapters/detectors/rapid_ocr.py` | текст → пиксельные маски глифов (RapidOCR, Apache-2.0; ~35 мс/стр FP16 [И]) | s |
| `detect.persistence` | `adapters/detectors/persistence.py` | декоратор «оверлей vs сцена»: bbox стабилен сквозь смены сцен (IoU>0.7 на ≥60% сэмплов) | s |
| `mask.static` | `adapters/segmenters/static_mask.py` | одна маска на видео/сегмент; ядро — `_or_masks`/`_tracks_from_mask_anchors` (`run_cleanup.py:918–965`) уже умеет static | s |
| `mask.keyframes_track` | `adapters/segmenters/sam2_keyframes.py` | SAM2-image на 1–3 ключевых (`sam2.py` — уже есть) + LK-смещения (`detectors/_cv.py` трекер уже есть) + `interpolate_gaps` | s |
| `mask.ocr_glyphs` | `adapters/segmenters/ocr_glyphs.py` | маски глифов + dilate → треки | s |
| `erase.migan` | `adapters/inpainters/migan.py` | MI-GAN (MIT) для мелких ROI | s |
| `erase.keyframe_warp` | `application/erase_strategy.py` | инпейт 10–30 ключевых + warp/копирование остальных (стратегия над `Inpainter`) | s |
| `erase.alpha_unblend` | `adapters/inpainters/alpha_unblend.py` | «снятие» полупрозрачной накладки до заливки | s |
| `temporal.warp_ema` | `adapters/temporal/warp_ema.py` | keyframe-warp + EMA по маске | s |
| `verify.ghost_flicker` | `adapters/verify/ghost_flicker.py` | ghost-гейт (градиентная энергия зоны маски) + flicker (warp-error) | s |
| `erase.sttn` | `adapters/inpainters/sttn.py` | clip-инпейнт по ROI (⚠️ лицензия STTN — проверить перед включением в пресеты) | m |

Затем — обновление **состава** пресетов `overlay-fast`/`overlay-quality` данными (§5.1).

---

## 9. Фаза 6 — UX/API (по фактическим файлам)

1. **webui — сценарии/пресеты** (задето в §5.5): селектор «Сценарий» показывает пресеты из `/api/options`; pro-раздел = существующие `expert-stations.tsx` (без изменений структуры).
2. **webui — карточки оверлеев** поверх существующего превью-конвейера (`features/detect-run`: `submitJob({kind:"preview"})`, артефакты `GET /api/jobs/{id}/preview/*`): новый виджет-грид кандидатов (тип-чип, позиция, тайм-диапазон, тумблер) + кнопка «Проверить на 3 секундах» (job с `count≈75` кадров). Данные — из стадий DETECT/TRACK (новые адаптеры делают их точнее без правок UI).
3. **webui — LLM**: `LlmChip` (`stage-parts.tsx:301–371`) только из `llm_model_options` (провайдеры); пустое состояние — «Добавьте openai-compatible провайдера в Настройках» со ссылкой на уже существующий `AddProviderDialog`.
4. **API — breaking changes (фиксируем в `api_schema.py`)**: удалить `GET /api/ollama`, `POST /api/ollama/install|start`; `POST /api/jobs`: поле `profile` → `preset`, поле `llm_place` удаляется; `GET /api/options`: `llm_places` удаляется, `profiles` → `presets`, добавить `adapters` (каталог реестра с `scope/cost/experimental`); `GET /api/providers` — без блока `ollama`. Контракт `POST /api/providers` (title/base_url/api_key/models) — **остаётся как есть**.
5. **Экспериментальные пресеты** — в UI за переключателем «Расширенные возможности», в API только при явном `preset`.

---

## 10. Тесты и гейты

Переписать (см. §1.7 список): профильные и ollama-тесты → под новые контракты. Новые: `test_registry.py`, `test_presets.py`, `test_intent_rules.py`, `test_verify_roi.py`, `test_persistence_filter.py`. Поведенческий гейт рефакторинга — §6.4. SLA-гейт — `scripts/check_sla_report.py` (контракт полей report.json не менять; `report["profile"]`/`["presetId"]` → `report["preset"]`).

---

## 11. Решения по спорным пунктам — **приняты 2026-09-29**

| # | Решение | Статус | Замечание |
|---|---|---|---|
| D1 | Удалить `llama_cpp.py` (GGUF-раннер) вместе с Ollama | **принято (подтверждено владельцем)** | локальный GGUF ≠ openai-compatible, text-only; локальные сценарии закрываются своим сервером через форму провайдера |
| D2 | Имена пресетов: `overlay-fast`, `overlay-quality`, `classic-quality`, `objects-experimental` | **принято** | продуктовая лексика; детали — `PLAN_OVERLAY_ERASER.md` §6 |
| D3 | `image_size=512` для сегментации в оверлей-пресетах (сейчас 1024) | **принято**, валидация — гейтом | маски оверлеев простые; ×4 экономии на тензоре/инференсе; если SLA-гейт по качеству покажет просадку масок — вернуть 1024 только для `overlay-quality` |
| D4 | Переименование в `report.json`: `profile`/`presetId` → `preset` | **принято** | внутренний контракт; `scripts/check_sla_report.py` правится синхронно (§10) |

---

## 12. Порядок работ (чек-лист)

**Неделя 1 — правки контрактов (фазы 1–2):**
- [ ] Фаза 1: удаление Ollama (§4.1–4.8), `resolve_llm` переписан, `grep -ri ollama` = 0.
- [ ] Фаза 2: `presets.py`, удаление `profiles.py`/`profile`-поля, server/cli/webui переведены на `preset`.
- [ ] `uv run pytest -q` + `bun run lint && bun run typecheck` зелёные; `docs/PARAMS.md` переписан (пресеты, без LLM-ollama).

**Неделя 2 — скелет + скорость (фазы 3–4):**
- [ ] `adapters/registry.py` + 3 новых порта + обёртки (`intent.rules`, `verify.residual`, `temporal.none`).
- [ ] Разбор `run_cleanup` на стадии (behavior-preserving) + гейт §6.4.
- [ ] P0-фиксы 7.1–7.8; прогон `classic-quality` до/после → цифры в `REPORT_2`, приложение B.

**Недели 3–4 — оверлей-адаптеры (фаза 5):**
- [ ] `mask.static`, `mask.keyframes_track`, `detect.ocr`, `detect.persistence`, `intent.rules`-чипы.
- [ ] Обновить состав `overlay-fast/quality` данными; SLA-гейт: p50/p95 на эталонном наборе.

**Неделя 5 — качество и UX (фазы 5–6):**
- [ ] `temporal.warp_ema`, `erase.alpha_unblend`, `verify.ghost_flicker`.
- [ ] Карточки оверлеев, превью «3 секунды», API-правки (§9.4), доки интеграции.
