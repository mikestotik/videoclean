# Спека: видеоредактор вокруг единого пайплайна (sources, interpret, detect, inpaint)

Дата: 2026-09-08
Статус: черновик на ревью

## 1. Цели

Один пайплайн удаления объектов с двумя точками входа и контролем на каждой ступени:

- **Точка входа A:** промпт на любом языке (LLM нормализует).
- **Точка входа B:** нарисованные маски на кадрах (Vision-LLM описывает выделенное).
- Ступени: **Интерпретация → Detect → Inpaint**. В ручном режиме между ступенями можно стоять, менять параметры и перезапускать; стоп после Detect — штатный сценарий.
- **Авто-режим** — тот же пайплайн одной кнопкой без остановок (сегодняшнее поведение `RunCleanup` для текстового промпта; для масок — клиентская цепочка prompt-job → run-job).
- Загрузил видео → сразу редактор: скраб кадров, рисование масок, без единого запуска.
- Все параметры `PipelineConfig` доступны в UI с дефолтами.
- Пресеты параметров (серверные, для будущего API-сценария).
- История: вернуться к видео (source), скорректировать, перезапустить.

## 2. Не-цели (v1)

- Инпейнт в preview-джобе (проверка качества заливки на одном кадре) — будущее.
- Правка найденных масок вручную (только просмотр оверлея; правка = перезапуск detect с другими параметрами или свои маски через masks_override).
- Мультипользовательский доступ, роли.
- Промежуточное хранение onnx/новые детекторы.

## 3. Модель пайплайна

```
Вход: промпт (любой язык) и/или маски-аннотации
   → ИНТЕРПРЕТАЦИЯ (kind="prompt"): VLM → targets[] + человекочитаемый промпт
   → DETECT (kind="preview"): Grounding DINO → SAM2 → маски на кадрах
   → INPAINT (kind="run"): заливка масок → мезонин → пакеты форматов
```

- Интерпретация и Detect — опциональные в ручном режиме: можно начать с текста и сразу Detect (mode=parse), или с таргетов из интерпретации (mode=detect), или с масок и Inpaint без детекта.
- Inpaint (kind="run") три режима входа:
  1. **По найденным трекам** (`tracks_override` из отчёта Detect) — parse+detect пропускаются, SAM+inpaint по готовым трекам. Основной ручной путь.
  2. **По своим маскам** (`masks_override`) — parse+detect+segment пропускаются; маски пользователя применяются ко всем кадрам (случай статичного логотипа; UI предупреждает про движущиеся объекты).
  3. **По промпту/таргетам заново** (`targets_override` или полный parse) — детект выполняется в run-джобе.
- Отчёты джобов несут сквозной контекст: preview-отчёт содержит `tracks` (уже `tracks_to_json`) — он же вход `tracks_override`; prompt-отчёт содержит `targets` — он же вход mode=detect и `targets_override`.

## 4. Библиотека: изменения

### 4.1 Новый use case `BuildPrompt` — `videoclean/application/use_cases/build_prompt.py`

```python
@dataclass
class BuildPromptRequest:
    input_path: Path
    prompt: str | None            # текст пользователя (любой язык), может быть пуст
    annotations: list[dict]       # [{"frame": int, "mask": Path}, ...] — путь к PNG маски
    config: PipelineConfig
    job_id: str | None = None
```

Зависимости (как у RunPreview): `media: MediaGateway`, `llm: LlmClient`, `jobs: JobStore`, `progress: ProgressPort`, `new_job_id`, `make_paths`, `read_image`, `write_image`.

Поведение:
1. probe; выделенные кадры (`extract_frames_subset` по индексам аннотаций; если аннотаций нет — `sample_frame_indices` по `prompt_frame_stride`/`prompt_frame_max`).
2. Для каждой аннотации: наложить маску на кадр полупрозрачным красным (50%, как `_write_overlays` в RunPreview) → JPEG bytes.
3. Один вызов `llm.complete(system, user, images=[...])`. System-промпт — новый шаблон `interpret_system.md` в `videoclean/adapters/prompt/prompts/` (требования: ответ строго JSON `{"prompt": "<human prompt in user's language>", "targets": [{kind, query, where?, motion?, ordinal?, from_side?}]}`; query — короткие английские визуальные имена для Grounding DINO; kind ∈ watermark|text_overlay|object; where ∈ SCREEN_WHERES).
4. JSON парсится (тот же стиль устойчивого парсинга, что в `adapters/prompt/llm.py`); валидируется через `targets_from_json`; `prompt` — произвольная строка (fallback: перечисление queries).
5. Отчёт: `{jobId, kind:"prompt", state, prompt, targets:[...], framesUsed, imagesSent, llmModel, parseMode:"interpret"}`. Job → COMPLETED/FAILED как обычно.

Валидация входа: хотя бы текст или одна аннотация, иначе `PipelineError`.

### 4.2 `RunCleanupRequest` — расширения (`application/config.py` + `run_cleanup.py`)

- `tracks_override: list[dict] | None = None` — JSON как `tracks_to_json`. В `_run`: если задан → пропустить parse и detect; `selected = tracks_from_json(tracks_override)`; `parseMode="manual-tracks"`; `detectorUsed="manual"`.
- `masks_override: dict[int, Path] | None = None` — ключ кадр-источник аннотации → PNG. Пропустить parse+detect+segment; маски читаются, dilate применяется, каждая маска копируется во **все** кадры (`masks/processed/{idx:06d}.png`), покрытие считается по факту; `parseMode="manual-masks"`.
- Контроль качества `min_mask_coverage` сохраняется для обоих режимов.
- `targets_override` — уже в датаклассе; добавить чтение из payload в `cleanup_request_from_row`.

### 4.3 `tracks_from_json` — `domain/tracks.py`

Обратная операция к `tracks_to_json` (track_id, label, boxes, scores, motion, part, notes). Невалидные строки пропускаются; пустой результат → `PipelineError`.

### 4.4 Composition + worker

- `composition.py`: `build_build_prompt(cfg, progress, jobs, job_id)` (llm = `resolve_llm(cfg)`); в `build_job_worker` — третий фабричный метод `build_prompt_runner` для `kind == "prompt"`.
- `application/jobs/worker.py`: dispatch по `payload["kind"]`: `"preview"` → preview, `"prompt"` → prompt, иначе cleanup.

## 5. Store: источники — `videoclean/store.py`

Новая таблица `sources` (миграция тем же `ALTER`-паттерном, где нужно):

```sql
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,          -- s_{utcnow}_{hex8}
    name TEXT NOT NULL,           -- исходное имя файла
    path TEXT NOT NULL,           -- data_dir/sources/{id}/input{suffix}
    created_at TEXT NOT NULL,
    probe_json TEXT,              -- MediaManifest после загрузки
    meta_json TEXT                -- {"width","height","duration_s","fps","frame_count",...}
);
```

`SourceIndex` (тот же файл jobs.sqlite): `register`, `list`, `get`, `delete` (+ удаление каталога `data_dir/sources/{id}`).

JobIndex: колонка `source_id TEXT` (nullable) тем же `ALTER`-паттерном; upsert принимает `source_id`; `job_dict` отдаёт `source_id` и `source_name`.

## 6. Сервер: API

Авторизация — без изменений (BasicOrBearer на всё, кроме /health).

### 6.1 Новые роуты

| METHOD | Path | Тело/квери | Ответ |
|---|---|---|---|
| POST | `/api/sources` | multipart `video` | 201 `{id, name, path, createdAt, probe:{fps,duration_s,width,height,frame_count,has_audio}}` |
| GET | `/api/sources` | — | `[{...Source}]` |
| GET | `/api/sources/{id}` | — | `{...Source}` / 404 |
| DELETE | `/api/sources/{id}` | — | `{ok, id}` (+ каталог) |
| GET | `/api/sources/{id}/video` | — | FileResponse (range-запросы — как сейчас для input/output) |
| GET | `/api/sources/{id}/frames/{n}.jpg` | — | JPEG кадра n; ffmpeg single-frame в `sources/{id}/frames/{n:06d}.jpg`, кэш на диске; 404 вне диапазона. *Опционально (v1.1): плеер работает на нативном `<video>`, этот эндпоинт нужен только API-пользователям и отладке* |
| GET | `/api/sources/{id}/annotations` | — | `{frames:[{frame, url, strokes, updatedAt}]}` (листинг каталога masks + geometry из meta) |
| PUT | `/api/sources/{id}/masks/{n}` | multipart: `mask` (PNG) + `strokes` (JSON — geometry штрихов) | 201 `{ok, frame}` — strokes сохраняются в `masks/{n:06d}.json` для повторного редактирования |
| GET | `/api/sources/{id}/masks/{n}` | — | PNG |
| DELETE | `/api/sources/{id}/masks/{n}` | — | `{ok}` |
| GET | `/api/presets` | — | `[{id, name, payload, createdAt}]` |
| POST | `/api/presets` | JSON `{name, payload}` | 201 `{id, ...}` (payload — словарь полей PipelineConfig) |
| DELETE | `/api/presets/{id}` | — | `{ok}` |

Хранение: `data_dir/sources/{id}/input{suffix}`, `.../frames/`, `.../masks/{n:06d}.png`. Пресеты — `data_dir/presets.json` (сервер пишет/читает; без библиотечных изменений).

### 6.2 Изменённые роуты

- `POST /api/jobs` — принимает `kind` (`run`|`preview`|`prompt`, по умолчанию `run`) и `source_id` вместо `video`; если пришёл файл — создаёт source неявно (как сейчас) и привязывает job. Поля по kind:
  - `run`: всё текущее + `targets` (JSON → `targets_override`), `tracks` (JSON → `tracks_override`), `masks` (JSON `[frame,...]` — кадры-источники аннотаций; пути резолвятся из масок source, поэтому требует `source_id`, иначе 400), `keep_workdir`, `min_mask_coverage`, `verify_max_coverage`, `llm_base_url`, `llm_api_key`, `formats` (список через запятую). `targets`/`tracks`/`masks` взаимоисключающие (400 при конфликте).
  - `preview`: текущие поля `/api/preview/from-job` + `source_id`. «Все кадры» = сервер сам подставляет `start=0, count=frame_count` (библиотечный `indices_from_request` при `count=None` даёт 16, поэтому клиентский «все кадры» сервер разворачивает явно).
  - `prompt`: `prompt` (опционально), `config`-поля парсера (`llm_*`, `vision_batch`, `prompt_frame_*`); аннотации берутся из масок source автоматически.
- `POST /api/preview` и `/api/preview/from-job` остаются для совместимости, фронт переезжает на `POST /api/jobs`.
- `serialize_clean_form` (service.py): добавить перечисленные поля; дефолты — дефолты `PipelineConfig`.
- Worker: `allow_download` по-прежнему `False` из HTTP.

### 6.3 job_dict

Добавить: `source_id`, `source_name`, `kind` уже есть (`run|preview` → добавить `prompt`).

## 7. Фронтенд (FSD)

### 7.1 Структура

```
entities/source/    types (Source), api (list/upload/delete, frameUrl, videoUrl, masks get/put/delete, annotations)
entities/annotation/ Annotation {frame, strokes[]}, Stroke {tool: brush|eraser|rect, points|rect, size},
                    чистые функции: strokesToMaskCanvas, maskCanvasToBlob, аннотации-локальный кэш
entities/preset/    types + api
entities/job/       +source_id в типе; api: submitJob(kind, ...) — единая точка
widgets/library/    левая колонка: источники + вложенные job'ы источника
widgets/editor-viewer/ видео + слои: аннотационный canvas, оверлей масок detect,
                    before/after (два <video> + вайп), таймкод, тулбар (кисть/ластик/undo/redo/clear/размер)
widgets/timeline/   миниатюры (клиентская генерация как сейчас) + маркеры: аннотации, кадры с масками
widgets/stage-rail/ правая колонка: Ступень 1 Вход, 2 Таргеты, 3 Маски, 4 Inpaint, 5 Результат;
                    кнопка «Запустить всё»; аккордеоны параметров с дефолтами; пресеты;
                    таргеты — редактируемый список с чекбоксами (искл. из Detect) и
                    бейджем источника («из маски 288» / «из текста»)
features/annotate/  pointer-рисование, undo-стек, автосохранение маски (debounce PUT)
features/interpret/ prompt-job submit+poll → заполнение ступени 2
features/detect-run/preview-job на source (indices все или stride) → манифест артефактов → оверлей
features/inpaint-run/ run-job с выбранным режимом входа → poll → результат
pages/workspace/    композиция; pages/config/ — без изменений (+ пресеты не здесь)
```

Публичные API слайсов — `index.ts`. Новые shadcn-компоненты по мере надобности: slider, tooltip, toggle-group, popover, dropdown-menu, switch.

### 7.2 Layout

Полноэкранное приложение (h-screen, без скролла страницы):

```
┌──────────┬────────────────────────────────────┬───────────────┐
│ Библиотека│  таймкод 00:00:12:04 · кадр 288   │ Ступени       │
│ источники │  ┌──────────────────────────┐     │ 1 Вход     ●  │
│ + job'ы  │  │   видео + слои масок     │     │ 2 Таргеты  ✓  │
│          │  └──────────────────────────┘     │ 3 Маски    ✓  │
│          ├────────────────────────────────────┤ 4 Inpaint  ●  │
│          │ таймлайн: ◉ annotated ◉ masked     │ 5 Результат   │
└──────────┴────────────────────────────────────┴───────────────┘
```

- Центр: **нативный `<video>`** (мгновенный скраб, петельный playback) + слои поверх: аннотационный canvas (штрихи в координатах видео), оверлей масок detect (`<img>` из артефактов, opacity-слайдер), before/after. Клавиши: ←/→ ±1 кадр, Shift+←/→ ±10, Space play/pause. Server frames endpoint в плеере не участвует.
- Before/after после run-джоба: два синхронных `<video>` (input/output) + вертикальный вайп-разделитель (draggable), переключатель вайп/стороны.
- Таймлайн кликом ставит текущий кадр; маркеры из annotations + preview manifest.

### 7.3 Дизайн-токены

- Домен — монтаж/цветокор, видео — герой: тёмный графитовый хром без карточек; панели тональными слоями и хайрлайнами.
- Палитра: bg `#141518`, panel `#1B1D21`, panel-2 `#22252A`, text `#E8E9EB`, muted `#9A9DA3`, hairline `#2A2D33`, **акцент safelight amber `#E8A33D`** (кнопки/активная ступень), красный `#E5484D` — только маски/деструктив, ok `#4CAF7D`.
- Шрифт: Inter Variable (есть) для UI; моно (`@fontsource/jetbrains-mono`) **только** таймкод и числовые значения, tabular-nums; таймкод — самый крупный числовой элемент.
- Надписи: sentence case, без all-caps eyebrow; без карточек-одинаковых-радиусов.
- Motion: только на действия пользователя (переключение ступеней, раскрытие аккордеона); `prefers-reduced-motion` уважается; focus-visible во всех контролах.

## 8. Сценарии (data flow)

1. **Загрузка:** файл → `POST /api/sources` → source в библиотеке → открыт редактор: probe, кадр 0, таймлайн генерирует миниатюры.
2. **Аннотация:** кисть на кадре 288 → debounce PUT mask/288 (PNG + strokes-JSON); кадр 1200 — вторая маска. Маркеры на таймлайне; после перезагрузки штрихи восстанавливаются из strokes.
3. **Интерпретация:** статус-чип LLM (готов/нет → ссылка в Конфиг); «Интерпретировать» → `POST /api/jobs kind=prompt source_id` → poll → `{prompt, targets}` в редактируемые поля ступени 2; у каждого таргета чекбокс и признак «пришёл из маски N / из текста».
4. **Detect:** «Найти» → `kind=preview mode=detect targets[...]` (или `mode=parse prompt`). «Всё видео» или авто-шаг (stride подбирается под цель ~400 кадров, можно вручную). Poll → манифест: оверлей масок на видео (кадр = `requestVideoFrameCallback`/currentTime), боксы с подписями, meanMaskCoverage; ETA и «Стоп» во время работы.
5. **Inpaint:** режим «по трекам» (tracks из отчёта detect) → `kind=run source_id tracks=...` → poll → результат: before/after вайп, форматы, скачивание.
6. **Авто:** «Запустить всё» при текстовом промпте = один `kind=run` (сегодняшний RunCleanup); при масках — клиентская цепочка: prompt-job → run-job с `targets_override` из результата.
7. **История:** клик по source в библиотеке → его job'ы, любой job можно посмотреть/повторить с правками.

## 9. Пограничные случаи и UX-риски

- Видео без аннотаций и текста: «Интерпретировать» disabled; run требует промпт/таргеты/маски — иначе 400.
- LLM недоступен (Ollama не запущена, модель не скачана): статус-чип в ступени 1 с ссылкой в Конфиг — до клика, не после.
- `meanMaskCoverage < min_mask_coverage` — job FAILED с понятной ошибкой в UI ступени.
- Длинные видео: detect-all дорог (SAM2 покадрово; CPU — минуты). Дефолт — авто-шаг под цель ~400 маскированных кадров с пометкой «маски интерполируются между кадрами»; ручной stride; ETA (job_dict) и «Стоп».
- Соответствие «маска → таргет» не гарантированно 1:1 (VLM обобщает): смягчается чекбоксами таргетов, бейджами источника и проверкой по боксам с подписями треков. Известное ограничение v1.
- Локальные VLM (llava-phi3) слабо читают русские промпты и маски-оверлеи: подсказка в UI — cloud LLM или qwen3-vl; vision_batch=2 для llava-phi3.
- sam2-video недоступен в preview — для detect-джобы сегментер форсируется в `sam2` (покадровый image predictor), независимо от выбора в конфиге run; в run доступен и `sam2-video`.
- Удаление source с активными job'ами (QUEUED/RUNNING у этого source_id) — 409. Прогресс upload — без него (v1): локальный файл копируется мгновенно.
- Старые job'ы до миграции (source_id NULL): секция «Без источника» в библиотеке, без редактора, только статус/скачивание.
- Range-запросы к `/video` — как для существующих `input_url`/`output_url`.

## 10. Тесты

- `tests/test_api_sources.py`: upload→list→get→frame{0,last}→mask put/get/delete→annotations→delete (ffmpeg-генерированный клип).
- `tests/test_build_prompt.py`: fake LlmClient — правильные images (кол-во), JSON парсинг, targets валидация, ошибки пустого входа.
- `tests/test_run_cleanup_overrides.py`: tracks_override (детектор не вызывается), masks_override (сегментер не вызывается, маски размножены на все кадры), targets_override из payload.
- `tests/test_api_jobs.py` (расширение): kind=prompt/preview/run c source_id; конфликт targets/tracks/masks → 400; presets CRUD.
- `tests/test_architecture.py`: build_prompt остаётся в application.
- Фронт: `bun run lint && bun run typecheck && bun run build`.

## 11. Риски

- `ALTER TABLE` для `sources.source_id` — тот же паттерн, что `_JOB_COLUMNS`; старые job'ы получают `source_id=NULL`.
- Маски-оверлеи как вход VLM — новая техника: маленькие локальные модели могут путаться; system-промпт явный, облачные модели надёжнее. Fallback пользователя — редактирование таргетов руками.
- Detect-all на длинных видео: много JPEG-артефактов (`3×N`); митигируется stride и предупреждением.
- Range-запросы: поведение FileResponse уже обкатано на input/output.
