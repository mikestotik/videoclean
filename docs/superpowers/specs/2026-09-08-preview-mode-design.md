# Preview mode + scoped targets: дизайн

Дата: 2026-09-08 · Статус: утверждено в диалоге (см. историю ниже) · План: `docs/superpowers/plans/2026-09-08-preview-mode.md`

## Задача

Пользователь запускает удаление по промпту и не видит, что делает пайплайн. Ошибки детекции (боксы не там, маски пустые, LLM собрал не те queries) видны только по финальному видео, итерация по параметрам медленная. Нужно: (1) показывать покрытие масками на выбранных кадрах до полного прогона, (2) показывать собранные LLM таргеты и давать править их руками, (3) привязать таргеты к участкам времени, чтобы VLM-парс на длинных видео с меняющимися оверлеями не схлопывал всё в один глобальный список.

## Решение

Две фичи, одна кодовая база:

1. **Preview mode** — прогон `parse → detect → select → sam2` на подмножестве кадров с визуальными артефактами. Без inpaint, mezzanine, verify, package.
2. **Scoped targets** — VLM-парс работает по временным чанкам; каждый таргет получает окно кадров, в котором он валиден. Детектор на ключевом кадре использует только таргеты своего окна.

В preview обе фичи работают вместе: пользователь выбирает кусок видео, смотрит разбор и маски, правит таргеты, повторяет. Утверждённые таргеты уходят в полный прогон.

## Preview mode

### Поток

```
POST /api/preview (или videoclean preview)
  → probe
  → extract только выбранных кадров (ffmpeg select='eq(n,a)+eq(n,b)+…')
  → parse (mode=parse) ИЛИ принять готовые targets (mode=detect)
  → discover на КАЖДОМ выбранном кадре (detector_keyframes = n_frames)
  → select_tracks
  → sam2.masks → оверлеи
  → артефакты + preview.json, джоба COMPLETED
```

Кадры превью = выбранные пользователем (по умолчанию 16 равномерных по клипу). Обязательное дополнение: индексы, где реально сработал детектор, и так уже видны — детектор в превью гоняется на всех выбранных кадрах, поэтому ключевые кадры = выбранные кадры. Отдельного объединения не требуется.

### Артефакты (workdir джобы, каталог preview/)

- `{idx:06d}_boxes.jpg` — кадр, красные прямоугольники, подписи `id label score`
- `{idx:06d}_mask.jpg` — кадр + полупрозрачная красная маска (после `mask_dilate_px`)
- `{idx:06d}_side.jpg` — пара «кадр с боксами | кадр с маской» для ленты
- `maskfilm.mp4` — слайдшоу из side-кадров на 2 fps (видно движение масок между кадрами)
- `preview.json` — queries, parseMode, targets, per-frame: индекс, боксы, покрытие маски; треки с coverage

### Инпейнтер не участвует

Маска, попавшая не туда, обесценивает любой инпейнтер. Качество заливки превью не показывает: TELEA мылит по определению, ProPainter требует весь клип как контекст. Это ограничение фиксируем в подписи UI.

### Ограничение sam2-video

Пропагация маски требует полного клипа, в превью сегментатор всегда покадровый `sam2` независимо от глобальной настройки. В превью-конфиге `segmenter` форсируется в `sam2`, оригинал сохраняется для полного прогона.

## Scoped targets

### Модель данных

```python
# domain/intent.py
@dataclass
class Target:
    kind: str
    query: str
    ...
    frames: tuple[int, int] | None = None  # [start, end) окно валидности, None = весь клип
```

`Intent.targets` остаются плоским списком. Новое поле опционально, весь существующий код совместим.

### Парс по чанкам

```
chunk = max(prompt_frame_max * prompt_frame_stride, parse_chunk_frames)  # кадров на чанк
шаг между чанками: перекрытие 0 (чанки впритык)
для каждого чанка: сэмпл ≤ prompt_frame_max кадров (stride), один vision-запрос на батч по vision_batch
tarгеты чанка получают frames=(start, end)
слив: соседние чанки с идентичными (kind, query, where, ordinal, from_side) → одно окно на объединение
```

Конфиг: `parse_chunk_frames: int = 0`. `0` = один чанк на весь клип, поведение идентично текущему. Рекомендованный тюнинг для длинных видео — в docs/PARAMS.md. Батчи внутри чанка — существующий `vision_batch`.

`parse_mode` в отчёте: `llm-vision` (один чанк), `llm-vision-scoped` (несколько чанков).

### Детекция с окнами

`RunCleanup._discover` и детекторы получают queries на кадр. Механика: discover вызывается как сейчас с полным списком уникальных queries, но в `select_tracks` трек фильтруется: боксы трека вне окна всех таргетов с этим query обнуляются до интерполяции. Реализация в `select.py` (чистая функция, тестируется без моделей).

## UI

### Панель «Превью» на рабочей странице

Поля: диапазон кадров (`start`, `count`, `stride`, дефолт `0/16/auto`) или свободный список индексов. Кнопки: «Превью: разбор» (parse + detect + masks), «Детекция» (detect + masks на существующих таргетах), «Перепарсить» (только parse). Валидация кнопок по стадиям.

### Карточка превью в истории

- Лента side-кадров в хронологическом порядке, клик — полноразмерный просмотр.
- Тумблеры «Боксы» / «Маски» на каждый кадр.
- Плеер `maskfilm.mp4`.
- Блок «Таргеты»: таблица `kind | query | where | motion | кадры`, каждая строка редактируемая, кнопки удалить/добавить. Кнопка «Детекция» с этими таргетами. Кнопка «Полный прогон» — POST /api/jobs с `targets_override` из текущей таблицы.

### API

- `POST /api/preview` — multipart как /api/jobs + `start`, `count`, `stride`, `mode` (`parse` | `detect`), `targets` (JSON при mode=detect)
- `GET /api/jobs/{id}/preview/{name}` — отдача `{idx}_boxes.jpg`, `{idx}_mask.jpg`, `{idx}_side.jpg`, `maskfilm.mp4`, `preview.json` (auth как у остальных эндпоинтов)
- `RunCleanupRequest.targets_override: list[dict] | None = None` — при задании парсер пропускается, Intent строится из JSON (kind/query/where/motion/ordinal/from_side; frames игнорируются — полный прогон без окон). CLI: `--queries "text [bottom], logo"` — парсинг строки в таргеты, взаимно исключает смысловой конфликт с `--prompt` (prompt становится необязательным при заданных queries).

## Конфиг

Новое: `parse_chunk_frames: int = 0`. Дефолт сохраняет текущее поведение целиком, оба дефолта прогонов идентичны до включения чанков.

## Файлы

| Файл | Изменение |
|---|---|
| `domain/intent.py` | `Target.frames` |
| `application/config.py` | `parse_chunk_frames`, валидация |
| `application/use_cases/run_preview.py` | новый use case |
| `application/use_cases/run_cleanup.py` | `targets_override`, scoped select |
| `application/ports/media.py`, `adapters/media/ffmpeg.py` | `extract_frames_subset` |
| `application/select.py` | фильтр треков по окнам |
| `adapters/prompt/llm.py` | чанковый vision-парс, окна |
| `composition.py` | `build_run_preview` |
| `cli.py` | `videoclean preview`, `--queries` |
| `adapters/web/fastapi_app.py`, `service.py` | preview-эндпоинты |
| `adapters/web/static/index.html`, `app.js` | панель, карточка, лента |
| tests/ | run_preview, select windows, ffmpeg subset, API, CLI |

## Порядок реализации

1. `Target.frames` + scoped select (select.py) — TDD, без UI.
2. Чанковый парс в llm.py — TDD.
3. `extract_frames_subset` в ffmpeg-адаптере — TDD на select-фильтре.
4. `RunPreview` use case + композиция — TDD с фейками.
5. CLI `preview`, `--queries` — TDD.
6. `targets_override` в RunCleanup — TDD.
7. API эндпоинты — TDD.
8. UI (панель + карточка).
9. docs/PARAMS.md, README; финальный прогон.

## Не делаем

- Параллельная обработка чанков в полном прогоне: трекинг, sam2-video и ProPainter зависят от временного контекста соседних кадров, границы чанков дают швы и разрывы треков. Ускорение мизерное: детектор и ProPainter на GPU и так загружают VRAM целиком.
- Inpaint в превью: не показывает ничего диагностического (см. выше).
- Живой просмотр стадий полного прогона: отдельная фича после проверки превью на практике.
- Ручная рисовалка масок: другой продукт.
