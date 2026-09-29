# План. Overlay Eraser: статичный пайплайн + подменяемые адаптеры (версия 2)

## Продукт: удаление логотипов, вотермарок, бейджей, лейблов и текста из видео · RTX 4090 · API + WebUI

> **База:** `REPORT_1_FAST_REMOVAL_RESEARCH.md` (технологии), `REPORT_2_CURRENT_PIPELINE_ANALYSIS.md` (разбор кода + прод-лог RunPod).
>
> **Принцип версии 2 (по решению владельца продукта): ничего не выбрасываем.** Все текущие реализации (SAM2-video, ProPainter, LLM-парс, verify) остаются в проекте как **адаптеры за портами**. Перестраивается не набор механизмов, а **скелет**: статичный пайплайн `INTENT → DETECT → MASK → ERASE → VERIFY`, где каждый этап — порт, под который можно подменять/добавлять реализации через конфиг и пресеты. Продуктовый фокус — оверлеи; механизмы «под любой объект» сохраняются как **экспериментальные адаптеры** и не мешают продуктовому SLA.

---

## 0. Четыре принципа архитектуры

1. **Пайплайн программно статичен** — порядок стадий и их контракты зафиксированы в коде оркестратора и **не меняются** при добавлении адаптеров. Это фундамент.
2. **Всё изменчивое — за портами.** Модели, эвристики, стратегии заливки и проверки — адаптеры. Новый компонент = новый файл адаптера + запись в реестре. Правок оркестратора — ноль.
3. **Ничего не удаляем, всё помечаем.** Адаптер получает метаданные `scope` (overlay | object), `cost` (s | m | xl), `experimental: bool`. «Экспериментальный» ≠ «недоделанный»: это адаптер, который **выходит за продуктовый SLA/назначение** (например, работает с любыми объектами, но медленно). Он доступен, поддерживается, но не входит в продуктовые пресеты и требует явного выбора.
4. **Пресеты собираются из адаптеров.** Продуктовые пресеты (оверлеи) — быстрые и качественные; классический тяжёлый путь остаётся пресетом `classic-quality`; «любой объект» — пресетом `objects-experimental`. Пользователь выбирает пресет; pro-пользователь — конкретные адаптеры по стадиям.

---

## 1. Инвентаризация: что уже есть (и никуда не денется)

| Есть в коде | Что это | Место в новом скелете |
|---|---|---|
| 10 Protocol-портов: `Detector`, `Segmenter`, `Inpainter`, `PromptParser`, `MediaGateway`, `LlmClient`, `ModelCatalog`, `ProgressPort`, `JobStore`, `JobControl` | интерфейсы стадий | **уже порты** — остаются как есть, дополняются 3 новыми |
| Фабрики `make_detector/segmenter/inpainter/parser` | выбор реализации | перерастают в **реестр адаптеров** (см. §5) |
| `application/profiles.py` (fast/balanced/quality) | пресеты | становятся тонкой обёрткой над новыми продуктовыми пресетами |
| `domain/intent.py`: `Target.kind = watermark \| text_overlay \| object` | домен намерения | **домен уже различает оверлей и объект** — расширяем значения (logo, badge, label, timestamp) |
| `adapters/prompt/refine.py`, `locations.py` | эвристики разбора промпта | ядро адаптера `intent.rules` |
| `adapters/llm/*`, `prompt/llm.py` | LLM-разбор | адаптер `intent.llm` (опциональный) |
| `adapters/detectors/grounding_dino.py`, `_cv.py` | детект + CV-трекинг | адаптер `detect.grounding_dino` |
| `adapters/segmenters/sam2.py`, `sam2_video.py` | сегментация | адаптеры `mask.sam2`, `mask.sam2_video` |
| `adapters/inpainters/lama.py`, `propainter.py` | заливка | адаптеры `erase.lama`, `erase.propainter` |
| `application/verify_quality.py` | residual-проверка | ядро адаптера `verify.residual` (обёртка + ROI-фикс) |
| `application/hole_policy.py` | геометрия ROI/кропов | стратегии `route.*` |
| `application/inpaint_runtime.py` | мёртвый код параллелизма | **реанимируем** как слой параллельной заливки (используется тестами — ломать нечего) |
| `run_cleanup.py` (`_run`, `_paint_holes`, `_segment_masks`, `_repaint_ranges`) | оркестратор | разбирается на стадии статичного пайплайна (§2), логика сохраняется |
| masks_override / static-маски | ввод маски извне | адаптер `mask.static` |
| `target.frames` (окно валидности), `where` (регион), `ordinal`, `motion` | фильтры таргета | уже покрывают «только 00:10–00:30», «правый верхний угол» — используем в UX/API как есть |

**Вывод:** скелет на 70 % существует. Не хватает: 3 портов (IntentResolver, TemporalConsistency, Verifier), реестра адаптеров с метаданными, статичного оркестратора и P0-фиксов из лога. Ничего из текущего не удаляется и не переписывается с нуля.

---

## 2. Скелет: статичный пайплайн (фундамент, который не меняется)

```
 I/O: MediaGateway (decode/encode, NVDEC/NVENC) + FrameStore          ← порт есть
 ┌───────────────────────────────────────────────────────────────────┐
 │ 1. INTENT   запрос → Intent (типизированные таргеты)              │  новый порт IntentResolver
 │ 2. DETECT   сэмплы кадров → кандидаты (боксы/маски + тип)         │  порт Detector
 │ 3. TRACK    кандидаты → треки (связка по кадрам)                  │  application/tracks (код есть)
 │ 4. MASK     треки → маски по кадрам (MaskSet)                     │  порт Segmenter
 │ 5. ROUTE    Intent+маски → план обработки (ROI/полный кадр/пропуск)│  стратегия из hole_policy
 │ 6. ERASE    кадры+маски → очищенные кадры                         │  порт Inpainter
 │ 7. TEMPORAL очищенные → согласованные по времени                  │  новый порт TemporalConsistency
 │ 8. VERIFY   результат → флаги «грязных» диапазонов → re-ERASE     │  новый порт Verifier
 │ 9. PACKAGE  encode + remux аудио + report                         │  порт MediaGateway
 └───────────────────────────────────────────────────────────────────┘
```

### Контракты стадий (фиксируем один раз)

| Стадия | Порт | Контракт (суть) | Покрывает ли оверлеи и объекты |
|---|---|---|---|
| INTENT | `IntentResolver` | `resolve(prompt, typed_targets, options) -> Intent` | да: `Target.kind` уже `watermark \| text_overlay \| object` |
| DETECT | `Detector` (есть) | `discover(images, queries, ...) -> list[Target-box]` | да: любой детектор, для объектов — свои query |
| MASK | `Segmenter` (есть) | `masks(images, tracks, ...) -> masks per frame` | да: маска-трек с нулевым смещением = оверлей, с динамикой = объект |
| ERASE | `Inpainter` (есть) | `inpaint` / `inpaint_clip` / `inpaint_masked` | да: per-frame и clip-режимы уже в сигнатурах |
| TEMPORAL | `TemporalConsistency` | `stabilize(frames, masks, cleaned) -> cleaned` | да: `none` для статики, `warp/ema` для оверлеев, `sttn` для объектов |
| VERIFY | `Verifier` | `check(frames, masks, cleaned) -> dirty ranges` | да: один контракт, разные метрики |
| ROUTE | стратегия `route.*` | `plan(intent, masks, manifest) -> HolesPlan` | да: для объектов — «всегда полный кадр» |

### Правило расширения (как добавляется новый компонент — без правки пайплайна)

1. Новый файл в `videoclean/adapters/<stage>/<name>.py`, реализующий порт стадии.
2. Запись в реестре: id, stage, factory, метаданные (`scope`, `cost`, `experimental`, зависимости моделей).
3. Опционально — включение в пресет.
4. Тест адаптера + строка в `docs/PARAMS.md`.
5. Оркестратор, остальные адаптеры, CLI-команды — **не трогаются**.

Пайплайн выдерживает и «убрать логотип в углу», и «убрать яблоко на столе» — различия целиком в наборе адаптеров и пресете. Поэтому способность к «любому объекту» не теряется из-за фокуса на оверлеях: она просто не в продуктовых пресетах.

---

## 3. Каталог адаптеров (все текущие остаются)

### 3.1 Маппинг «текущий артефакт → дом в скелете» (ничего не выбрасывается)

| Текущий артефакт | Становится | Статус |
|---|---|---|
| `adapters/segmenters/sam2_video.py` | адаптер `mask.sam2_video` | остаётся, **experimental** (xl-cost, объекты) |
| `adapters/segmenters/sam2.py` | адаптер `mask.sam2` | остаётся, дефолт для ручных масок |
| `adapters/inpainters/propainter.py` | адаптер `erase.propainter` | остаётся, **experimental** (xl-cost; лицензия NTU non-commercial — см. §10) |
| `adapters/inpainters/lama.py` | адаптер `erase.lama` | остаётся, дефолт продуктовых пресетов |
| `adapters/detectors/grounding_dino.py` | адаптер `detect.grounding_dino` | остаётся |
| `adapters/prompt/llm.py` + `adapters/llm/*` | адаптер `intent.llm` (единственный контракт LLM — **openai-compatible**; `ollama_setup.py`, `llama_cpp.py`, `llm_place` удаляются) | остаётся ядро, Ollama-слой выпиливается |
| `adapters/prompt/refine.py`, `locations.py` | ядро `intent.rules` | остаётся, становится дефолтом INTENT |
| `application/verify_quality.py` | ядро `verify.residual` | остаётся, обёртка в порт + ROI-фикс |
| `application/hole_policy.py` | стратегии `route.auto / route.roi / route.full` | остаётся |
| `application/inpaint_runtime.py` | слой параллелизма ERASE | **реанимируется** (сейчас задействован только тестами) |
| `application/profiles.py` (fast/balanced/quality) | заменяется модулем `presets.py` с продуктовыми пресетами | **удаляется без алиасов и обратной совместимости** (пользователей нет, API ломать можно) |

### 3.2 Каталог по стадиям

| Стадия | Адаптер | Происхождение | Метки | Назначение / скорость |
|---|---|---|---|---|
| INTENT | `intent.rules` | **новый** (из refine/locations + типизированные чипы) | scope=any, cost=s | дефолт; детерминированно, без LLM |
| | `intent.llm` | есть | scope=any, cost=m | свободный текст, опционально |
| DETECT | `detect.grounding_dino` | есть | scope=any, cost=s | open-vocab боксы |
| | `detect.ocr` | **новый** (RapidOCR, Apache-2.0) | scope=overlay, cost=s | текст: пиксельные маски глифов (35.7 мс/стр FP16 [И]) |
| | `detect.persistence` | **новый** (декоратор) | scope=overlay, cost=s | фильтр «оверлей vs сцена»: bbox стабилен сквозь смены сцен |
| MASK | `mask.static` | **новый** (masks_override уже поддержан) | scope=overlay, cost=s | одна маска на видео/сегмент |
| | `mask.keyframes_track` | **новый** | scope=overlay, cost=s | SAM2-image на 1–3 ключевых + LK-смещения (`_cv.py` уже есть) |
| | `mask.sam2` | есть | scope=any, cost=m | покадровый SAM2 по ROI |
| | `mask.sam2_video` | есть | **experimental**, cost=xl | полная видеопропагация (объекты, динамика) |
| | `mask.ocr_glyphs` | **новый** | scope=overlay, cost=s | маски глифов из OCR + dilate |
| ERASE | `erase.lama` | есть | scope=any, cost=s | дефолт; ROI-патчи, fp16-опция |
| | `erase.migan` | **новый** (MIT) | scope=any, cost=s | альтернатива LaMa, быстрее на мелких ROI |
| | `erase.sttn` | **новый** | scope=any, cost=m | видео-инпейнт по ROI-клипу (путь C) |
| | `erase.propainter` | есть | **experimental**, cost=xl | тяжёлая универсальная заливка (объекты) |
| | `erase.keyframe_warp` | **новый** (стратегия над Inpainter) | scope=overlay, cost=s | инпейнт 10–30 ключевых + warp/копирование остальных |
| | `erase.alpha_unblend` | **новый** (препроцессор) | scope=overlay, cost=s | «снятие» полупрозрачной накладки до заливки |
| TEMPORAL | `temporal.none` | **новый** (заглушка) | cost=s | статичные маски/статичный фон |
| | `temporal.warp_ema` | **новый** | cost=s | keyframe-warp + EMA по маске (мерцание) |
| | `temporal.sttn` | **новый** | cost=m | согласованность через clip-инпейнт |
| VERIFY | `verify.none` | **новый** (заглушка) | cost=s | пресет fast |
| | `verify.residual` | есть (обёртка) | cost=s | residual по ROI (сейчас по полному кадру — фикс) |
| | `verify.ghost_flicker` | **новый** | cost=s | ghost-гейт (градиентная энергия) + flicker (warp-error) |
| ROUTE | `route.auto` | есть (hole_policy) | cost=s | авто по признакам |
| | `route.roi` / `route.full` | есть | cost=s | явное принудительное |

Итого: **11 текущих реализаций остаются** (6 из них — в продуктовых пресетах, 2 — experimental), добавляются **12 новых адаптеров** — каждый независим и подключается без правки пайплайна.

---

## 4. P0 — устранение проблем из прод-лога (делается в первую очередь, ничего не ломая)

| # | Проблема (из лога) | Фикс | Файл | Эффект |
|---|---|---|---|---|
| 0.1 | `propagate in video` ×2 — по прогона на каждый трек (два tqdm-бара 0/672→672/672) | один прогона на все маски (OR-маска / multi-prompt) | `sam2_video.py` | −6.5 мин на каждом доп. треке |
| 0.2 | Тензор `sam2.f32` ≈ 8.4 ГБ + per-frame resize на CPU (часть 7-минутной преамбулы) | стриминг кадров в модель без дампа на диск; image_size 512 | `sam2_video.py` | −2–4 мин, −8.4 ГБ диска |
| 0.3 | Пересоздание моделей на каждую джобу (7 мин «до» propagate) | **кэш моделей между jobs** (warm-воркер) | `composition.py`, `weights_cache.py` | −1–2 мин на каждой джобе |
| 0.4 | LLM-парс при недоступном Ollama (`Connection refused`, ретраи) | дефолт `intent.rules` без LLM; LLM — с таймаутом и мгновенным fallback | `prompt/llm.py`, `config.py` | −десятки секунд, детерминизм |
| 0.5 | `nvenc=False` в контейнере — encode на CPU | образ с ffmpeg + `h264_nvenc` | Dockerfile/RunPod image | encode → секунды |
| 0.6 | `sam2._C` не собран — пост-обработка масок пропускается | собрать расширение ИЛИ перейти на transformers-бэкенд | сборка/venv | +качество масок |
| 0.7 | `raft_iter=20` (в статье ProPainter — 5) | дефолт 20→5 | `config.py`, `cli.py` | ×~4 на flow-этапе |
| 0.8 | ProPainter: двойное окно трансформера + CPU-композитинг | одно окно + композитинг на GPU | `propainter.py:325–369` | ×~2 на видео-инпейнте |
| 0.9 | Verify по полным кадрам (лишние проходы) | verify только по ROI, 1 проход | `run_cleanup.py:349–415` → `verify.residual` | −1–2 полных прохода |
| 0.10 | LaMa fp32 из-за cuFFT-ограничения | pad до степени двойки → fp16 | `lama.py:245–300` | ×1.5–2 |

**Сами по себе фиксы 0.1–0.10 переводят путь `classic-quality` с ~32 мин на ~8–12 мин на минуту 720p — до всяких новых адаптеров.** Это «починить то, что есть», и ничего из существующего при этом не выбрасывается.

---

## 5. Реестр адаптеров и схема конфига

### Реестр (`videoclean/adapters/registry.py` — новый, над существующими фабриками)

```python
@dataclass(frozen=True)
class AdapterInfo:
    id: str                 # "erase.lama"
    stage: str              # intent | detect | mask | erase | temporal | verify | route
    factory: Callable[..., Any]
    scope: str              # overlay | object | any
    cost: str               # s (<1 мин/мин) | m (1–4 мин) | xl (>4 мин)
    experimental: bool = False
    models: tuple[str, ...] = ()   # id из ModelCatalog — для проверки готовности
    note: str = ""          # лицензии, ограничения
```

- `composition.py::make_*` остаются обратно-совместимыми тонкими обёртками над реестром (CLI/сервер не ломаются).
- UI и API получают список адаптеров через реестр (для pro-раздела и документации) — один источник правды.

### Ключи конфига (по одному на стадию + пресет)

```text
preset            = overlay-fast | overlay-quality | classic-quality | objects-experimental
intent_adapter    = intent.rules | intent.llm
detect_adapters   = [detect.grounding_dino, detect.ocr, detect.persistence]   # композиция
mask_adapter      = mask.keyframes_track | mask.static | mask.ocr_glyphs | mask.sam2 | mask.sam2_video
erase_adapter     = erase.lama | erase.migan | erase.sttn | erase.propainter
erase_strategy    = per_frame | keyframe_warp          # поверх erase_adapter
temporal_adapter  = temporal.none | temporal.warp_ema | temporal.sttn
verify_adapter    = verify.none | verify.residual | verify.ghost_flicker
route_adapter     = route.auto | route.roi | route.full
```

Явные поля формы/CLI (как сейчас) побеждают пресет — поведение `profiles.py` сохраняется.

---

## 6. Пресеты (продуктовые — собираются из адаптеров)

| Пресет | INTENT | DETECT | MASK | ERASE | TEMPORAL | VERIFY | Ожидание на 60 с / 720p |
|---|---|---|---|---|---|---|---|
| **overlay-fast** (дефолт) | intent.rules | grounding_dino + persistence | mask.keyframes_track / static | erase.lama + keyframe_warp | temporal.none | verify.residual | **15–90 с** |
| **overlay-quality** | intent.rules (+llm опц.) | + detect.ocr | mask.keyframes_track / ocr_glyphs | erase.lama (+alpha_unblend) | temporal.warp_ema | verify.ghost_flicker | **90–200 с** |
| **classic-quality** (текущий путь — остаётся!) | intent.llm/rules | grounding_dino | mask.sam2_video | erase.propainter | temporal.sttn | verify.residual | ~8–12 мин после P0 (было 32) |
| **objects-experimental** | intent.rules (kind=object) | grounding_dino | mask.sam2_video | erase.propainter | temporal.sttn | verify.residual | xl; дисклеймер в UI |

- Старые `fast/balanced/quality` **удаляются полностью** — алиасов и миграций нет (пользователей 0, API можно ломать). Замена имён происходит в одной фазе с новым `presets.py`.
- Экспериментальные пресеты в UI скрыты за переключателем «Расширенные возможности», в API требуют явного `preset` (без дефолта) — честность ожиданий вместо тихого 30-минутного прогона.

---

## 7. UX/UI и API под эту архитектуру

### WebUI (одностраничный флоу «три экрана»)

1. **Загрузка** (уже есть: sources, downscale).
2. **«Что стереть»** — карточки найденных оверлеев (из DETECT-превью): тип-чип, позиция, тайм-диапазон, тумблер. Кнопка «Проверить на 3 секундах» (прогон overlay-fast на окне → слайдер до/после). Тайм-линия с полосами оверлеев. Это буквально данные стадий DETECT/TRACK — новые адаптеры делают превью точнее без правок UI.
3. **Результат** — слайдер до/после, метки verify («остатки не обнаружены», «мерцание не обнаружено»), бейдж скорости («за 47 с»), скачивание.

Настройки: один переключатель **«Быстро / Максимум качества»** = пресет; **pro-раздел** = выпадашки по стадиям из реестра (с метками «экспериментально», «медленно», «только для объектов»). Текущие 30+ параметров — в pro-раздел.

### API (контракт detect → confirm → clean; адаптеры — деталь реализации)

```http
POST /api/v1/detect            # синхронно ≤10 с: превью-кандидаты (DETECT+TRACK)
POST /api/v1/jobs              # {video_id, targets[], preset, overrides?: {stage: adapter}}
GET  /api/v1/jobs/{id}         # статус, стадия, ETA (progress-порт уже умеет)
GET  /api/v1/jobs/{id}/report  # timings/budget/verify-метрики
```

- Типизированные таргеты (`type`, `where`, `frames`, `ordinal`) — домен `Target` уже это поддерживает.
- `experimental`-адаптеры доступны только при явном `overrides` — интегратор не получит 30-минутный сюрприз по дефолту.
- Webhooks, `dry_run`, идемпотентность, SDK-примеры — как в версии 1 плана.

---

## 8. Дорожная карта (ничего не выбрасывая на каждом шаге)

| Фаза | Срок | Содержание | Состояние после |
|---|---|---|---|
| **1. Скелет + гигиена** | 1–1.5 нед | Реестр адаптеров; вынос стадий из `run_cleanup._run` в статичный пайплайн (поведение не меняется!); порты `IntentResolver`, `TemporalConsistency`, `Verifier` (обёртки над текущим); **P0-фиксы 0.1–0.10**; NVENC-образ, `sam2._C` | всё текущее живёт как адаптеры; `classic-quality` **32 мин → 8–12 мин**; добавление адаптера не требует правок пайплайна |
| **2. Оверлей-адаптеры** | 1–2 нед | `intent.rules`, `detect.ocr`, `detect.persistence`, `mask.static`, `mask.keyframes_track`, `mask.ocr_glyphs`, `erase.migan`, `erase.keyframe_warp`; пресеты overlay-fast/quality | продуктовый путь **p50 45–90 с, p95 ≤ 180 с** на минуту 720p |
| **3. Качество заливки** | 1–2 нед | `temporal.warp_ema`, `erase.alpha_unblend`, `verify.ghost_flicker`, HF-возврат заплатки | нет мерцания/призраков/«мыла» — гейты в verify |
| **4. UX + API** | 1 нед | карточки оверлеев, превью «3 секунды», pro-раздел с реестром, `/api/v1/detect`, SDK-доки | продукт «понятен за 30 секунд», интеграция «за 5 минут» |
| **5. Экспериментальный рост** | по demand | новые адаптеры под объекты (детекторы/заливки), TensorRT/compile, конвейеризация I/O, батч-параллелизм через `inpaint_runtime` | расширение функционала **без единой правки пайплайна** — именно для этого и сделан скелет |

Проверка архитектуры на каждом шаге: «появился ли новый компонент без правки оркестратора?» — если нет, чиним скелет, а не обходим его.

---

## 9. Метрики успеха

| Метрика | Цель | Измерение |
|---|---|---|
| overlay-fast / overlay-quality, 60 с / 720p | p50 ≤ 60 с / ≤ 200 с; p95 ≤ 180 с / ≤ 300 с | SLA-гейт (`check_sla_report.py`), `report.json` |
| classic-quality после P0 | ≤ 12 мин (было 32) | тот же гейт |
| Детект-превью | ≤ 10 с | тот же гейт |
| Слепое A/B «до/после» (эталонные оверлеи) | ≥ 90 % одобрений | ручной QA, 20+ роликов |
| Ghost / flicker | 0 критичных | гейты verify.ghost_flicker |
| Джобы, доведённые до конца с первого раза | ≥ 95 % (в логе — 1 из 5) | телеметрия |
| «Новый адаптер = 0 правок оркестратора» | всегда | code-review правило |

---

## 10. Риски и митигации

| Риск | Митигация |
|---|---|
| Рефакторинг ломает рабочее поведение | Фаза 1 — behavior-preserving: пресет `classic-quality` обязан давать тот же отчёт/результат до и после; прогон A/B на эталоне как гейт |
| Полупрозрачные водяные знаки — «призраки» | `erase.alpha_unblend` + `verify.ghost_flicker`; ручная зона в UI для сложных alpha |
| Лицензии (ProPainter non-commercial, STTN без LICENSE-файла) | ProPainter и STTN помечены в реестре (`note=`) и не включаются в коммерческие пресеты без выяснения; продуктовая база — Apache-2.0/MIT (LaMa, MI-GAN, SAM2, GroundingDINO, RapidOCR) |
| Пользователи ждут «любой объект» | scope-метки в UI/API, пресет `objects-experimental` с дисклеймером и честным ETA |
| Реестр превращается в свалку | правило из §8 (код-ревью), метаданные обязательны, каждый адаптер — свой тест |

---

## 11. Ответ на главный вопрос: «почему пайплайн не придётся менять»

Контракты стадий покрывают **оба** класса задач уже сейчас: `Target.kind` различает `watermark / text_overlay / object`; `Inpainter` умеет и per-frame, и clip; `Segmenter` возвращает маски-треки (нулевое смещение = оверлей, динамика = объект); `route` решает, ROI это или полный кадр. Разница между «стереть логотип» и «убрать яблоко на столе» — это **разные адаптеры и разный бюджет времени, а не другая программа**. Поэтому фокус на оверлеях ничего не стоит будущим функциям: когда понадобится расширение — это новые файлы в `adapters/` и новые строки в пресетах.

---

## 12. Ближайшая неделя (чек-лист фазы 1)

- [ ] `adapters/registry.py` + метаданные для всех 11 текущих реализаций (2 experimental).
- [ ] Порты `IntentResolver`, `TemporalConsistency`, `Verifier`; обёртки над текущим кодом.
- [ ] Разбор `run_cleanup._run` на стадии §2 (без изменения поведения; тесты как есть).
- [ ] P0-фиксы: 0.1 один прогона propagate, 0.2 без 8.4 ГБ тензора, 0.3 кэш моделей, 0.4 LLM-fallback, 0.7 raft=5, 0.8 окно/GPU-композитинг, 0.9 verify-ROI.
- [ ] Окружение: NVENC-образ, `sam2._C` (0.5–0.6).
- [ ] Прогон `classic-quality` до/после на эталонном ролике → зафиксировать цифры в `REPORT_2`, приложение B.
- [ ] Параллельно (дизайн): карточки оверлеев + схема `/api/v1/detect`.
