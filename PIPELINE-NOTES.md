# Замечания по пайплайну VideoClean

Рабочий документ для поэтапной проработки качества. Прогоны на машине делает владелец; здесь только открытый долг (обновлено после закрытия must/should волны).

Порядок работы: один участок → правки → прогон → следующий. Не смешивать этапы в одном заходе. Любая правка сверяется с **продуктовым контрактом** ниже.

---

## Продуктовый контракт (север)

**Задача пользователя:** убрать с видео названный или обведённый объект (текст, лого, вещь). Объект может быть **статическим** или **динамическим**. На выходе — скачать очищенный ролик.

**Три допустимых входа** (все first-class):

| Вход | Что делает пользователь | Что должен сделать движок |
|---|---|---|
| A. Промпт | Пишет обычным текстом, что вырезать | Parse → detect → track → segment → inpaint |
| B. Обводка | На нескольких кадрах грубо обводит | Якоря → `mask_policy=static|propagate` → segment/inpaint |
| C. Промпт + обводка | И то и другое | Обводка якорит; prompt/targets подписывают kind/query/where — **один job** |

**Два способа запуска:** по ступеням / «Запустить всё».  
**Два клиента:** WebUI и API (`POST /api/jobs`) — одни семантики.

**Стыковка с кодом (актуально):**

| Контракт | Сейчас |
|---|---|
| A | Parse → DINO → track_across_frames → sam2/sam2-video → lama/propainter |
| B | `masks` + `mask_policy`; propagate строит трек из якорей |
| C | `targets`/`prompt` + `masks` в одном job; `tracks` по-прежнему exclusive |
| Preview | всегда `sam2`, без inpaint; UI плашка честная |
| Dynamic | multi-anchor sam2-video + авто-keyframes 8–24 + профили |

---

## Карта пайплайна

```
вход: prompt | targets | tracks | masks(+targets = C)
  → LLM parse / interpret   → Intent / queries
  → Grounding DINO          → боксы на N keyframes (авто ~2/с, clamp 8–24)
  → optical-flow → CSRT/KCF → template → боксы на остальные кадры
  → select_tracks           → фильтр (+ optional relax) + interpolate_gaps
  → sam2 | sam2-video       → маски
  → lama | propainter       → заливка
  → verify 2.0 (опц.)       → residual + dirty ranges re-inpaint
  → ffmpeg encode/package
```

Превью (`kind=preview`): parse → detect → sam2 на подмножестве кадров. Без inpaint / без sam2-video.

---

## Закрыто недавно (не тащить обратно в backlog)

- **P0.1 / 3.1** Track: `track_across_frames` (flow → CSRT/KCF → template)
- **P0.2 / 3.2** `interpolate_gaps` hold first/last
- **P0.3 / 7.1** preview→full guard (nulls + full-length)
- **P0.4 / 4.3** sam2-video multi-anchor (до 8)
- **P0.5 / 5.1** TELEA удалён; дефолт LaMa
- **P0.6 / 5.6 (частично)** `mask_policy=static|propagate`
- **P0.7** вход C: targets+masks в одном job; tracks exclusive; WebUI runAll/masks шлёт оба
- **1.2** vision_batch авто-clamp ≤2 для llava (server + UI)
- **1.5** parseMode / defaulted / frames рядом с таблицей targets
- **1.6–1.8, 1.10–1.12** Parse open-vocab / refine / en-only / CJK / where / paint-colors
- **2.2** авто-keyframes от длительности (~2/с, 8–24) + UI hint ≥16–24
- **2.3** `max_box_area` дефолт 0.45; для text/watermark bump до ≥0.55
- **2.5 / 7.6** readiness из `/api/options` блокирует Найти маски / Удаление / Запустить всё + красная причина
- **2.6** en-only gate на query (уже был)
- **3.5** `select_relax` + `selectRelaxed` / «relaxed match» в preview/report + UI switch
- **4.5** плашка «Превью: sam2 покадрово, без inpaint / sam2-video»
- **4.9** каталог SAM2/SAM2.1 + UI Режим/Модель
- **6.4** WebUI `keep_workdir=true` по умолчанию
- **7.2 / 7.3** placeholder/hint sam2 (ранее)
- **7.9** битые ссылки на BUG-001 убраны
- **Стек скорости:** profiles, Verify 2.0, inpaint_workers, chunk overlap

---

## Открытый долг

### 1. Parse

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 1.1 / 1.9 | Дефолт/качество llava-phi3 | P0/P1 | Открыто по фидбеку: сменить vision-дефолт на qwen-vl |
| 1.3 | Текстовый режим без кадров слабый | P1 | Усилить system.md + post-refine |
| 1.4 | Reasoning-модели (`<think>`) | P2 | strip / blacklist |

### 2. Detect

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 2.1 | tiny vs base не сравнивали | P1 | Прогон на мелком тексте/лого |
| 2.4 | NMS 0.3 без отчёта raw hits | P2 | Тюнинг + визуальный отчёт |

### 3. Track

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 3.3 | Линейная интерполяция ≠ траектория | P1 | Flow-warp / чаще keyframes |
| 3.4 | `infer_motion` грубый | P2 | UX «motion any» по умолчанию |
| 3.6 | Дубль `videoclean/tracks.py` vs `domain/tracks.py` | P2 | Удалить legacy |

### 4. Segment

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 4.2 | sam2 покадровый → flicker масок | P1 | Temporal smooth / majority / sam2-video |
| 4.4 | sam2-video материализует весь клип в JPEG-temp | P1 | Чанки с overlap |
| 4.6 | Нет +/- точек, только кисть/боксы | P2 | Points editor |
| 4.7 | `mask_dilate` глобальный | P2 | Per-kind / per-target |
| 4.8 | Transformers vs pkg SAM2 backends | P2 | Doctor + один preferred |

### 5. Inpaint

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 5.2 | LaMa flicker на видео | P1 | Профиль quality / propainter |
| 5.3 | ProPainter resize×8 → швы | P2 | Проверить на тонких краях |
| 5.4 | ProPainter OOM на длинных | P1 | Жёстче чанки (частично есть) |
| 5.5 | Нет probe-inpaint ступени | P1 | Превью заливки на 16–48 кадрах |

### 6. Verify + encode

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 6.1 | Один pass по умолчанию | P2 | Уже есть `verify_max_passes`; UX stop-when-clean |
| 6.2 | Verify на том же detect+track | P1 | Следить после Track-улучшений |
| 6.3 | `verify_max_coverage` без why-skipped в UI | P2 | Логировать в report/UI |
| 6.5 | Encode обычно ок | P3 | Не трогать |

### 7. Server / WebUI / UX

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 7.4 | Advanced knobs спрятаны | P1 | Часть уже в ступени Маски; добить UX |
| 7.5 | Нет side-by-side слоёв full run | P1 | Viewer scrub по artifacts |
| 7.7 | Run-all vs per-stage | P2 | Full-length guard уже есть |
| 7.8 | Config ↔ workspace knobs | P2 | Profiles уже есть; добить sync |
| 7.10 | Дубли `/api/jobs` + `/api/preview` | P3 | Схлопнуть позже |

### 8. Архитектура / долг

- Два `tracks.py` (legacy + domain).
- `sam2-video` / ProPainter — полная материализация клипа.
- Нет first-class CLI `segment`/`inpaint`-only (overrides есть, DX сырой).

---

## Как гонять участки

| Участок | Что смотреть | Как гонять |
|---|---|---|
| Parse | targets + строка parseMode в UI | interpret / preview parse |
| Detect | оверлеи боксов, keyframes авто/ручные | preview detect |
| Track | `tracks.json`, non-null coverage | full-length detect |
| Segment | mask overlays / meanMaskCoverage | preview sam2 или full sam2-video |
| Inpaint | ролик / workdir | full run, keep_workdir on |
| Verify | `verifyPasses` / `verifyNote` | profile balanced/quality |

Правило: пока Track даёт дыры, Segment/Inpaint не оценивать как «модель плохая».

---

## Сделано: скорость + качество (стек)

| Кусок | Что |
|---|---|
| Track | Optical-flow → CSRT/KCF → template |
| Segment | sam2-video multi-anchor (≤8) |
| Verify 2.0 | residual + dirty ranges re-inpaint |
| Inpaint | workers + chunk overlap |
| Profiles | fast / balanced / quality |
| Entry C | targets+masks one job |
| Detect defaults | auto keyframes, max_box_area 0.45/+text |
| UX honesty | readiness→Run, preview badge, parse meta, keep_workdir |

Ключевые файлы: `application/profiles.py`, `frames_sample.py`, `verify_quality.py`, `inpaint_runtime.py`; `adapters/detectors/_cv.py`, `grounding_dino.py`; `adapters/segmenters/sam2_video.py`; `use_cases/run_cleanup.py`; `server/fastapi_app.py`; WebUI `stage-rail.tsx`, `params.ts`, `inpaint-run`.
