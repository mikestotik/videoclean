# Замечания по пайплайну VideoClean

Рабочий документ для поэтапной проработки качества. Прогоны на машине делает владелец; здесь только разбор по коду и докам (обновлено под продуктовый контракт).

Порядок работы: один участок → правки → прогон → следующий. Не смешивать этапы в одном заходе. Любая правка сверяется с **продуктовым контрактом** ниже: если участок не обслуживает один из трёх входов (промпт / обводка / оба) или ломает WebUI↔API паритет — это дефект, не «потом».

---

## Продуктовый контракт (север)

**Задача пользователя:** убрать с видео названный или обведённый объект (текст, лого, вещь). Объект может быть **статическим** или **динамическим**. На выходе — скачать очищенный ролик.

**Три допустимых входа** (все first-class, не «костыль»):

| Вход | Что делает пользователь | Что должен сделать движок |
|---|---|---|
| A. Промпт | Пишет обычным текстом, что вырезать | Parse → detect → track → segment → inpaint |
| B. Обводка | На нескольких кадрах грубо обводит «фломастером» (криво/косо ок) | Из штрихов понять цель, протянуть по клипу (static и dynamic), segment/inpaint |
| C. Промпт + обводка | И то и другое | Обводка якорит «что именно»; промпт уточняет kind/имя/где; не игнорировать ни одно |

**Два способа запуска:**

- по ступеням (parse / detect / inpaint отдельно) — отладка и контроль;
- одной кнопкой «всё сразу» — счастливый путь.

**Два клиента с одним контрактом:**

- WebUI (ручной режим, видимые ступени, превью масок, скачивание);
- API для других сервисов (`POST /api/jobs` и связанные source/mask эндпоинты) — те же входы и те же семантики, без скрытой логики «только для UI».

**UX / DX, которых держимся:**

- грубая обводка достаточна: не требовать идеальный контур;
- после каждой ступени видно, что получилось (боксы / маски / результат), можно править и перезапустить только хвост;
- ошибка говорит *какая* ступень и *что* поправить (targets, keyframes, device, missing weights);
- API: предсказуемые поля (`prompt`, `targets`, `tracks`, `masks` / annotations), стабильные артефакты job, документированный happy path для интеграции;
- static и dynamic — оба рабочих режима, не «dynamic потом».

**Как это стыкуется с текущим кодом (дыры относительно контракта):**

| Контракт | Сейчас | Разрыв |
|---|---|---|
| A. Промпт | Есть (LLM parse → DINO → …) | Queries схлопываются (см. §1.A); track/segment слабые на динамике |
| B. Обводка | `masks_override` / source masks; interpret-job по штрихам | Маска с кадра **копируется на все кадры** — ок для static logo, **ломает dynamic**. Нет нормальной пропагации обводки → track/sam2-video |
| C. Промпт + обводка | Частично через interpret (`kind=prompt` + annotations) или ручные targets | Взаимоисключение `targets`/`tracks`/`masks` в API мешает «и якоря и queries» в одном run без обходных путей |
| Ступени / всё сразу | Stage rail + Run all | Preview-треки ≠ full-length (P0.3); inpaint-by-tracks после sparse detect врёт |
| WebUI + API | Общий `/api/jobs` | DX API ок по форме; семантика masks/dynamic и длина tracks должны быть явными в доке и валидации |
| Dynamic | sam2-video + больше keyframes (ручной тюнинг) | Дефолтный путь заточен под static; product требует dynamic first-class |

**Следствие для порядка работ:** Track + пропагация с пользовательских масок/якорей — не «улучшение детекта», а выполнение контракта B/C и динамики. Parse-ослабление queries — опора контракта A. UI/API guards по длине tracks и честные режимы masks — DX контракта.

Рекомендуемый порядок участков (с учётом контракта):

1. Parse (LLM → queries/targets) — вход A, часть C
2. Detect (Grounding DINO → боксы на keyframes)
3. Track + пропагация якорей (шаблон / interpolate / **обводка→клип**) — B/C + dynamic
4. Segment (sam2 / sam2-video → маски)
5. Inpaint (lama / propainter)
6. Verify + encode
7. Server / WebUI / API DX (ступени, валидации, паритет)

---

## Карта пайплайна (как есть)

```
вход: prompt | targets | tracks | masks(annotations)
  → LLM parse / interpret   → Intent / queries   (если не manual)
  → Grounding DINO          → боксы на N keyframes (дефолт 8)
  → template match          → боксы на остальные кадры (часто None)
  → select_tracks           → фильтр + interpolate_gaps
  → sam2 | sam2-video       → маски
  → lama | propainter → заливка
  → verify (опц.)           → re-detect + один re-inpaint
  → ffmpeg encode/package
```

Превью (`kind=preview`): только parse → detect → sam2 на подмножестве кадров. Inpaint и `sam2-video` в превью не участвуют (`server/service.py` форсит `segmenter=sam2`).

---

## P0. Корневые причины «всё сыро»

### P0.1 Трекинг между keyframes слабый

Файл: `videoclean/adapters/detectors/_cv.py` (`match_template`), вызов в `grounding_dino.py`.

Детектор реально смотрит ~8 кадров. Остальное: grayscale `cv2.matchTemplate` по кропу. На движущемся фоне / камере / аэросъёмке шаблон цепляется за фон или даёт `None`. Пустой бокс → пустая маска → объект остаётся кусками. Это уже зафиксировано в `docs/PARAMS.md`.

Что делать дальше (участок Track):

- заменить template match на нормальный трекер (CSRT / KCF как минимум; лучше optical-flow / ByteTrack / SAM2 video memory);
- или поднимать `detector_keyframes` до плотной сетки (дорого, но честно);
- или для динамики сразу требовать `sam2-video` и кормить его несколькими якорями, не одним боксом.

### P0.2 `interpolate_gaps` в domain обрезан

**Сделано:** domain `interpolate_gaps` снова hold first/last (и single-anchor hold на весь клип).

### P0.3 Preview-треки нельзя честно кормить в полный inpaint

**Сделано (прототип):** `parseTracks` хранит `null`; «По трекам» только при full-length; бэкенд отвергает короткие `tracks_override`; sparse = только осмотр. Params detect реально уходят в preview.

### P0.4 `sam2-video` якорится только на первый бокс трека

Файл: `videoclean/adapters/segmenters/sam2_video.py`, `_first_box`.

Пропагация стартует с одного кадра. Поздние keyframe-боксы детектора игнорируются. Объект уехал, сменил размер, пропал/вернулся: маска плывёт или теряется. Для динамики это главный потолок сегментера.

Что делать (участок Segment): multi-frame prompts (добавлять бокс на нескольких keyframes), re-anchor при drift, опционально points + negative points.

### P0.5 Дефолтный инпейнтер

**Сделано:** OpenCV TELEA убран из продукта. Дефолт — LaMa (CPU/CUDA). ProPainter — путь качества на CUDA (vendor + 3 `.pth`).

LaMa покадрово может давать flicker; на CUDA для финала — профиль «Качество» / propainter.

### P0.6 Обводка не протягивается по клипу (ломает вход B и dynamic)

**Сделано (прототип):** явный `mask_policy=static|propagate` (UI: Держать / Протянуть). Static = tile как раньше. Propagate = bbox с кадров-якорей → interpolate/hold → segmenter. Multi-anchor sam2-video ещё впереди (P0.4).

### P0.7 Промпт + обводка как один вход (C)

API сейчас часто взаимоисключает `targets` / `tracks` / `masks`. Для входа C нужно: annotations как якоря + prompt/targets как подпись «что это», один job. Иначе WebUI и интеграторы вынуждены двумя прогонами или interpret-only обходом.

---

## 1. Parse (LLM → targets)

Файлы: `videoclean/adapters/prompt/llm.py`, `refine.py`, `prompts/*.md`, `docs/MODELS.md`.

Слой называется prompt-parser / intent refine (не отдельный «enricher»-модуль): user prompt → LLM JSON targets → `refine_intent` → `intent.queries` → Grounding DINO caption.

### 1.A Что реально ограничивает query сейчас

Жёсткого лимита «16 символов / 16 токенов» в текущем коде **нет**. Это наследие удалённого OWL-ViT: в плане `docs/superpowers/plans/2026-09-08-remove-owlvit.md` у адаптера был `OWL_VIT_MAX_QUERY_TOKENS` и `fit_owlvit_queries` / `split_visual_phrases`. OWL-ViT вырезан; Grounding DINO кормится caption-строкой целиком (`grounding_dino.py` → `_detect_caption`, к caption только добавляется точка).

Что **всё ещё** сужает queries под старую привычку «коротко, как для OWL»:

| Где | Что делает | Жёсткость |
|---|---|---|
| `prompts/system.md`, `vision_system.md`, `bridge_system.md`, `interpret_system.md` | В контракте везде: `query: short English visual name`; vision прямо толкает к TYPE: `"text"`, `"caption"`, `"logo"` | Мягкая (LLM слушается промпта) |
| `llm.py` `_query_ok` | Отбрасывает query: длина `< 3`, `BAD_QUERIES` (`the`, `a`, `overlay`, `object`…), любой non-ASCII (кириллица), символ `\|` | Жёсткая (target выкидывается) |
| `refine.py` `_rewrite_query` | Если `kind=text_overlay` и в query нет слова из `_TEXT_QUERIES` → **принудительно `"text"`**. Если `kind=watermark` и нет слова из `_MARK_QUERIES` → **принудительно `"logo"`**. Длинные нормальные фразы (`red youtube subscribe button`) при kind=object не трогает | Жёсткая перепись |
| `refine.py` `refine_intent` | Подменяет `where`/ordinal из user prompt; может сменить kind object→text_overlay по whitelist `_OVERLAY_WORDS` | Жёсткая эвристика |
| Grounding DINO adapter | Лимита длины caption нет; threshold/NMS/max_box_area режут боксы, не текст query | — |

Итог: детектор уже умеет многословные английские фразы (это прямо отмечено в `docs/MODELS.md` для tiny). Узкое место — system-промпты («short» + TYPE-слова) и `_rewrite_query`, который схлопывает text/watermark к `"text"`/`"logo"`. Число `16` в коде сейчас встречается у preview frame count и regex-окон `where`, не у длины query.

### 1.B Что убрать / ослабить на участке Parse

1. В system/vision/bridge/interpret: заменить `short English visual name` на формулировку в духе «English open-vocab phrase for Grounding DINO, 1–8 words OK; prefer concrete (`red channel logo`, `bottom news ticker`), not only `logo`/`text`».
2. Убрать или сузить `_rewrite_query`: не затирать осмысленную фразу до `"text"`/`"logo"`, если в ней уже есть английские content-слова.
3. Оставить `_query_ok` en-only + anti-junk (кириллица в DINO бесполезна; `BAD_QUERIES` нужны).
4. Не возвращать OWL-лимиты токенов.

Проверка после правки: один и тот же ролик с ручным query `red subscribe button top right` vs автопарс; автопарс не должен схлопываться в голое `logo`, если модель уже выдала нормальную фразу.

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 1.1 | `llava-phi3` ломает контракт: `kind: object` вместо `text_overlay`, OCR-осколки в query (`any`, `wrong.side`) | P0 | Сменить дефолт vision-модели на qwen-vl; жёстче валидировать query |
| 1.2 | Vision-batch > 2 на llava даёт мусор | P1 | В UI/doctor блокировать или авто-clamp по модели |
| 1.3 | Текстовый режим без кадров слабый: «remove all text» не становится query `text` | P1 | Усилить system.md + post-refine |
| 1.4 | Reasoning-модели (deepseek-r1 и т.п.) заворачивают JSON в рассуждения | P2 | Чёрный список / strip-reasoning |
| 1.5 | Нет стабильного «что увидел LLM» в UI кроме таблицы таргетов | P2 | Показывать raw parseMode, defaulted, visionFrameIndices |
| 1.6 | System-промпты требуют `short` / TYPE-слова (`text`/`logo`) — наследие OWL-мышления при живом Grounding DINO | P1 | **Сделано:** prompts → `English open-vocab phrase` (1–8 words OK) |
| 1.7 | `_rewrite_query` затирает фразы text_overlay/watermark до `"text"`/`"logo"` | P1 | **Сделано:** схлопывает только junk OCR; concrete phrases сохраняются |
| 1.8 | Жёсткого лимита 16 символов/токенов нет (OWL удалён); путаница с preview `count=16` и regex `.{0,16}` в where | — | Документировано; в коде длины query не резать |
| 1.9 | Дефолт/качество llava-phi3 (1.1–1.2) | P0/P1 | **Открыто** — отдельно после твоего фидбека по queries |
| 1.10 | Interpret при пустом промпте: VLM (gemma и др.) подставляет китайский `prompt` | P0 | **Сделано:** interpret → English pipeline prompt; CJK без CJK у пользователя чинится из targets |
| 1.11 | `where` из закраски угадывал VLM (top-right → top/center), геометрии маски не было | P0 | **Сделано:** centoid/CC маски → where; геометрия перекрывает VLM |
| 1.12 | VLM описывает объект как `red logo` из‑за красного tint аннотации | P0 | **Сделано:** запрет в interpret + strip paint-colors из prompt/query |

### Статус участка Parse (частичный)

Сделано в коде: 1.6, 1.7 (+ тесты `tests/test_refine.py`). Юнит-тесты зелёные. Нужен твой прогон на реальном LLM (чеклист ниже).

Критерий готовности участка: на 5–10 своих роликов ручные targets и LLM-targets дают близкий набор queries; мусорные OCR-queries отсекаются; осмысленные многословные en-фразы доходят до DINO без схлопывания.

### Чеклист прогона Parse (для фидбека)

Перезапусти backend/webui после pull правок (`make dev` или как обычно), иначе подтянутся старые `.md` из памяти процесса.

1. **Только промпт (вход A)**  
   Workspace → видео → ступень промпта / interpret или preview «разбор».  
   Примеры: «убери красный логотип в углу», «удали нижние титры», «remove the subscribe button».  
   Смотри таблицу targets: `query` должны быть **конкретные en-фразы**, не обязательно голое `text`/`logo`.  
   В job: `analysis/prompt.json` или report preview — поле `targets[].query`.

2. **Сравнение до/после ожидания**  
   Если модель всё же вернула `wrong` / `side` — ок, refine схлопнет в `text`/`logo`.  
   Если модель вернула `breaking news ticker` / `red channel logo` — фраза **должна остаться** как есть.

3. **Ручные targets**  
   Впиши сам `red subscribe button` → detect: боксы должны искать по этой фразе (не по `text`).

4. **Кириллица в query**  
   Если LLM вдруг вернул русское слово в query — target должен отфильтроваться (`_query_ok`); в таблице его не будет / парсер уйдёт в ошибку или другие targets.

5. **Фидбек мне**  
   Пришли 2–3 примера: user prompt → итоговые targets (kind, query, where) + какая LLM. Отметь, где фраза хорошая, где всё ещё мусор или слишком общее.

---

## 2. Detect (Grounding DINO)

Файл: `videoclean/adapters/detectors/grounding_dino.py`.

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 2.1 | Только tiny в проверенном дефолте; base/swint не гоняли | P1 | Сравнить tiny vs base на мелком тексте/лого |
| 2.2 | Дефолт keyframes=8 на длинном клипе = огромные дыры | P0 | Дефолт от fps/длительности или UI-подсказка «для динамики ≥ 16–24» |
| 2.3 | `max_box_area=0.25` режет широкие титры на всю ширину | P1 | Отдельный пресет для captions / поднять лимит |
| 2.4 | NMS 0.3: соседние буквы/иконки могут схлопнуться или наоборот размножиться | P2 | Тюнинг + визуальный отчёт по raw hits |
| 2.5 | При `status != ready` детектор тихо скипается в цепочке | P1 | В UI явный красный статус до запуска |
| 2.6 | Open-vocab по-английски: русские queries из LLM почти бесполезны (парсер обязан переводить; если не перевёл — пусто) | P1 | Жёсткий en-only gate на query |

Критерий: на keyframes боксы стабильно покрывают целевой объект; false positives видны в оверлее и правятся через targets/threshold.

---

## 3. Track

Файлы: `_cv.match_template`, `domain/tracks.interpolate_gaps`, `application/select.py`.

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 3.1 | Template match — главный источник дыр на динамике | P0 | См. P0.1 |
| 3.2 | Domain `interpolate_gaps` без hold на концах | P0 | См. P0.2 |
| 3.3 | Линейная интерполяция бокса ≠ траектория объекта при ускорении/повороте | P1 | Flow-based warp бокса или частые keyframes |
| 3.4 | `infer_motion` по std центров (порог 0.025) грубый | P2 | Нужен только для фильтра intent.motion; упростить UX «motion any» по умолчанию |
| 3.5 | `select_tracks(relax=True)` снимает where/ordinal и может взять чужой объект | P1 | В превью показывать «relaxed match»; в full run опция strict |
| 3.6 | Дубль `videoclean/tracks.py` vs `domain/tracks.py` | P2 | Удалить legacy после сверки поведения |

Критерий: `tracks.json` почти без `null` на целевом объекте; coverage трека высокий на всём окне появления.

---

## 4. Segment (маски)

Файлы: `adapters/segmenters/sam2.py`, `sam2_video.py`.

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 4.1 | Качество маски = качество бокса. Плохой track не спасает SAM | P0 | Сначала Track, потом Segment |
| 4.2 | `sam2` покадровый: края масок мерцают кадр к кадру → flicker после inpaint | P1 | Временное сглаживание масок / majority vote / sam2-video |
| 4.3 | `sam2-video`: один якорь на трек (P0.4) | P0 | Multi-frame box prompts |
| 4.4 | `sam2-video` пишет весь клип в tempfile JPEG и гоняет целиком: RAM/диск/время | P1 | Нативный путь без полной материализации или чанки с overlap |
| 4.5 | Превью всегда `sam2`: выбрать `sam2-video` в форме и судить по превью нельзя | P1 | Либо короткий video-preview на выбранном окне, либо явная плашка «в превью всегда sam2» |
| 4.6 | Только box-prompt: нет клика «плюс/минус» точками в UI для доводки маски | P2 | Editor: positive/negative points → segmenter |
| 4.7 | `mask_dilate` глобальный: тонкий текст vs толстый лого combо ломается одним числом | P2 | Dilate per-target / per-kind |
| 4.8 | Transformers `Sam2Model` vs pkg `SAM2ImagePredictor` — два бэкенда, разное поведение | P2 | Doctor показывает активный backend; зафиксировать один preferred |
| 4.9 | В каталоге были только sam2 tiny/large; UI выбирал драйвер, модель — свободный текст | P1 | **Сделано:** SAM2+SAM2.1 × tiny/small/base+/large в каталоге; в форме «Режим» + «Модель» |

Критерий: mean mask coverage адекватен; визуально маска кроет объект без огромного halo и без дыр внутри букв/лого; на динамике нет кадров с пустой маской при видимом объекте.

---

## 5. Inpaint (заливка)

Файлы: `lama.py`, `propainter.py`.

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 5.1 | ~~Telea как дефолт~~ | — | **Сделано:** TELEA удалён; дефолт lama |
| 5.2 | LaMa покадровая → temporal flicker на видео | P1 | Для видео propainter / профиль quality |
| 5.3 | ProPainter: resize к кратности 8, потом upsample — мягкость/швы | P2 | Проверить артефакты на тонких краях |
| 5.4 | ProPainter жрёт клип в память; длинные ролики = OOM | P1 | Жёсткие чанки с overlap + blend (частично есть `subvideo_length`) |
| 5.5 | Превью не показывает качество заливки | P1 | Отдельная ступень «probe inpaint» на 16–48 кадрах окна |
| 5.6 | `masks_override` копирует одну маску на все кадры | P0 для динамики | Для движения запретить / требовать per-frame masks или tracks |

Критерий: на динамическом фоне нет residual ghost и сильного flicker; статичный лого исчезает без каши.

---

## 6. Verify + encode

Файл: `run_cleanup.py` (блок verify).

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 6.1 | Один pass re-detect + re-inpaint | P2 | Опционально 2 pass / stop when clean |
| 6.2 | Verify гоняет тот же слабый detect+track по cleaned кадрам | P1 | После починки Track verify станет осмысленнее; иначе false leftover |
| 6.3 | Порог `verify_max_coverage` может отсечь полезный re-inpaint | P2 | Логировать why skipped в report и UI |
| 6.4 | При `keep_workdir=0` кадры/маски сносятся: отладка после факта невозможна | P1 | В WebUI дефолт keep_workdir=1 на время проработки качества |
| 6.5 | Encode/package обычно ок; жалобы на качество почти никогда не отсюда | P3 | Не трогать, пока не стабильны маски/inpaint |

---

## 7. Server / WebUI / UX

| # | Замечание | Серьёзность | Направление |
|---|---|---|---|
| 7.1 | Стык preview tracks → full inpaint сломан (P0.3) | P0 | **Сделано (прототип):** nulls + full-length guard UI/API |
| 7.2 | Placeholder сегментера в UI: `sam2-video`, дефолт движка `sam2` | P3 | Placeholder = `sam2` |
| 7.3 | Нет пояснения разницы sam2 / sam2-video рядом с селектом | P2 | Hint 1 строка + ссылка на MODELS |
| 7.4 | Advanced-параметры (keyframes, tracker thresholds, dilate) спрятаны; для качества они главные | P1 | Вынести «качество масок» в видимый блок ступени Detect/Segment |
| 7.5 | Нет side-by-side: кадр / боксы / маска / inpaint в одной шкале времени для full run | P1 | Viewer слои + scrub по job artifacts |
| 7.6 | Статус readiness моделей (doctor) слабо связан с формой запуска | P1 | Disable Run + причина, если segmenter/inpainter unavailable |
| 7.7 | «Запустить всё» смешивает ступени; при отладке нужен жёсткий per-stage | P2 | Ок как есть, но блокировать inpaint-tracks без full-length tracks |
| 7.8 | Конфиг моделей vs workspace: скачал на Config, а в форме старые knobs | P2 | Профили fast/balanced/quality + готовность lama/propainter |
| 7.9 | `docs/BUG-001-removal-quality.md` указан в MODELS/AGENTS, файла нет | P3 | Восстановить или убрать ссылки |
| 7.10 | Дублирующие API: `/api/jobs` и старые `/api/preview` | P3 | Со временем схлопнуть, не блокер качества |

---

## 8. Архитектура / долг (не срочно для качества кадра)

- Два `tracks.py` (legacy + domain).
- `LazyFrames` ок для покадрового sam2; для `sam2-video` и ProPainter всё равно полная материализация.
- Полный extract всех кадров на диск до любой работы: итерации медленные; для отладки участка Detect нужен subset-path как в preview (частично есть).
- Нет first-class CLI «только segment из tracks.json» / «только inpaint из masks/» без полного job orchestration (overrides есть, но DX сырой).

---

## Как гонять участки (без смешивания)

| Участок | Что смотреть | Как гонять у себя |
|---|---|---|
| Parse | `analysis/prompt.json`, таблица targets | preview mode=parse, править targets руками |
| Detect | оверлеи боксов, `tracks_raw.json` | preview mode=detect, крутить threshold/keyframes |
| Track | `tracks.json`, доля non-null boxes | те же артефакты + покрытие по кадрам |
| Segment | `*_mask.jpg`, meanMaskCoverage, masks/ | preview (sam2) или full с `tracks_override` |
| Inpaint | выходной ролик / inpainted/ | full run с готовыми tracks или masks, keep_workdir=1 |
| Verify | report.verifyPasses, leftover | full run с verify on/off |

Правило: пока Track даёт дыры, Segment/Inpaint не оценивать как «модель плохая».

---

## Предлагаемый первый заход

Участок **Track** (P0.1 + P0.2 + P0.3): без него динамика не взлетит, какой бы сегментер ни стоял.

После Track: **Segment** (multi-anchor sam2-video) → **Inpaint defaults/UX** → **Parse model defaults**.

Когда скажешь «начинаем участок N», разбираем только его: гипотезы, конкретный дизайн правки, список файлов. Код и прогоны — после твоего ок на дизайн участка.

---

## Сделано: скорость + качество (стек)

| Кусок | Что |
|---|---|
| Track | Optical-flow → CSRT/KCF → template (`track_across_frames`) |
| Segment | `sam2-video` multi-anchor (до 8 боксов на трек) |
| Verify 2.0 | residual unchanged + re-detect → grow mask → **локальный** re-inpaint ranges; `verify_max_passes` |
| Inpaint | framewise `inpaint_workers` (auto на CPU); video chunked + overlap blend |
| Profiles | `fast` / `balanced` / `quality` (+ UI селект, CLI `--profile`) |

Ключевые файлы: `application/profiles.py`, `verify_quality.py`, `inpaint_runtime.py`; `adapters/detectors/_cv.py`; `adapters/segmenters/sam2_video.py`; `use_cases/run_cleanup.py`; WebUI `params.ts` + `InpaintControls`.

### Чеклист прогона

1. Перезапуск `make dev` (бэкенд подхватит новые поля config).
2. В «Удаление» выбрать профиль **Баланс** (CPU) или **Качество** (CUDA).
3. Короткий клип: Маски → Удаление; в report смотреть `verifyPasses`, `verifyNote`, `inpaintWorkers`, `profile`.
4. На CPU с LaMa: Advanced → «Потоки инпейнта» = 0 (auto) vs 4 — сравнить wall-time.
5. На CUDA: профиль Качество → sam2-video + ProPainter; убедиться что leftover-pass не гоняет весь клип зря (ranges в verifyNote).
6. Регресс: профиль **Быстро** (lama, verify off) — быстрый smoke; без `big-lama.pt` джоба должна явно сказать скачать веса.
