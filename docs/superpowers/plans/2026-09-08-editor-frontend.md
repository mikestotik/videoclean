# Editor UI — План B: Фронтенд (webui)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Редактор поверх бэкенда Плана A: загрузка видео → сразу редактор (нативный `<video>`, скраб, рисование масок), ступени Интерпретация → Таргеты → Detect → Inpaint → Результат, before/after, пресеты, авто-режим «Запустить всё».

**Architecture:** FSD, слои снизу вверх: entities (source, annotation, preset, job) → features (annotate, interpret, detect-run, inpaint-run) → widgets (library, editor-viewer, timeline, stage-rail) → pages/workspace. Существующие viewer/filmstrip/preview-grid/jobs-sidebar/target-table/run-form заменяются и удаляются. UI только из shadcn-компонентов, vanilla fetch.

**Tech Stack:** React 19, TypeScript strict, Vite 8, Tailwind v4 (CSS-first), shadcn (base-mira, Base UI), lucide-react. Без новых runtime-зависимостей кроме `@fontsource/jetbrains-mono`.

**Spec:** `docs/superpowers/specs/2026-09-08-editor-pipeline-design.md` §7–9 — читать вместе с планом.

**Worktree:** тот же `/Users/mikestotik/Projects/videoclean/.worktrees/editor-backend` (ветка feat/editor-backend, бэкенд уже в ней).

## Global Constraints

- FSD: импорт только «вниз»; публичный API слайса — `index.ts`; между слайсами одного слоя — нельзя.
- TS strict, функции-компоненты, vanilla fetch (`api` из `@/shared/api/client`), никаких axios/react-query.
- UI-тексты — на русском, идентификаторы — en. Без эмодзи.
- UI только из shadcn-компонентов (`bunx shadcn@latest add <name>` из `webui/`), Tailwind utilities для layout. Кастомные стили — только токены в index.css.
- Верификация каждой задачи: `bun run lint && bun run typecheck` в `webui/`; полный билд `bun run build` на задачах-мильштонах (T1, T8, T11, T12) и в конце.
- Коммит после каждой задачи: `feat(ui): ...` / `refactor(ui): ...`.
- Дизайн-токены (спека §7.3): bg `#141518`, panel `#1B1D21`, panel-2 `#22252A`, text `#E8E9EB`, muted `#9A9DA3`, hairline `#2A2D33`, accent amber `#E8A33D`, маски/деструктив `#E5484D`, ok `#4CAF7D`. Inter Variable для UI; JetBrains Mono только таймкод/числа. Sentence case, без all-caps eyebrow. Тёмная тема — единственная.
- Моушн: только на действия пользователя; `prefers-reduced-motion` уважается; focus-visible везде.
- Бэкенд-контракты (План A, уже на ветке): `POST /api/sources` (multipart video) → `{id, name, createdAt, probe:{fps,duration_s,width,height,frame_count,has_audio}, video_url, annotations_url}`; `GET /api/sources` → list; `DELETE /api/sources/{id}` (409 при активных job'ах); `GET /api/sources/{id}/video`; `GET /api/sources/{id}/frames/{n}(.jpg)`; `PUT /api/sources/{id}/masks/{n}` (multipart: mask PNG + strokes JSON) → 201 `{ok, frame}`; `GET .../masks/{n}` → PNG; `DELETE .../masks/{n}`; `GET /api/sources/{id}/annotations` → `{frames:[{frame,url,strokes,updatedAt}]}`; `POST /api/jobs` multipart (kind=run|preview|prompt, source_id, prompt, targets/tracks/masks JSON, mode, start/count/stride/indices/all, llm_base_url/llm_api_key, keep_workdir, min_mask_coverage, verify_max_coverage, formats, + все конфиг-поля) → 201 job_dict; job_dict несёт `kind` (run|preview|prompt), `source_id`, `source_name`, `stage`, `fraction`, `detail`, `eta`, `error`; `/api/poll` → `{jobs, models, downloads, doctor, options, ollama, device}`; `/api/presets` CRUD (`{id,name,payload,createdAt}`); preview-артефакты `/api/jobs/{id}/preview/{name}` + `preview.json` (frames[].{index,maskCoverage,boxes}, tracks в отчёте).
- Preview-отчёт job'а: `GET /api/jobs/{id}` → job_dict; отчёт целиком лежит в `report_json`, но API отдаёт только job_dict — манифест detect'а берём из `preview/preview.json` (как сейчас), а `tracks` для tracks_override — тоже из preview.json (поле `tracks` уже пишется туда RunPreview).

---

### Task 1: Токены редактора и каркас App

**Files:**
- Modify: `webui/src/index.css`
- Modify: `webui/src/app/App.tsx`
- Modify: `webui/src/main.tsx`
- Create: `webui/src/shared/ui/timecode.tsx` (маленький моно-компонент)

**Interfaces:**
- Produces: CSS-переменные `--background/--foreground/--card/--muted/--accent…` перекрашены в графит+амбер (shadcn-семантика сохранена — все существующие компоненты продолжают работать); `Timecode` компонент (`children: string`, моно, tabular-nums).
- Код-скелет App: `h-svh flex flex-col` без скролла страницы; header (лого + табы Редактор/Конфиг + кнопка темы не нужна); `<main className="flex-1 min-h-0">`. WorkspacePage и ConfigPage пока остаются прежними (их реворк в T12).

- [ ] **Step 1: Токены.** В `index.css`: заменить значения `:root` на графитовую палитру (oklch-конверсии: `#141518` ≈ `oklch(0.20 0.005 260)`, `#1B1D21` ≈ `oklch(0.235 0.006 260)`, `#22252A` ≈ `oklch(0.27 0.007 260)`, `#E8E9EB` ≈ `oklch(0.93 0.003 260)`, `#9A9DA3` ≈ `oklch(0.68 0.008 260)`, `#2A2D33` ≈ `oklch(0.30 0.008 260)`, `#E8A33D` ≈ `oklch(0.75 0.13 75)`, `#E5484D` ≈ `oklch(0.62 0.19 25)`, `#4CAF7D` ≈ `oklch(0.68 0.12 155)`). primary=amber, destructive=красный, ring=amber/50, border=hairline, card=panel, popover=panel-2. Убрать переключение тем из main.tsx (ThemeProvider можно оставить, но html получает класс dark всегда — проще: значения прямо в `:root`, `.dark` блок удалить, ThemeProvider выкинуть из main.tsx). Добавить `@import "@fontsource/jetbrains-mono";` (пакет добавить: `bun add @fontsource/jetbrains-mono`) и `--font-mono: 'JetBrains Mono Variable', monospace` в @theme.
- [ ] **Step 2: Timecode.**

```tsx
// webui/src/shared/ui/timecode.tsx
import { cn } from "@/shared/lib/utils"

export function Timecode({ children, className }: { children: string; className?: string }) {
  return <span className={cn("font-mono text-sm tabular-nums tracking-tight", className)}>{children}</span>
}
```

- [ ] **Step 3: App.tsx** — заменить контейнер на `h-svh overflow-hidden flex flex-col`, header сжать (`border-b border-border px-4 py-2`), main `flex-1 min-h-0`. Текст tab'ов: «Редактор», «Конфиг».
- [ ] **Step 4:** `cd webui && bun add @fontsource/jetbrains-mono && bun run lint && bun run typecheck && bun run build`.
- [ ] **Step 5: Commit** `feat(ui): editor dark tokens, mono timecode, app shell`.

---

### Task 2: shared/hooks + новые shadcn-компоненты

**Files:**
- Create: `webui/src/shared/hooks/usePoll.ts`
- Create: `webui/src/shared/lib/format.ts`
- Add (bunx shadcn): `slider`, `tooltip`, `toggle-group`, `popover`, `switch` в `webui/src/shared/ui/`

**Interfaces:**
- Produces:

```ts
// usePoll: интервал-поллинг с очисткой; fn вызывается сразу и затем каждые ms; paused=true останавливает
export function usePoll(fn: () => void | Promise<void>, ms: number, paused?: boolean): void
```

```ts
// format.ts — чистые функции
export function formatTimecode(frame: number, fps: number): string  // "00:00:12:04" (HH:MM:SS:FF)
export function formatEta(seconds: number): string                  // "1м 20с" / "3с" / ""
export function formatBytes(n: number): string                      // для размеров моделей (config page)
```

- [ ] **Step 1:** `cd webui && bunx shadcn@latest add slider tooltip toggle-group popover switch` — проверить, что компоненты легли в `src/shared/ui/` и используют `cn`/Base UI; если CLI изменил алиасы — поправить импорты.
- [ ] **Step 2: usePoll + format** (код выше; реализация тривиальна — setInterval в useEffect с cleanup, id renegotiation через useRef).
- [ ] **Step 3:** `bun run lint && bun run typecheck`.
- [ ] **Step 4: Commit** `feat(ui): usePoll, formatters, editor shadcn components`.

---

### Task 3: entities/source + entities/preset + job-расширение

**Files:**
- Create: `webui/src/entities/source/{types.ts, api.ts, index.ts}`
- Create: `webui/src/entities/preset/{types.ts, api.ts, index.ts}`
- Modify: `webui/src/entities/job/types.ts` (kind + source_id/source_name)
- Modify: `webui/src/entities/job/api.ts` (unified submitJob)
- Create: `webui/src/entities/annotation/{types.ts, mask.ts, api.ts, index.ts}`

**Interfaces:**

```ts
// entities/source/types.ts
export type SourceProbe = { fps: number; duration_s: number; width: number; height: number; frame_count: number; has_audio: boolean }
export type Source = { id: string; name: string; createdAt: string; probe: SourceProbe; video_url: string; annotations_url: string }
export type AnnotationEntry = { frame: number; url: string; strokes: Stroke[]; updatedAt: string }
// Stroke определён в entities/annotation, source импортирует тип из annotation (слой entities: между слайсами import типов допустим через index.ts вниз? — НЕТ, один слой: тогда Stroke живёт в source? РЕШЕНИЕ: Stroke и mask-математика живут в entities/annotation; entities/source НЕ импортирует annotation — AnnotationEntry.strokes типизируется как unknown[] в source, а annotation переэкспортирует парсер. Проще: AnnotationEntry объявлен в entities/annotation (masks API — часть annotation). В source остаётся только видео-часть.

// entities/annotation/types.ts
export type Tool = "brush" | "eraser"
export type Stroke = { tool: Tool; size: number; points: [number, number][] }  // координаты в пикселях видео
export type AnnotatedFrame = { frame: number; url: string; strokes: Stroke[]; updatedAt: string }

// entities/annotation/mask.ts — чистые функции (canvas)
export function renderStrokesToCanvas(strokes: Stroke[], w: number, h: number): HTMLCanvasElement
// — чёрный прозрачный фон, БЕЛЫЕ штрихи brush (globalCompositeOperation "source-over", round cap/join), eraser — "destination-out"
export function canvasToPngBlob(c: HTMLCanvasElement): Promise<Blob>
export function imageToCanvas(url: string): Promise<HTMLCanvasElement>  // загрузка PNG маски (для восстановления)
```

```ts
// entities/annotation/api.ts
export const fetchAnnotations = (sourceId: string) => api<{ frames: AnnotatedFrame[] }>(...)
export const putMask = (sourceId: string, frame: number, png: Blob, strokes: Stroke[]) => FormData → PUT
export const deleteMask = (sourceId: string, frame: number) => DELETE
export const maskUrl = (sourceId: string, frame: number) => `/api/sources/${sourceId}/masks/${frame}`
```

```ts
// entities/preset/api.ts
export const listPresets = () => api<Preset[]>("/api/presets")
export const savePreset = (name: string, payload: Record<string, unknown>) => api<Preset>("/api/presets", { method: "POST", body: JSON.stringify({ name, payload }) })
export const deletePreset = (id: string) => api(`/api/presets/${id}`, { method: "DELETE" })
// types.ts: Preset = { id: string; name: string; payload: Record<string, unknown>; createdAt: string }
```

```ts
// entities/job: Job.kind расширяется до "run" | "preview" | "prompt"; добавляются source_id: string | null, source_name: string | null
// entities/job/api.ts: submitRun заменяется на
export function submitJob(fields: {
  kind: "run" | "preview" | "prompt"
  source_id?: string
  prompt?: string
  targets?: unknown[]       // сериализуется в JSON-строку
  tracks?: unknown[]
  masks?: number[]
  mode?: "parse" | "detect"
  start?: number; count?: number; stride?: number; indices?: number[]; all?: boolean
  params?: Record<string, string | number | boolean>  // конфиг-поля serialize_clean_form
  video?: File              // legacy-загрузка без source
}): Promise<Job>
// — собирает FormData: промпт/строки как есть, targets/tracks через JSON.stringify, masks через join(","), params по одному полю, all="1".
```

- [ ] **Step 1:** написать entities по контрактам выше; `index.ts` каждого слайса реэкспортирует публичное.
- [ ] **Step 2:** `bun run lint && bun run typecheck`.
- [ ] **Step 3: Commit** `feat(ui): source, annotation, preset entities; unified submitJob`.

---

### Task 4: features — движки ступеней

**Files:**
- Create: `webui/src/features/annotate/index.ts` — `useAnnotate`
- Create: `webui/src/features/interpret/index.ts` — `useInterpret`
- Create: `webui/src/features/detect-run/index.ts` — `useDetectRun`
- Create: `webui/src/features/inpaint-run/index.ts` — `useInpaintRun`
- Modify: `webui/src/features/preview-run/index.ts` — УДАЛИТЬ (заменён detect-run); `features/frame-selection` — УДАЛИТЬ (клик по таймлайну заменяет выделение)

**Контракты:**

```ts
// useAnnotate(source: Source | null) — движок рисования; штрихи хранятся per-frame в памяти,
// автосохранение (debounce 600ms) PUT mask при изменении кадра.
// Состояние: strokesByFrame: Record<number, Stroke[]>, activeFrame, tool, size,
// beginStroke(pt), extendStroke(pt), endStroke(), undo(), clearFrame(),
// undoStack (для undo последнего штриха), isDirty.
// renderStrokesToCanvas вызывается потребителем (viewer) — feature не держит DOM.
export function useAnnotate(source: Source | null): {...}
```

```ts
// useInterpret(source) — ступень 1→2.
// run(prompt: string): POST submitJob({kind:"prompt", source_id}) → poll getJob(1s, макс 600) →
// отчёт: job COMPLETED → fetchPreviewManifest НЕ нужен; таргеты и промпт лежат в report.
// Как достать report? job_dict НЕ отдаёт report_json. РЕШЕНИЕ (бэкенд уже умеет): после COMPLETED
// prompt-джобы читаем report через существующий /api/jobs/{id}/preview/preview.json? Нет — prompt-джоба
// пишет только report.json (нет preview/). ИНСТРУМЕНТ: ReportJob() — Add GET /api/jobs/{id}/report route? 
// ЭТОГО НЕТ В БЭКЕНДЕ. ОБОРОТ: submitJob kind=prompt уже возвращает job_dict; report недоступен.
// РЕШЕНИЕ ПЛАНА: fetch(`/api/jobs/${id}/preview/preview.json`) не сработает → используем
// job_dict полей… их нет. ЗНАЧИТ: эта задача добавляет крошечный бэкенд-эндпоинт
// GET /api/jobs/{job_id}/report → report_json (FileResponse JSON) — СМОТРИ Task 4b ниже.
export function useInterpret(source: Source | null): { run(prompt: string): Promise<void>; running: boolean; error: string; result: { prompt: string; targets: TargetJson[] } | null }
```

**Task 4b (бэкенд, маленький):** `GET /api/jobs/{job_id}/report` — отдаёт `report_json` джобы как JSON (или 404/409 если нет). Тест в `tests/test_api_jobs_kinds.py`: у COMPLETED prompt-джобы (созданной через unified endpoint с worker=False — просто положить report_json через `jobs.upsert`) эндпоинт возвращает `{"kind": "prompt", ...}`. Файлы: `server/fastapi_app.py` (+5 строк), `tests/test_api_jobs_kinds.py`. Коммит `feat: GET /api/jobs/{id}/report for editor stage results`. Выполнить ПЕРЕД фронтенд-частью Task 4.

```ts
// useDetectRun(source, params) — ступень 3.
// run({ mode: "parse"|"detect", prompt, targets, stride | all }):
//   kind=preview, source_id; all → all: true; иначе stride: number.
// poll getJob до терминального; при COMPLETED fetchPreviewManifest(jobId) + fetchJobReport(jobId) (tracks).
// Состояние: running, progress (job.fraction/detail/eta), error, manifest, tracks.
export function useDetectRun(source: Source | null): {...}
```

```ts
// useInpaintRun(source) — ступень 4.
// run(mode: "tracks" | "masks" | "prompt", payload): kind=run c tracks/masks/undefined,
// params — конфиг ступени. poll; при COMPLETED → jobId результата (для before/after).
export function useInpaintRun(source: Source | null): {...}
```

Все четыре хука используют `useJobPoll`-паттерн (общий локальный helper внутри features? — нет, один слой; дублирование цикла poll в каждом хуке недопустимо → вынести `pollJobToCompletion(id, onProgress)` в `entities/job/api.ts` — он там уместен: это API-логика).

```ts
// entities/job/api.ts добавка
export async function pollJobToCompletion(
  id: string,
  onProgress?: (j: Job) => void,
  intervalMs = 1000,
  maxSeconds = 1800,
): Promise<Job>
```

- [ ] **Step 1: Task 4b (бэкенд):** тест → эндпоинт → commit.
- [ ] **Step 2:** `pollJobToCompletion` в entities/job + теста нет (фронт) — typecheck.
- [ ] **Step 3:** четыре feature-хука по контрактам; удаление preview-run и frame-selection (сначала убедиться, что pages/workspace ещё не сломан: workspace импортирует их — на это есть T12; ВРЕМЕННО workspace оставляем компилируемым: замените импорты на заглушки? НЕТ — правильный порядок: features создаём, старые не трогаем до T12. Удаление старых — ТОЛЬКО в T12. Поэтому этот таск только СОЗДАЁТ файлы.)
- [ ] **Step 4:** `bun run lint && bun run typecheck`.
- [ ] **Step 5: Commit** `feat(ui): stage engines — annotate, interpret, detect, inpaint (+ report endpoint)`.

---

### Task 5: widgets/library

**Files:**
- Create: `webui/src/widgets/library/{index.tsx, upload.tsx}`

**Контракт:** левая колонка (~280px, scroll-area): список источников (`listSources` + usePoll 5s) — имя, длительность (formatTimecode из probe), количество аннотаций не показываем; клик → выбор источника; кнопка загрузки (Input type=file → submitSource upload) сверху; под источниками — секция «Без источника» (job'ы с source_id=null из /api/poll) — только статус/скачивание; под выбранным источником — его job'ы (из /api/poll filter by source_id, usePoll 3s): kind-иконка (промпт/маски/инпейнт), state-цвет (ok green / running amber pulse / failed red), время; действия job'а: Отмена (QUEUED/RUNNING), Повторить (FAILED/CANCELLED), Скачать (has_output), Удалить (confirm через window.confirm не используем — просто DELETE; в UI подпись «Удалить»).

Props: `{ selectedId: string | null; onSelect: (s: Source) => void; refreshKey: number }`.

- [ ] Реализация; **Step: lint+typecheck; Commit** `feat(ui): library — sources and their jobs`.

---

### Task 6: widgets/editor-viewer (видео + слои + тулбар)

**Files:**
- Create: `webui/src/widgets/editor-viewer/{index.tsx, annotation-layer.tsx, mask-overlay.tsx, compare.tsx}`

**Контракт:**

```tsx
type EditorViewerProps = {
  source: Source
  currentFrame: number
  onFrameChange: (frame: number) => void
  annotate: ReturnType<typeof useAnnotate>   // активен в режиме annotate
  mode: "annotate" | "detect" | "result"
  detectMaskUrl: string | null               // /api/jobs/{id}/preview/{frame}_mask.jpg для текущего кадра
  maskOpacity: number                        // 0..1, слайдер снаружи
  detectBoxes: (number[] | null)[]           // боксы текущего кадра из манифеста
  resultJobId: string | null                 // для before/after в режиме result
}
```

- **Координаты:** контейнер `relative` с `aspect-ratio: width/height` от probe; `<video>` абсолютный `inset-0 h-full w-full`; canvas-слои поверх с `width={probe.width} height={probe.height}` и CSS `h-full w-full` — координаты указателя переводятся: `(e.clientX - rect.left) / rect.width * probe.width`.
- **Скраб:** `<video>` без нативных controls; клик/драг по видео ставит кадр: `video.currentTime = frame / fps` (listener `seeked` не зацикливать); requestAnimationFrame не нужен. Клавиатура (на контейнере tabIndex=0): ←/→ ±1 кадр, Shift+←/→ ±10, Space play/pause (по умолчанию видео paused; после Space играет, `timeupdate` → onFrameChange(timeToFrameIndex)).
- **annotation-layer:** canvas, перерисовывается на strokesByFrame[currentFrame] изменение и во время рисования (rAF-троттлинг); pointerdown/move/up → annotate.beginStroke/extend/end; курсор crosshair; размер кисти — annotate.size.
- **mask-overlay:** `<img src={detectMaskUrl}>` c `style={{ opacity: maskOpacity }}`, `pointer-events-none`, скрыт при отсутствии.
- **detectBoxes:** абсолютные div'ы поверх (координаты/100 → %), бордер `border border-[#E5484D]` + подпись трека сверху (label из tracks по совпадению индекса; допускается просто номер бокса) — pointer-events-none.
- **compare.tsx:** before/after — контейнер с двумя `<video>` (input_url джобы-источника и output_url resultJobId), output сверху с `clipPath: inset(0 ${100 - pos}% 0 0)`; вертикальный разделитель draggable (pointer events, pos в %); синхронизация play/pause/currentTime (output следует за input). Разметка подписей «До»/«После» по краям. `prefers-reduced-motion`: без анимаций.
- **Тулбар** (annotate mode, снизу или сбоку viewer'а): ToggleGroup кисть/ластик (иконки lucide Brush/Eraser), Slider размера (4..120), Undo (Undo2), Очистить кадр (Trash2), opacity слайдер (detect mode), подпись «Кадр N · маска сохранена/сохранение…».

- [ ] Реализация трёх файлов; **lint+typecheck+build; Commit** `feat(ui): editor viewer — video, annotation layer, mask overlay, compare`.

---

### Task 7: widgets/timeline

**Files:**
- Create: `webui/src/widgets/timeline/{index.tsx, thumbs.ts}`

**Контракт:** логика миниатюр переносится из старого filmstrip (canvas seek-to-draw) в `thumbs.ts` (чистая функция `grabThumbs(src, fps, frameCount, count, onThumb): () => void` с отменой); маркеры: аннотированные кадры (annotate.strokesByFrame keys, amber точка под миниатюрой), кадры с масками detect (manifest frames indices, красная точка); клик по миниатюре → onFrameChange; текущий кадр — ring-amber. Горизонтальный скролл, высота ~64px. Props: `{ src, fps, frameCount, currentFrame, onFrameChange, annotatedFrames: number[], maskedFrames: number[] }`. Иконки не нужны.

- [ ] Реализация; **lint+typecheck; Commit** `feat(ui): timeline with thumbs and stage markers`.

---

### Task 8: widgets/stage-rail

**Files:**
- Create: `webui/src/widgets/stage-rail/{index.tsx, params.tsx, targets-editor.tsx}`

**Контракт:** правая колонка (~360px, scroll), 5 секций-ступеней (шапка: номер-буква не numbered-маркер не нужен — статус-точка + название; активная ступень подсвечена amber-левой-линией):

1. **Вход** — Textarea промпта (любой язык); статус-чип LLM (ollama.ok из /api/poll: «LLM готов»/«Ollama недоступна → Конфиг», ссылка на таб Конфиг); кнопка **Интерпретировать** (disabled если нет промпта И нет масок; running-состояние с прогрессом джобы).
2. **Таргеты** — TargetsEditor: список {kind select, query input, where select, enabled checkbox, бейдж источника «маска N»/«текст»}; кнопка «+ Таргет»; бейджи источников приходят из interpret-результата (targets с пометкой which mask их породила — в report prompt-джобы этой информации нет, значит бейдж только «из интерпретации»/«вручную» — РЕШЕНИЕ: бейдж «авто»/«вручную»); редактируемый промпт (из интерпретации, правится).
3. **Маски (detect)** — выбор: «Всё видео» (all) / «Каждый N-й кадр» (stride, Slider 1..30, авто-подсказка: stride = max(1, round(frame_count/400))); кнопка **Найти маски** (mode=detect по таргетам или mode=parse по промпту — переключатель «По таргетам / По промпту»); прогресс (fraction, eta, кнопка Стоп → cancelJob); после завершения — meanMaskCoverage и число треков.
4. **Inpaint** — три режима (ToggleGroup): «По трекам» / «По маскам» (предупреждение: «Маски применяются ко всем кадрам — только для неподвижных объектов» AlertDescription-стилем) / «По промпту»; select inpainter (opencv-telea/lama/propainter), параметр mask_dilate_px (Slider 0..15), verify (Switch), телета-радиус при telea; кнопка **Запустить**; прогресс.
5. **Результат** — при COMPLETED: превью compare открывается в viewer (mode=result), кнопки «Скачать» (output_url), форматы вывода (список formats = mp4/webm/mov/mkv чекбоксами → formats param), «Создать пресет из параметров» (Popover: имя → savePreset со снимком всех ступеней) и селектор пресетов (apply/delete).

`params.tsx`: **аккумулятор параметров** — единый стейт ступеней живёт в pages/workspace (single source of truth), stage-rail получает values+onChange (контролируемый). Тип:

```ts
export type EditorParams = {
  prompt: string
  targets: TargetRow[]            // + enabled: boolean
  detect: { mode: "targets" | "prompt"; all: boolean; stride: number }
  run: { inpainter: string; mask_dilate_px: number; telea_radius: number; verify: boolean; formats: string[]; keep_workdir: boolean; min_mask_coverage: number; verify_max_coverage: number; llm_base_url: string; llm_api_key: string }
  advanced: Record<string, string>  // остальные поля serialize_clean_form (аккордеон «Все параметры» внизу ступени 4: full list из serialize_clean_form, дефолты подставлены)
}
```

Дефолты заполняются из констант (файл `params.ts` внутри stage-rail): inpainter=opencv-telea, dilate=3, telea=9, verify=true, formats=["mp4"], min_mask_coverage=0.0004, verify_max_coverage=0.12, остальные из дефолтов бэкенда (detector_threshold=0.15 и т.д. — список в docs/PARAMS.md). Кнопка «Сбросить к дефолтам».

- [ ] Реализация; **lint+typecheck+build; Commit** `feat(ui): stage rail — interpret, targets, detect, inpaint, result + presets`.

---

### Task 9: pages/workspace + удаление старых виджетов

**Files:**
- Modify: `webui/src/pages/workspace/index.tsx` (полный реворк)
- Delete: `webui/src/widgets/{viewer,filmstrip,preview-grid,jobs-sidebar,target-table,run-form}/`
- Delete: `webui/src/features/preview-run/`, `webui/src/features/frame-selection/`
- Modify: `webui/src/entities/frame/index.ts` — оставить чистые функции (используются thumbs/format)

**Контракт workspace:** `h-full grid grid-cols-[280px_1fr_360px] grid-rows-[1fr_auto]` (левая library на 2 строки, центр: viewer (row 1) + timeline (row 2), правая stage-rail на 2 строки). Состояние: selectedSource, currentFrame, viewerMode (annotate всегда доступен; detect → mask overlay; result → compare), params (EditorParams), хуки ступеней. Пустое состояние (нет source): центр — приглашение «Перетащите видео или выберите источник слева» (drag&drop опционально; кнопка загрузки в library обязательна). Заголовок над viewer: Timecode текущего кадра + имя source.

Авто-режим «Запустить всё» — кнопка в шапке stage-rail: если маски есть и промпта нет → цепочка prompt→(авто-заполнение таргетов)→run; если промпт есть и масок нет → сразу kind=run по промпту; если есть и то и другое → prompt→run с targets_override. Индикация: ступени последовательно подсвечиваются.

- [ ] Реворк workspace; удалить старые виджеты/фичи (проверить, что никто их не импортирует: `grep -r "jobs-sidebar\|preview-grid\|run-form\|target-table\|widgets/viewer\|widgets/filmstrip\|preview-run\|frame-selection" webui/src` → пусто).
- [ ] `bun run lint && bun run typecheck && bun run build` (билд обязателен — статика для server).
- [ ] **Commit** `feat(ui): editor workspace; remove superseded job-based widgets`.

---

### Task 10: Конфиг-страница + фиксы полировки

**Files:**
- Modify: `webui/src/pages/config/index.tsx` (минимально: русские подписи уже есть; добавить кнопку «Отменить загрузку модели» → POST /api/models/cancel; отобразить source-независимые вещи не нужно)

**Acceptance:** конфиг работает с новыми токенами; кнопка отмены загрузки; без регрессий.

- [ ] Правки; **lint+typecheck+build; Commit** `feat(ui): config page — cancel downloads, token alignment`.

---

### Task 11: Дымовое тестирование и исправления

**Files:** по обстоятельствам.

- [ ] **Step 1:** контроллер запускает `make serve`-эквивалент в worktree: `uv run videoclean serve` (VIDEOCLEAN_UI_PASSWORD=pw) — и через agent-browser проходит сценарий: загрузить короткий тестовый клип (ffmpeg testsrc), нарисовать маску, интерпретация (если LLM нет — ожидаем ошибку чипа), detect по ручным таргетам на 2 кадрах, запуск inpaint по трекам, before/after, скачивание. Скриншоты ключевых экранов.
- [ ] **Step 2:** найденные дефекты правятся отдельным(и) коммитами `fix(ui): ...` с обычным review-циклом SDD (или контроллер-фикс при тривиальности — НЕТ, по процессу: через субагента).
- [ ] **Step 3:** финальный `bun run lint && bun run typecheck && bun run build` + `uv run pytest -q` (бэкенд-тесты не должны пострадать).
- [ ] **Commit** (если фиксы были).

---

## Самопроверка плана (выполнена)

1. **Покрытие спеки §7:** layout (T1/T9), library (T5), editor-viewer+слои+compare (T6), timeline+маркеры (T7), stage-rail с чекбоксами таргетов, бейджами, ETA/Стоп, пресетами (T8), features (T4), entities (T3), токены (T1). §8 сценарии — T9/T11. §9 edge cases: статус-чип (T8.1), предупреждение masks_override (T8.4), «Без источника» (T5), auto-stride (T8.3), reduced-motion/focus (Global).
2. **Ключевое добавление к бэкенду:** Task 4b — `GET /api/jobs/{id}/report` (для результатов prompt-джоб). Это единственное серверное изменение Плана B; plan-mandated, протестирован.
3. **Известные упрощения:** бейдж таргета «авто/вручную» (не «из маски N» — prompt-отчёт не несёт маппинг маска→таргет; отмечено в спеке §9 как ограничение v1); drag&drop загрузки опционален; удаление старых виджетов отложено до T9 (workspace остаётся компилируемым между T4 и T9).
4. **Типы:** Stroke рендерится в координатах видео; submitJob — единственная точка POST /api/jobs; EditorParams — single source of truth в workspace.
