# Спека: React-редактор масок, фаза 1 (+ разделение library / server / webui)

Дата: 2026-09-08
Статус: утверждена (брейнсторм 2026-09-08, вариант «React + shadcn», фаза «сначала просмотр»)

## Цель

1. Разделить репозиторий на три контура: **library** (`videoclean/`, Clean Architecture),
   **server** (`server/`, FastAPI-хост), **webui** (`webui/`, React, FSD).
2. Фаза 1 редактора: филмстрип с мультивыбором кадров, превью на выделении,
   таблица таргетов, запуск полных прогонов и конфиг — всё на React.
3. Зафиксировать архитектурные правила в `AGENTS.md`.

Вне скоупа (фаза 2): рисование прямоугольников, VLM-называние региона,
ручные маски-боксы в пайплайне.

## Структура репозитория

```
videoclean/   — библиотека (Clean Architecture): domain/ application/ adapters/ + cli.py, composition.py
server/       — FastAPI-хост: fastapi_app, app_state, service, статика; зависит ТОЛЬКО от videoclean
webui/        — Vite + React + TS + Tailwind 4 + shadcn (base-ui), FSD
```

Правило зависимостей (одностороннее):
- `domain` ← `application` ← `adapters` ← `composition` (composition root библиотеки);
- `server → videoclean` (one-way); `videoclean` НЕ импортирует `server`;
- `webui → server API` (HTTP); webui не импортирует python и наоборот.
- Контролируется `tests/test_architecture.py` (расширен).

## Backend-изменения

1. `videoclean/adapters/web/*` → `server/` (fastapi_app, app_state, service, статика).
   `progress_bridge.py` НЕ едет в server: он чистая обёртка над JobStore и нужен
   composition (build_job_worker). Переезжает в `videoclean/adapters/progress/job_store.py`
   (класс ProgressBridge сохраняет имя) — это фикс нарушения Dependency Rule
   «composition импортирует web-адаптер».
2. `cli.py serve` импортирует `server.fastapi_app.launch_from_env`.
3. `pyproject.toml`: hatch packages = ["videoclean", "server"].
4. Новые эндпоинты (нужны фронту):
   - `GET /api/jobs/{id}/probe` — fps/duration/width/height/frame_count входного видео
     (FFmpegMedia().probe), для маппинга таймкод ↔ номер кадра.
   - `POST /api/preview/from-job` — превью по уже загруженному видео (JSON body:
     job_id, prompt, mode, targets, indices, ...), без re-upload файла.
5. Статика: `server/static_dist/` (собранный webui, Vite build) монтируется при наличии;
   `/` и `/config` отдают SPA. Если dist не собран — HTML-заглушка с инструкцией.
   Legacy `app.js`/`index.html` удаляются (финальная задача фазы).

## Frontend (FSD)

Слои (импорты только «вниз», между слайсами — только через public API/index.ts):
```
app/        — корень приложения: табы (Рабочая/Конфиг), провайдеры
pages/      — workspace, config
widgets/    — filmstrip, viewer, target-table, preview-grid, jobs-sidebar, run-form
features/   — preview-run (отправка превью на выделении), frame-selection
entities/   — job, frame, preview (типы + API + чистая логика с vitest-тестами)
shared/     — api-клиент (fetch, Basic из браузера), ui/ (shadcn), lib/
```
- `components.json` aliases → `@/shared/ui`, `@/shared/lib`, `@/shared/hooks`.
- Vite aliases: `@app @pages @widgets @features @entities @shared`.
- Роутинг без зависимостей: табы-состояние (2 страницы).
- Филмстрип: thumbnails в браузере (canvas + seek), равномерная сетка N штук;
  клик — кадр в просмотрщик; ctrl/cmd+клик — toggle; shift+клик — диапазон.
  Маппинг thumbnail→frame index: `round(t * fps)` (fps из probe).
- Превью: `POST /api/preview/from-job` с выбранными indices; результат —
  preview.json + `{index}_boxes.jpg / _mask.jpg / _raw.jpg` с тумблером слоёв и
  лайтбоксом.
- Полный прогон: multipart `POST /api/jobs` (upload + форма) — как в legacy.
- Конфиг: модели (группы, download), doctor, ollama — данные из `/api/poll`.

## Сборка и поставка

- Пакетный менеджер webui: **bun** (bun.lock в шаблоне).
- `bun run build` → `webui/dist` → копируется в `server/static_dist/`
  (скриптом `webui/scripts/build.mjs` или vite `build.outDir = ../server/static_dist`).
- Vitest для чистой логики (selection, маппинг кадров, парсинг preview.json).
- Dockerfile: node-stage для сборки webui + копирование dist (RUNPOD-образ).

## AGENTS.md

Создаётся в корне: структура трёх контуров, правила зависимостей, команды
(uv run pytest; cd webui && bun run lint/typecheck/build), указатели на
docs/PARAMS.md, docs/MODELS.md, docs/superpowers/.
