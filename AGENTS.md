# AGENTS.md — videoclean

CLI + WebUI для удаления объектов/текста/логотипов из видео по текстовому промпту.

## Структура репозитория (три контура)

| Контур | Путь | Что это |
|---|---|---|
| library | `videoclean/` | Python-библиотека, Clean Architecture |
| server  | `server/`    | FastAPI API + job-воркер (опционально раздаёт static) |
| webui   | `webui/`     | React-фронт (Vite + FSD); в split-деплое — отдельный nginx-образ |

### library: `videoclean/` (Clean Architecture)

- `domain/` — сущности и правила (Target, Track, MediaManifest). Не импортирует ничего из внешних кругов.
- `application/` — use cases (`use_cases/`), порты (`ports/`), конфиг, выбор треков. Импортирует только domain.
- `adapters/` — реализации портов: media (ffmpeg), detectors, segmenters, inpainters, prompt/llm, models, progress. Импортируют application/domain.
- `composition.py` — composition root: собирает зависимости. Может импортировать всё из библиотеки.
- `cli.py` — тонкий CLI-вход (typer) поверх use cases.

Правило зависимостей — source dependencies указывают внутрь:
`domain ← application ← adapters ← composition`. Контролируется `tests/test_architecture.py`.

### server: `server/`

FastAPI-хост: `fastapi_app.py` (роуты, auth), `app_state.py`, `service.py`, статика.
- Зависит от `videoclean` (one-way). `videoclean` НЕ импортирует `server`.
- Новые эндпоинты — тонкие: парсинг формы → use case из библиотеки. Бизнес-логики в server нет.

### webui: `webui/` (Feature-Sliced Design)

Слои (импорт только «вниз»; между слайсами одного слоя — нельзя):
- `app/` — корень: провайдеры, табы-роутинг
- `pages/` — workspace, config
- `widgets/` — крупные блоки: filmstrip, viewer, preview-grid, jobs-sidebar, target-table, run-form
- `features/` — сценарии: preview-run, frame-selection
- `entities/` — job, frame, preview, targets: типы + API-клиенты + чистая логика
- `shared/` — api-клиент, `ui/` (shadcn), `lib/`, `hooks/`

Public API слайса — `index.ts`. Снаружи слайс импортируется только через него.
`components.json` aliases указывают на `@/shared/*`.
UI собирается ТОЛЬКО из shadcn-компонентов (`bunx shadcn@latest add <name>`),
кастомные стили не выдумываем — Tailwind utilities для layout и всё.

## Команды

Все повседневные команды — через `make` (см. `make help`):

```bash
make setup    # разовая установка: uv sync + bun install
make dev      # дев: FastAPI :7860 + vite hot-reload :5173 (proxy /api); VIDEOCLEAN_AUTH=off
make serve    # локальный one-box: FastAPI :7860 + static_dist
make web      # собрать фронт → server/static_dist/
make docker   # образы api + web (split)
make compose-up  # docker compose: api :7860 + web :8080
make test     # pytest
make lint     # eslint + tsc для webui
```

Split-деплой (разные инстансы): `Dockerfile.api` + `Dockerfile.web`, `VITE_API_BASE_URL` (URL API глазами браузера), на внутреннем контуре `VIDEOCLEAN_AUTH=off`. One-box: `docker compose --profile monolith up` или `make docker-monolith`.

Те же команды напрямую:

```bash
# Python (из корня)
uv run pytest -q
uv run videoclean --help
uv run videoclean serve            # http://127.0.0.1:7860 (admin/admin)

# Frontend (из webui/)
bun install
bun run dev        # vite dev (proxy /api → :7860)
bun run lint && bun run typecheck && bun run build   # билд → server/static_dist/
bunx shadcn@latest add <component>                   # новые shadcn-компоненты
```

## Документация

- `docs/PARAMS.md` — все параметры пайплайна (CLI/WebUI/doctor).
- `docs/MODELS.md` — модели: что брать/не брать и почему.
- `docs/superpowers/specs|plans/` — спеки и планы фич (в .gitignore, не коммитятся).

## Конвенции

- Python: python 3.11, типизация, docstring кратко по-английски, без комментариев-мусора.
- Тесты: `tests/test_<area>.py`, pytest. Новые тесты — только на новые эндпоинты/фиксы (максимум необходимого).
- Frontend: TS strict, функции-компоненты, vanilla fetch (без axios/react-query), shadcn-компоненты для всего UI.
- UI-тексты — на русском; идентификаторы/код — en.
