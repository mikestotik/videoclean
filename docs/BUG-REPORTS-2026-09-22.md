# Bug reports — 2026-09-22 (RunPod test deploys)

## #1 — Cancel не останавливает джобу (OPEN)
- Симптом: `POST /api/jobs/20260922_144717_8fd5b520/cancel → 200 OK` ×4, работа продолжает висеть. Кнопка не дизейблится, шлёт повторы, бэк их «успешно» глотает.
- Ожидание: первый cancel → `CANCELLED` + остановка воркера.
- Не хватает: статус джобы после cancel, состояние воркера (очередь vs `IN_PROGRESS`).
- Тяжесть: high.

## #2 — `no detector could run. grounding-dino: 0 tracks` на GPU (RESOLVED, причина — хост)
- Симптом: «Найти маски» на `device=cuda` → красная ошибка; на `device=cpu` медленно едет (~10%/мин).
- Расследование: исключены по очереди — версия torch (5 сборок: cu121, cu124, cu126, cu128, cu130 — одна и та же `CUDA unknown error`), device nodes (`/dev/nvidia0` symlink не помог), `nvidia-uvm` на месте, `nvidia-smi` живой (драйвер 580.178.04 / CUDA 13.0).
- Root cause: битые хосты RunPod — драйвер не создаёт CUDA-контекст (`_cuda_getDeviceCount: CUDA unknown error`), два хоста подряд (GPU minor 2 и 3). Из контейнера не чинится.
- Решение: другой хост/DC. На здоровом хосте `torch 2.8.0+cu128 cuda=yes`, детекция едет.
- Связано: формулировка «0 tracks» врёт (см. #3).

## #3 — Детектор маскирует ошибку загрузки под «0 tracks» (OPEN, код)
- `GroundingDinoDetector.discover()` (`videoclean/adapters/detectors/grounding_dino.py:84-86`): при провале `_try_load()` тихо возвращает `[]`, `_load_error` никуда не попадает. `run_preview._discover` (`run_preview.py:322`) рапортует `grounding-dino: 0 tracks` вместо `error (CUDA...)`.
- Фикс (отложен): прокидывать текст `_load_error` в `attempts` как `error (...)`. Одним коммитом, после стабилизации деплоя.

## #5 — «Найти маски» игнорирует выбранную модель SAM (FIXED)
- Симптом: в Масках выбрана `SAM2.1 small` (скачана), а превью требует `facebook/sam2-hiera-tiny`.
- Root cause: `toDetectParams` (`webui/src/widgets/stage-rail/params.ts`) выкидывал `segmenter_model` из запроса; сервер (`service.py:102-103`) падал на дефолт. Сервер форсит только тип (`segmenter="sam2"`, `service.py:300`), модель умеет брать из запроса.
- Фикс: не выкидывать `segmenter_model` (одна строка + коммент). `tsc` + `eslint` чисто.
- Применение на поде: pull → пересобрать UI → повторить «Найти маски».

## #4 — `sqlite3.OperationalError: disk I/O error` (RESOLVED, workaround)
- Симптом: `POST /api/models/cancel → 500`, после рестарта падает сам старт сервера (`PRAGMA journal_mode=WAL` на свежей базе).
- Root cause: сетевой том `mfs#eu-cz-1` не поддерживает SQLite-локи/WAL. Обычные файлы пишутся, sqlite — нет. Наши файлы ни при чём (кеш 388M).
- Workaround: `VIDEOCLEAN_DATA_DIR` на локальном диске контейнера (`/root/.videoclean`), `HF_HOME` остаётся на томе. Pod volumes — локальный диск, там проблемы нет.
- Закалка на будущее (open): ловить `OperationalError` на старте с понятным сообщением; рассмотреть `journal_mode=DELETE` для сетевых FS. Отдельным коммитом.
- Побочка из того же вечера: квота mfs (`Errno 122` на запись refs) — код её уже игнорирует, закачки идут. Лечится переносом `HF_HOME` на локальный диск.

## #6 — `ModuleNotFoundError: No module named 'sam2'` (FIXED, env)
- Симптом: «Найти маски» падает с отсутствием модуля sam2, хотя он ставился.
- Root cause: sam2 ставится через `uv pip` из гита мимо локфайла; любой следующий `uv sync` (включая частичный `uv sync --extra web`) сносит его как extraneous. Добили пустые METADATA у sam2-зависимостей (записи, оборванные квотой) — uv не мог даже переустановить, пока метки не удалены вручную.
- Фикс: `uv sync --inexact` во всех pod-синках + pinned pip install + `UV_CACHE_DIR` на локальном диске (кеш 37G на томе ронял квоту).
- Отклонено: git-зависимость sam-2 в `pyproject` — `uv lock` не может склонировать репозиторий из dev-сети (3 обрыва fetch-pack). Вернуться, когда сеть позволит.
- Восстановление пода: `rm -rf` битых dist-info → pip install sam2 с пином коммита → без рестарта serve (ленивый импорт).
