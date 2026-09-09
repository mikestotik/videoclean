# План реализации: Preview mode + scoped targets

Спека: `docs/superpowers/specs/2026-09-08-preview-mode-design.md`

Все шаги TDD: падающий тест → реализация → рефактор. Коммит после каждого шага. Прогон всего сьюта перед каждым коммитом.

## Шаг 1. Target.frames + scoped select

1. tests/test_select.py: тесты `select_tracks` с intent, где targets имеют `frames=(a, b)`:
   - трек с боксами внутри окна остаётся, боксы вне окна обнуляются до интерполяции
   - `frames=None` — совместимость: поведение как раньше
   - трек целиком вне окна отфильтрован
2. domain/intent.py: `frames: tuple[int, int] | None = None`.
3. application/select.py: фильтрация боксов по окнам до `interpolate_gaps`.
4. Коммит: `feat: scoped targets with frame windows in select`.

## Шаг 2. Чанковый vision-парс

1. tests/test_llm_parser.py:
   - `parse_chunk_frames` больше prompt_frame_max*stride → парс по чанкам, таргеты получают окна
   - соседние чанки с одинаковыми таргетами сливаются в одно окно
   - один чанк → parse_mode `llm-vision`, несколько → `llm-vision-scoped`
   - конфиг: `parse_chunk_frames=0` — текущее поведение
2. application/config.py: поле + валидация (>= 0).
3. adapters/prompt/llm.py: сигнатура `parse(prompt, frames, frame_indices, parse_chunk_frames)`; при чанках — окна.
4. composition.py: прокинуть `parse_chunk_frames` в парсер.
5. Коммит: `feat: chunked vision parse with scoped targets`.

## Шаг 3. extract_frames_subset

1. tests/test_frames.py: `extract_frames_subset(input, indices, out_dir, log)`:
   - извлекает только перечисленные кадры, имена `frame_{idx:06d}.png` как у полного извлечения
   - сортировка и дедуп индексов
   - выход: список Path в порядке возрастания индексов, параллельно индексам
2. adapters/media/ffmpeg.py: `-vf select='eq(n,a)+eq(n,b)'`, `vsync 0` (VFR), лог в тот же файл.
3. application/ports/media.py: метод в протокол.
4. Коммит: `feat: ffmpeg frame subset extraction`.

## Шаг 4. RunPreview use case

1. tests/test_run_preview.py (фейки из test_run_cleanup):
   - happy path: парс → discover на всех кадрах → маски → артефакты, отчёт COMPLETED
   - mode=detect: парсер не вызывается, targets из запроса
   - segmenter форсируется в покадровый (фейк с video_aware-инпейнтером не запускает inpaint вообще)
   - coverage в отчёте совпадает с масками
   - preview.json содержит queries, targets, per-frame боксы
   - пустая детекция → PipelineError с explain_unmatched (как в run_cleanup)
2. application/use_cases/run_preview.py: класс с теми же портами + `write_image` для оверлеев; отрисовка боксов/масок OpenCV; maskfilm.mp4 через существующий FFmpegMedia (image2pipe или concat demuxer на файлах — выбрать при реализации, артефакт опционален при ошибке кодирования: превью не должно падать целиком).
3. composition.py: `build_run_preview(cfg, progress, jobs, job_id)`.
4. Коммит: `feat: RunPreview use case with mask overlay artifacts`.

## Шаг 5. CLI

1. tests/test_cli_serve_helpers.py:
   - `videoclean preview --frames 0:64:4 --prompt ...` — SmokeFakeFFmpeg (как существующие), артефакты в outdir
   - `--queries "text [bottom], logo"` строит таргеты без LLM
   - `--queries` без `--prompt` — валидно
2. cli.py: команда `preview` (input, output-dir, prompt, frames, queries, + общие флаги детектора/сегментатора/LLM).
3. Коммит: `feat: videoclean preview command`.

## Шаг 6. targets_override в RunCleanup

1. tests/test_run_cleanup.py: запрос с `targets_override` не вызывает парсер (фейк-парсер с фиксацией вызовов), Intent строится из dict.
2. application/use_cases/run_cleanup.py + manage_jobs.py: поле request, билд Intent (shared helper с CLI --queries).
3. Коммит: `feat: targets override skips prompt parsing in full run`.

## Шаг 7. API

1. tests (fastapi TestClient):
   - POST /api/preview создаёт preview-джобу (kind=preview в request_json)
   - mode=detect требует targets
   - GET /api/jobs/{id}/preview/preview.json и картинки — auth, 404 на чужие пути (path traversal guard на имени файла)
2. fastapi_app.py + service.py: эндпоинты, queue_preview_job.
3. Коммит: `feat: preview API endpoints`.

## Шаг 8. UI

1. index.html: панель «Превью» (поля start/count/stride, кнопки «Превью: разбор» / «Детекция» / «Перепарсить»), карточка превью: лента side-кадров, тумблеры, плеер, таблица таргетов с редактированием, кнопки «Детекция» и «Полный прогон».
2. app.js: submit превью, рендер карточки, редактор таргетов, проброс в полный прогон.
3. Проверка вручную: сервер локально, прогон на example_2_480.mp4.
4. Коммит: `feat: preview panel and job card in web UI`.

## Шаг 9. Документация

1. docs/PARAMS.md: `parse_chunk_frames`, глава «Превью».
2. README.md: раздел Preview, `--queries`, пример.
3. Коммит: `docs: preview mode and scoped targets`.

## Шаг 10. Финал

1. `uv run pytest -q` зелёный.
2. Локальный e2e: CLI preview на example_2_480.mp4, затем полный прогон с targets_override.
3. Проверка doctor: `parse-chunk` в выводе.
