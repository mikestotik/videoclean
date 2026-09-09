# Полное удаление OWL-ViT из videoclean

**Goal:** вырезать OWL-ViT целиком — адаптер, конфиг, CLI, web-UI, каталог моделей, тесты, документацию. Единственный детектор после удаления — `grounding-dino`.

**Основание:** Grounding DINO устойчиво точнее на zero-shot (52.5 AP против ~27–32 у OWL-ViT), OWL-ViT держали только как «резерв в цепочке». Решение — не поддерживать два детектора.

**Проверка контекста:** `RunCleanup.execute` вызывает `cfg.validate()` (run_cleanup.py:57), поэтому в тестах, где конфиг собирается с `detectors=["owlvit"]`, валидация после удаления упадёт — их правим в том же коммите.

---

## Инвентаризация (все упоминания owlvit/OWL-ViT в репо)

### Удалить целиком
- `videoclean/adapters/detectors/owlvit.py` — класс `OwlVitDetector`, `fit_owlvit_queries`, `split_visual_phrases`, `BoxHit`, `OWL_VIT_MAX_QUERY_TOKENS`. Всё используется только здесь (+ тесты). `BoxHit` в `grounding_dino.py` — своя копия, не импортирует из owlvit.
- `tests/test_owlvit_queries.py` — файл целиком.

### Правки кода
| Файл | Что менять |
|---|---|
| `videoclean/application/config.py` | `DETECTORS` → `("grounding-dino",)`; `READY_ADAPTERS["detector"]` → `frozenset({"grounding-dino"})`; удалить константу `DEFAULT_DETECTOR_MODEL` (строка 23); `PORT_HELP` строка detector: `"grounding-dino | owlvit"` → `"grounding-dino"`; удалить запись `owlvit` из `BACKENDS` (строки 61–69); комментарий строки 185: `(dino 8, owlvit 12)` → `(dino 8)` |
| `videoclean/composition.py` | удалить импорты `OwlVitDetector` (стр. 11) и `DEFAULT_DETECTOR_MODEL` (стр. 23); в `_detector_model_id` убрать ветку owlvit и guard `"owlvit" not in mid.lower()` → оставить `if mid: return mid; return DEFAULT_GROUNDING_DINO_MODEL`; в `make_detector` удалить ветку `if name == "owlvit":` (стр. 165–176) |
| `videoclean/adapters/detectors/__init__.py` | убрать импорт `OwlVitDetector` и из `__all__` |
| `videoclean/cli.py` | help-тексты: `_device_opt` (стр. 159, убрать `OWL-ViT`); `_detector_opt` (стр. 167, `grounding-dino | owlvit` → `grounding-dino`); `_detector_model_opt` (стр. 175, убрать `or owlvit`); `--detector-threshold` (стр. 414, убрать `and owlvit`); `--detector-keyframes` (стр. 445, `grounding-dino 8, owlvit 12` → `grounding-dino 8`) |
| `videoclean/adapters/web/service.py` | удалить импорт `DEFAULT_DETECTOR_MODEL` (стр. 20); строки 77–80 → `detector_model = detector_model or DEFAULT_GROUNDING_DINO_MODEL` (если пусто) |
| `videoclean/adapters/web/static/app.js` | строка 224: `detector: ["grounding-dino", "owlvit"]` → `["grounding-dino"]` |
| `videoclean/adapters/web/static/index.html` | строка 150: placeholder `8 (dino) / 12 (owlvit)` → `8 (dino)` |
| `videoclean/adapters/models/catalog.py` | удалить `ComponentInfo(id="detector:owlvit", ...)` (стр. 37–43) и маппинг `("detector", "owlvit"): "detector:owlvit"` (стр. 105) |
| `videoclean/adapters/prompt/llm.py` | строки 31 и 51 (инлайн SYSTEM/VISION_SYSTEM): `Grounding DINO / OWL-ViT` → `Grounding DINO` |
| `videoclean/adapters/prompt/prompts/system.md` | строка 1: `Grounding DINO / OWL-ViT` → `Grounding DINO` |
| `videoclean/adapters/prompt/prompts/vision_system.md` | строка 2: то же |

### Правки тестов
| Файл | Что менять |
|---|---|
| `tests/test_owlvit_queries.py` | **удалить файл** |
| `tests/test_hf_cache.py` | убрать импорт `OwlVitDetector` (стр. 6) и тест `test_owlvit_status_uses_hf_home` (стр. 50–60). Покрытие HF_HOME для детектора не теряем: `_status()` логика общая, проверяется в `test_hf_hub_dir_honors_hf_home`. Опционально — переписать тест на `GroundingDinoDetector` с `IDEA-Research/grounding-dino-tiny` |
| `tests/test_tunables.py` | убрать импорт (стр. 4) и блок `owlcfg/owl` (стр. 86–89); dino-часть теста остаётся |
| `tests/test_config.py` | `test_detector_chain` (стр. 37–39): `parse_name_list("owlvit,grounding-dino", ...)` после удаления бросит PipelineError — заменить на дедуп-тест `parse_name_list("grounding-dino,grounding-dino", ...)` → `["grounding-dino"]`; комментарий стр. 82 → `dino 8` |
| `tests/test_run_cleanup.py` | фейковые детекторы `BoomDetector`/`CancelDetector`/`CountingDetector`: `name = "owlvit"` → `"grounding-dino"` (стр. 147, 191, 258); конфиги `detectors=["owlvit", ...]` → `["grounding-dino", ...]` (стр. 180, 224, 315); ассерты `"owlvit: error"` → `"grounding-dino: error"` (стр. 187), `detectorUsed == "owlvit"` → `"grounding-dino"` (стр. 321). NB: `execute()` вызывает `cfg.validate()`, без правки конфигов тесты упадут |
| `tests/test_model_catalog.py` | `expected` без `"detector:owlvit"`, `len(COMPONENT_IDS) == 7` (стр. 22–35); `component_id_for("detector", "owlvit")` — заменить на `is None` (стр. 56); `backend_ready("detector", "owlvit")` — заменить имя на заведомо неизвестное (стр. 77); все `"detector:owlvit"` в download-тестах → `"detector:grounding-dino"`, `google/owlvit-base-patch32` → `IDEA-Research/grounding-dino-tiny` (стр. 86–90, 138–148, 164, 180–181, 212–213); в `test_run_download_dispatches_hf` ожидание `seen == ["IDEA-Research/grounding-dino-tiny"]` |
| `tests/test_cli_serve_helpers.py` | `models download detector:owlvit` → `detector:grounding-dino` (стр. 157–160) |

### Документация
| Файл | Что менять |
|---|---|
| `README.md` | стр. 35 (убрать OWL-ViT из перечня); стр. 42 (`--detector` без owlvit); стр. 43 (убрать `google/owlvit-base-patch32`); стр. 44 (порог «для grounding-dino»); стр. 60 (`dino 8`); стр. 163 (удалить `huggingface-cli download google/owlvit-base-patch32`); стр. 192 (список адаптеров без owlvit) |
| `docs/MODELS.md` | удалить строку `google/owlvit-base-patch32` из таблицы детекторов (стр. 10) |
| `docs/PARAMS.md` | стр. 12: `--detector` без owlvit и без «dino точнее / owlivit быстрее грузится»; стр. 37: `dino: 8` |
| `docs/RUNPOD.md` | стр. 144: убрать `detector:owlvit` из optional |

---

## Порядок выполнения (шаги)

### Задача 1 — ядро: конфиг + компоновка + адаптер
1. Удалить `videoclean/adapters/detectors/owlvit.py` и `tests/test_owlvit_queries.py`.
2. Правки в `config.py`, `composition.py`, `detectors/__init__.py` по таблице выше.
3. Правки в `tests/test_tunables.py`, `tests/test_config.py`, `tests/test_run_cleanup.py`.
4. Прогон: `uv run pytest tests/test_config.py tests/test_tunables.py tests/test_run_cleanup.py -q` → PASS.
5. Коммит: `refactor: remove owlvit detector from config, composition and tests`

### Задача 2 — CLI и web
1. Правки в `cli.py`, `web/service.py`, `web/static/app.js`, `web/static/index.html`.
2. Прогон: `uv run pytest tests/test_cli_serve_helpers.py -q` и `uv run pytest -q` → PASS.
3. Коммит: `refactor: remove owlvit from CLI and web UI`

### Задача 3 — каталог моделей и загрузки
1. Правки в `models/catalog.py`.
2. Правки в `tests/test_model_catalog.py`, `tests/test_cli_serve_helpers.py`.
3. Прогон: `uv run pytest tests/test_model_catalog.py tests/test_cli_serve_helpers.py -q` → PASS.
4. Коммит: `refactor: drop detector:owlvit from model catalog`

### Задача 4 — промпты
1. `prompt/llm.py` (стр. 31, 51), `prompts/system.md`, `prompts/vision_system.md`.
2. Прогон: `uv run pytest tests/ -q -k "prompt or llm"` → PASS.
3. Коммит: `chore: mention only Grounding DINO in parser prompts`

### Задача 5 — документация
1. `README.md`, `docs/MODELS.md`, `docs/PARAMS.md`, `docs/RUNPOD.md` по таблице.
2. Коммит: `docs: remove owlvit references`

### Задача 6 — финальная верификация
1. `grep -ri owlvit .` (вне `.git`, `docs/superpowers/plans`) → 0 совпадений.
2. `uv run pytest -q` → весь набор зелёный.
3. `uv run videoclean doctor` → таблица backends без owlvit, детектор один.
4. `uv run videoclean backends` → без строки owlvit.

---

## Что НЕ трогаем
- Зависимость `transformers` в `pyproject.toml` — нужна для grounding-dino и sam2.
- `_cv.py` — трекинг/NMS используется обоими детекторами.
- `DEFAULT_GROUNDING_DINO_MODEL` — становится единственным дефолтом детектора.
- История в `docs/BUG-001-removal-quality.md` — не упоминает owlvit.
