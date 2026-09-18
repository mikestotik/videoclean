.DEFAULT_GOAL := help

VIDEOCLEAN_UI_USER ?= admin
VIDEOCLEAN_UI_PASSWORD ?= admin
export VIDEOCLEAN_UI_USER VIDEOCLEAN_UI_PASSWORD

.PHONY: help setup dev serve web test lint doctor docker clean

help: ## список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## разовая установка: python-зависимости + bun install
	# web = FastAPI/uvicorn; lama = simple-lama-inpainting. Together — иначе один --extra снимает другой.
	uv sync --extra web --extra lama
	cd webui && bun install

web: ## собрать фронт → server/static_dist/
	cd webui && bun run build

dev: ## дев-режим: FastAPI на :7860 + vite с hot-reload на :5173
	@command -v bun >/dev/null 2>&1 || { echo "bun не установлен: https://bun.sh"; exit 1; }
	@bash -euo pipefail -c '\
		uv run videoclean serve --port 7860 & \
		pid=$$!; \
		trap "kill $$pid 2>/dev/null || true" EXIT INT TERM; \
		cd webui && bun run dev'

serve: ## прод-режим: собрать фронт (если ещё нет) и поднять FastAPI на :7860
	@if [ ! -f server/static_dist/index.html ]; then $(MAKE) web; fi
	uv run videoclean serve --host 127.0.0.1 --port 7860

test: ## pytest
	uv run pytest -q

lint: ## eslint + tsc для webui
	cd webui && bun run lint && bun run typecheck

doctor: ## состояние бэкендов и моделей
	uv run videoclean doctor

docker: ## собрать образ (включая webui)
	docker build -t videoclean:local .

clean: ## билд-артефакты фронта
	rm -rf webui/dist server/static_dist
