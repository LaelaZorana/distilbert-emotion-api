# Convenience targets. Everything defaults to OFFLINE=1 so it runs with no
# model download and no network access.

.DEFAULT_GOAL := help
PORT ?= 8000
OFFLINE ?= 1
export OFFLINE

PY ?= python3

.PHONY: help install install-ml install-dev demo serve run test lint \
        bench bench-table docker-build docker-run compose-up compose-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install lean runtime deps (offline-capable)
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime + test deps
	$(PY) -m pip install -r requirements-dev.txt

install-ml: ## Install the heavy ML stack for a REAL model run (OFFLINE=0)
	$(PY) -m pip install -r requirements-ml.txt

demo: serve ## Serve the API + demo UI locally (alias for `serve`)

serve: ## Run the API (UI at /demo, docs at /docs, metrics at /metrics)
	@echo "API + UI on http://localhost:$(PORT)/demo  (OFFLINE=$(OFFLINE))"
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

run: serve ## Alias for `serve`

test: ## Run the test suite offline
	$(PY) -m pytest -q

bench: ## Run the load test (self-hosted, offline) and print a summary
	$(PY) scripts/loadtest.py --requests 3000 --concurrency 8 --warmup 100

bench-table: ## Print the README benchmark table row (markdown)
	$(PY) scripts/loadtest.py --requests 3000 --concurrency 8 --warmup 100 --markdown

docker-build: ## Build the slim runtime image
	docker build -t distilbert-emotion-api:local .

docker-run: docker-build ## Run the image locally on $(PORT)
	docker run --rm -e OFFLINE=$(OFFLINE) -p $(PORT):8000 distilbert-emotion-api:local

compose-up: ## Start API + Prometheus + Grafana
	docker compose up --build

compose-down: ## Stop the observability stack
	docker compose down

clean: ## Remove caches
	rm -rf .pytest_cache **/__pycache__ .ruff_cache
