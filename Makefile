PYTHON ?= python

.PHONY: install install-dev test lint format serve train drift bench

install:
	$(PYTHON) -m pip install -r requirements-serving.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt ruff

test:
	$(PYTHON) -m pytest -rs

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

serve:
	uvicorn src.serving.app:app --reload

train:
	$(PYTHON) -m src.serving.train_baseline

drift:
	$(PYTHON) -m src.monitoring.drift --current data/processed/test.parquet

bench:
	$(PYTHON) -m src.serving.bench_latency
