PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PROJECT ?= semrag
IMAGE ?= $(PROJECT):dev

.PHONY: help install dev-install test lint format typecheck run-api \
        docker-build docker-run docker-shell coverage clean lock check lock-check

help:
	@echo "Targets:"
	@echo "  install       Install runtime deps from requirements.txt"
	@echo "  dev-install   Install runtime + dev deps"
	@echo "  lock          Regenerate requirements.txt from pyproject.toml"
	@echo "  test          Run unit tests"
	@echo "  coverage      Run tests with coverage report"
	@echo "  lint          Run ruff"
	@echo "  format        Run black in check mode"
	@echo "  typecheck     Run mypy"
	@echo "  check         Run lint + format + typecheck"
	@echo "  run-api       Start the FastAPI server"
	@echo "  docker-build  Build the production image"
	@echo "  docker-run    Run the production image"
	@echo "  clean         Remove caches and build artifacts"

install:
	$(PIP) install -r requirements.txt

dev-install:
	$(PIP) install -r requirements-dev.txt -e .

lock:
	$(PIP) install pip-tools
	pip-compile --strip-extras --generate-hashes --output-file=requirements.txt pyproject.toml
	pip-compile --strip-extras --output-file=requirements-dev.txt requirements-dev.in

test:
	$(PYTHON) -m pytest tests/unit -q

coverage:
	$(PYTHON) -m coverage run -m pytest tests/unit -q
	$(PYTHON) -m coverage report -m
	$(PYTHON) -m coverage xml -o coverage.xml

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m black --check src tests

typecheck:
	$(PYTHON) -m mypy src

check: lint format typecheck

run-api:
	$(PYTHON) -m uvicorn semrag.api.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	rm -f coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
