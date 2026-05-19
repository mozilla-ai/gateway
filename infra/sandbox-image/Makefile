.PHONY: help install test lint build run clean

IMAGE_NAME ?= gateway-sandbox
IMAGE_TAG  ?= dev

help:
	@echo "sandbox-image targets:"
	@echo "  make install   - uv sync"
	@echo "  make test      - run pytest"
	@echo "  make lint      - ruff check + format"
	@echo "  make build     - docker build $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "  make run       - docker run -p 8080:8080"
	@echo "  make clean     - remove caches"

install:
	uv sync

test:
	uv run pytest -v

lint:
	uv run ruff check sandbox tests
	uv run ruff format --check sandbox tests

format:
	uv run ruff format sandbox tests
	uv run ruff check --fix sandbox tests

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	docker run --rm -p 8080:8080 $(IMAGE_NAME):$(IMAGE_TAG)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
