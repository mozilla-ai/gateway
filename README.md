<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/main/docs/public/images/any-llm-logo-mark.png" width="18%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# gateway

[![Tests](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-tests.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-tests.yml)
[![Lint](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-lint.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-lint.yml)
[![Typecheck](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-typecheck.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-typecheck.yml)
[![Docker](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-docker.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-docker.yml)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

OpenAI-compatible LLM gateway with API key management, budget enforcement, and usage tracking.

</div>

## Why gateway?

`gateway` sits between your applications and LLM providers so you can control access, cost, and observability in one place.

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- Virtual API key management (`/v1/keys`) for safe client access
- User and budget controls (`/v1/users`, `/v1/budgets`)
- Usage and pricing tracking (`/v1/messages`, `/v1/pricing`)
- Health and metrics endpoints (`/health`, optional `/metrics`)

## Quickstart

### 1) Install

```bash
uv venv
source .venv/bin/activate
uv sync --dev
```

### 2) Configure

```bash
cp config.example.yml config.yml
```

Edit `config.yml` and set at least:

- `master_key`
- one provider credential in `providers` (for example `openai.api_key`)

### 3) Run

```bash
uv run gateway serve --config config.yml
```

Open API docs at `http://localhost:8000/docs`.

## Start in platform mode

Platform mode is enabled automatically when `ANY_LLM_PLATFORM_TOKEN` is set.

1) Export platform env vars:

```bash
export ANY_LLM_PLATFORM_TOKEN=gw_xxx
export PLATFORM_BASE_URL=https://your-platform.example/api/v1
```

2) Start the gateway:

```bash
uv run gateway serve --config config.yml
```

Notes:

- `GATEWAY_MODE` is optional; effective mode is derived from `ANY_LLM_PLATFORM_TOKEN`.
- If you explicitly set `GATEWAY_MODE=platform`, startup fails unless `ANY_LLM_PLATFORM_TOKEN` is also set.
- In platform mode, local `providers` configuration is not used.

## First request (OpenAI SDK)

On startup, the gateway can bootstrap an API key in logs. Export it as `GATEWAY_API_KEY`, then call the gateway as an OpenAI-compatible server:

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GATEWAY_API_KEY"],
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Hello from gateway"}],
)

print(response.choices[0].message.content)
```

## Local development

Run with hot reload and `.env`:

```bash
cp .env.example .env
make dev
```

## Tests and checks

```bash
make test
make lint
make typecheck
```

Run a single test file:

```bash
uv run pytest tests/unit/test_gateway_cli.py -v
```

## Docker

The gateway image is published on [Docker Hub](https://hub.docker.com/r/mzdotai/gateway).

### Run with docker compose (gateway + PostgreSQL)

```bash
cp config.example.yml config.yml
docker compose up -d
```

### Run with docker only

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/config.yml:/app/config.yml:ro" \
  mzdotai/gateway:latest \
  gateway serve --config /app/config.yml
```

Gateway will be available at `http://localhost:8000`.

## API surface

- `GET /health`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models`
- `POST/GET /v1/keys`
- `POST/GET /v1/users`
- `POST/GET /v1/budgets`
- `GET /v1/messages`
- `GET /v1/pricing`

Full schema: `docs/public/openapi.json`

## Useful CLI commands

```bash
uv run gateway init-db --config config.yml
uv run gateway migrate --config config.yml
uv run gateway migrate --config config.yml --revision <rev>
uv run python scripts/generate_openapi.py --check
```

## License

Apache 2.0. See `LICENSE`.
