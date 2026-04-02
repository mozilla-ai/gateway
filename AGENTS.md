# AGENTS.md

Guidance for agentic coding tools working in this repository.
Scope: entire repo (`/Users/tbille/Documents/mozilla.ai/move/gateway`).
## Project Snapshot
- Language/runtime: Python 3.13+.
- Package manager + task runner: `uv`.
- App type: FastAPI gateway service with SQLAlchemy + Alembic.
- Source root: `src/gateway`.
- Tests: `tests/integration` and `tests/unit`.
- Database: SQLite by default, PostgreSQL in integration tests.
## Setup Commands
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`
- Install deps (dev): `uv sync --dev`
- Install deps exactly as lockfile (CI-style): `uv sync --dev --frozen`
## Run Commands
- Run gateway from config: `uv run gateway serve --config config.yml`
- Run dev server (reload + `.env`): `make dev`
- Initialize DB schema: `uv run gateway init-db --config config.yml`
- Run migrations to head: `uv run gateway migrate --config config.yml`
- Run migrations to specific revision: `uv run gateway migrate --revision <rev>`
## Build / Packaging Commands
- Python package build backend is configured via `setuptools` in `pyproject.toml`.
- If you need a local package build artifact, use: `uv build`
- Docker local build/run: `docker compose up --build`
- CI Docker smoke check is implemented in `scripts/docker_liveness_check.sh`.
## Lint / Typecheck Commands
- Ruff is configured for linting (rules: `E`, `F`, `I`; line length: 120) in `pyproject.toml`.
- Run lint checks with `make lint` (or `uv run ruff check src tests scripts`).
- Ruff is also enforced in CI via `.github/workflows/gateway-lint.yml`.
- Primary static checks present in dev dependencies: `ruff`, `mypy`.
- Run type checks (if adding/maintaining typed modules): `uv run mypy src`
- If introducing a formatter/linter, keep changes in a separate PR unless requested.
## Test Commands
- Main local suite (matches Makefile):
  - `make test`
  - expands to `uv run pytest -v tests/integration tests/unit/test_gateway_*.py`
- CI-style tests with coverage + parallel:
  - `uv run pytest tests/integration tests/unit -v --cov --cov-report=xml --cov-append -n auto`
### Running a Single Test (important)
- Single test file:
  - `uv run pytest tests/unit/test_gateway_cli.py -v`
- Single test function via node id:
  - `uv run pytest tests/unit/test_gateway_cli.py::test_gateway_config_defaults_to_sqlite -v`
- Single integration test function:
  - `uv run pytest tests/integration/test_health.py::test_health_endpoint -v`
- Pattern-filtered run:
  - `uv run pytest tests/integration -k "budget and reset" -v`
### Test Environment Notes
- `pytest.ini` sets:
  - `timeout = 120`
  - `reruns = 5`
  - `reruns_delay = 10`
- Integration tests can use:
  - `TEST_DATABASE_URL` (if provided), or
  - Testcontainers PostgreSQL (`postgres:17`) if not provided.
- Running integration tests without Docker may fail when Testcontainers is required.
## OpenAPI Spec Commands
- Generate OpenAPI JSON:
  - `uv run python scripts/generate_openapi.py`
- Check OpenAPI is up to date (CI behavior):
  - `uv run python scripts/generate_openapi.py --check`
- Default output path:
  - `docs/public/openapi.json`
## Repository Conventions
### Imports
- Use grouped imports in this order:
  1) stdlib
  2) third-party
  3) local (`gateway.*`)
- Separate groups with one blank line.
- Prefer explicit imports over wildcard imports.
- Use `TYPE_CHECKING` for type-only imports when helpful (`routes/_helpers.py`).
### Formatting
- Follow existing style consistent with Black-like formatting:
  - trailing commas in multiline literals/calls,
  - one statement per line,
  - clear vertical spacing between top-level defs/classes.
- Keep docstrings concise and meaningful for public functions/classes.
- Avoid adding comments unless logic is non-obvious.
### Typing
- Add type hints to new/changed functions (project is strongly typed in practice).
- Prefer modern syntax:
  - `str | None` over `Optional[str]`
  - built-in generics like `list[str]`, `dict[str, Any]`
- Annotate SQLAlchemy fields with `Mapped[...]` in ORM models.
- For FastAPI dependencies, prefer `Annotated[..., Depends(...)]`.
### Naming
- `snake_case`: functions, variables, module names.
- `PascalCase`: classes, Pydantic models, SQLAlchemy entities.
- Constants: `UPPER_SNAKE_CASE` (`API_KEY_HEADER`, `_PUBLIC_PREFIXES`).
- Test files: `test_*.py`; test functions: `test_*`.
### FastAPI / API Patterns
- Define request/response schemas with Pydantic models near route handlers.
- Keep route prefixes/tags explicit on each router.
- Return typed response models instead of raw dicts when practical.
- Use HTTP status constants from `fastapi.status`.
### Database Patterns
- Use Alembic migrations; do not manually mutate live schemas.
- For write operations:
  - `db.commit()` in `try`,
  - `db.rollback()` on `SQLAlchemyError`,
  - re-raise mapped API/domain errors.
- Reuse repository helpers (e.g., `get_active_user`) for shared query logic.
### Error Handling
- Raise explicit `HTTPException` in API layer with clear `detail` messages.
- Preserve security posture: do not leak internals in public error responses.
- Domain-layer errors live in `src/gateway/exceptions/domain.py`.
- Prefer specific exceptions (`ValueError`, `SQLAlchemyError`) over broad `except Exception`.
### Logging
- Use module logger from `gateway.log_config`.
- Prefer structured/contextual log messages with `%s` formatting placeholders.
- Do not log secrets, tokens, or raw API keys (bootstrap exception is intentional one-time behavior).
## CI Rules to Mirror Locally
- Python version for CI: 3.13 (`.github/workflows/gateway-tests.yml`).
- Install deps with frozen lockfile in CI.
- Tests run with coverage and xdist in CI.
- OpenAPI spec freshness is enforced in CI (`--check`).
## Practical Agent Workflow
- Before coding: read nearby module + related tests.
- After coding: run the smallest relevant pytest node id first.
- Then run broader impacted tests.
- If API contract changed: regenerate or check OpenAPI spec.
- Keep diffs focused; avoid opportunistic refactors unless requested.

## Key Paths
- App entry and wiring: `src/gateway/main.py`
- CLI entrypoint: `src/gateway/cli.py`
- Config + env resolution: `src/gateway/core/config.py`
- DB init/session helpers: `src/gateway/core/database.py`
- API routers: `src/gateway/api/routes/`
- ORM models: `src/gateway/models/entities.py`
- Alembic migrations: `alembic/versions/`
- OpenAPI generator: `scripts/generate_openapi.py`

## Change Validation Checklist
- If you touched API routes or schemas, run relevant integration tests first.
- If you touched DB models/repositories, run related integration tests and migration paths.
- If you touched config loading, run config/env tests in `tests/integration`.
- If you touched CLI behavior, run `tests/unit/test_gateway_cli.py`.
- If you touched auth headers or key handling, run key-management and auth-related tests.
- If OpenAPI-affecting code changed, run `uv run python scripts/generate_openapi.py --check`.

## Notes for Agents
- Prefer minimal, targeted edits over broad refactors.
- Maintain import order and existing typing style in touched files.
- Preserve security-relevant behavior (header parsing, auth checks, error detail boundaries).
- Keep test additions close to changed behavior (unit for pure logic, integration for route/database behavior).
