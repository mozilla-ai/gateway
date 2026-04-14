FROM python:3.14-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime

WORKDIR /app

RUN useradd -m -u 1000 gateway && chown gateway:gateway /app

COPY --from=builder --chown=gateway:gateway /app/.venv /app/.venv
COPY --chown=gateway:gateway src ./src
COPY --chown=gateway:gateway alembic ./alembic
COPY --chown=gateway:gateway alembic.ini ./alembic.ini

USER gateway

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ARG GATEWAY_VERSION=0.0.0-dev
ENV GATEWAY_VERSION=${GATEWAY_VERSION}
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8000

CMD ["gateway", "serve"]
