FROM python:3.13-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY . .

RUN useradd -m -u 1000 gateway && \
    chown -R gateway:gateway /app

USER gateway

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8000

CMD ["uv", "run", "gateway", "serve"]
