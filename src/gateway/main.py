from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from gateway.api.deps import set_config
from gateway.api.main import register_routers
from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, init_db
from gateway.rate_limit import RateLimiter
from gateway.services.bootstrap_service import bootstrap_first_api_key
from gateway.services.log_writer import LogWriter, NoopLogWriter, create_log_writer
from gateway.services.pricing_init_service import initialize_pricing_from_config
from gateway.version import __version__

_PUBLIC_PREFIXES = ("/health",)

_ROOT_TUTORIAL_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AI Gateway</title>
    <style>
      body {
        background: #efefef;
        color: #111827;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
        margin: 0;
      }
      main {
        margin: 16px auto;
        max-width: 860px;
        padding: 0 16px;
        line-height: 1.5;
      }
      h1 {
        font-size: 32px;
        margin: 6px 0 10px;
      }
      .sub {
        font-size: 20px;
        margin-bottom: 16px;
      }
      .link {
        font-size: 20px;
        font-weight: 700;
        color: #0f62fe;
        text-decoration: underline;
      }
      .note {
        font-size: 16px;
      }
      .block {
        background: #e5e7eb;
        border-radius: 2px;
        margin: 18px 0;
        padding: 18px;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        overflow-wrap: break-word;
      }
      code {
        color: #111827;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 14px;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>AI Gateway (Proxy Server)</h1>
      <div class="sub">
        <a class="link" href="https://mozilla-ai.github.io/any-llm/gateway/quickstart/">Gateway Quickstart</a>
      </div>

      <p class="note">
        On first startup, the gateway prints a bootstrap API key in logs. Export it as
        <code>GATEWAY_API_KEY</code> and use that value in your client.
      </p>

      <div class="block">
        <pre><code>export GATEWAY_API_KEY=YOUR_BOOTSTRAP_GATEWAY_KEY
</code></pre>
      </div>

      <div class="block">
        <pre><code>pip install openai
</code></pre>
      </div>

      <div class="block">
         <pre><code>import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GATEWAY_API_KEY"],
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
</code></pre>
      </div>
    </main>
  </body>
</html>
"""


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses.

    Sets standard security headers on every response, plus cache-control
    headers on non-health endpoints to prevent CDN/proxy caches from
    storing authenticated responses.
    """

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not request.url.path.startswith(_PUBLIC_PREFIXES):
            response.headers["Cache-Control"] = "private, no-store, no-cache"
            vary_values = {part.strip() for part in response.headers.get("Vary", "").split(",") if part.strip()}
            vary_values.add("Authorization")
            response.headers["Vary"] = ", ".join(sorted(vary_values))
        return response


def _validate_platform_config(config: GatewayConfig) -> None:
    config.validate_mode_selection()
    if not config.is_platform_mode:
        return
    if not config.platform.get("base_url"):
        msg = "platform.base_url is required when platform mode is active"
        raise ValueError(msg)
    if config.providers:
        msg = "Local provider credentials are not supported in platform mode"
        raise ValueError(msg)


def _create_lifespan(config: GatewayConfig):  # noqa: ANN201 - FastAPI contract
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_writer: LogWriter
        if config.is_platform_mode:
            log_writer = NoopLogWriter()
        else:
            init_db(config)
            async with create_session() as session:
                await bootstrap_first_api_key(config, session)
                await initialize_pricing_from_config(config, session)
            log_writer = create_log_writer(config.log_writer_strategy)

        await log_writer.start()
        app.state.log_writer = log_writer

        try:
            yield
        finally:
            await log_writer.stop()

    return lifespan


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application."""

    _validate_platform_config(config)
    set_config(config)

    app = FastAPI(
        title="any-llm-gateway",
        description="A clean FastAPI gateway for any-llm with API key management",
        version=__version__,
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None,
        lifespan=_create_lifespan(config),
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root_tutorial() -> str:
        return _ROOT_TUTORIAL_HTML

    app.add_middleware(SecurityHeadersMiddleware)

    if config.cors_allow_origins:
        allow_credentials = "*" not in config.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Authorization",
                "X-AnyLLM-Key",
                "x-api-key",
            ],
        )

    if config.enable_metrics:
        from gateway.metrics import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)

    if config.rate_limit_rpm is not None:
        app.state.rate_limiter = RateLimiter(config.rate_limit_rpm)
    else:
        app.state.rate_limiter = None

    app.state.gateway_mode = config.effective_mode

    register_routers(app, config)

    if config.enable_metrics:
        from gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
