from fastapi import FastAPI

from gateway.api.routes import (
    budgets,
    chat,
    embeddings,
    health,
    keys,
    messages,
    models,
    platform_mode,
    pricing,
    responses,
    usage,
    users,
)
from gateway.core.config import GatewayConfig


def register_routers(app: FastAPI, config: GatewayConfig) -> None:
    app.include_router(chat.router)
    app.include_router(health.router)

    if config.is_platform_mode:
        app.include_router(platform_mode.router)
        return

    app.include_router(messages.router)
    app.include_router(responses.router)
    app.include_router(embeddings.router)
    app.include_router(models.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(usage.router)
