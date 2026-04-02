"""Compatibility aliases for legacy `api.routes.*` imports in tests."""

import sys

from gateway.api.routes import (
    budgets,
    chat,
    embeddings,
    health,
    keys,
    messages,
    models,
    pricing,
    users,
)

chat = chat
messages = messages
embeddings = embeddings
users = users
keys = keys
budgets = budgets
pricing = pricing
models = models
health = health

sys.modules[f"{__name__}.chat"] = chat
sys.modules[f"{__name__}.messages"] = messages
sys.modules[f"{__name__}.embeddings"] = embeddings
sys.modules[f"{__name__}.users"] = users
sys.modules[f"{__name__}.keys"] = keys
sys.modules[f"{__name__}.budgets"] = budgets
sys.modules[f"{__name__}.pricing"] = pricing
sys.modules[f"{__name__}.models"] = models
sys.modules[f"{__name__}.health"] = health

__all__ = [
    "budgets",
    "chat",
    "embeddings",
    "health",
    "keys",
    "messages",
    "models",
    "pricing",
    "users",
]
