"""Backward-compatible aliases for legacy `core.*` imports in tests."""

import sys

from gateway.core import config, database

sys.modules[f"{__name__}.config"] = config
sys.modules[f"{__name__}.database"] = database

__all__ = ["config", "database"]
