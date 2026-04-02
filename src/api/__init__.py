"""Backward-compatible aliases for legacy `api.*` imports in tests."""

import sys

from gateway.api import deps, main

from . import routes

sys.modules[f"{__name__}.deps"] = deps
sys.modules[f"{__name__}.main"] = main
sys.modules[f"{__name__}.routes"] = routes

__all__ = ["deps", "main", "routes"]
