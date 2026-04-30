"""Moderation types.

Gateway-local shim that prefers ``any_llm.types.moderation`` when available
and falls back to locally defined Pydantic models when the installed
``any-llm-sdk`` does not yet ship moderation support.

This keeps the gateway importable against older SDK releases while
preserving the OpenAI-compatible wire contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModerationResult(BaseModel):
    """Single moderation result (one per input item).

    Mirrors the OpenAI moderation result shape. Extra provider-specific
    fields are preserved via ``model_config`` to avoid data loss.
    """

    model_config = ConfigDict(extra="allow")

    flagged: bool
    categories: dict[str, bool] = Field(default_factory=dict)
    category_scores: dict[str, float] = Field(default_factory=dict)
    category_applied_input_types: dict[str, list[str]] | None = None


class ModerationResponse(BaseModel):
    """OpenAI-compatible moderation response envelope."""

    model_config = ConfigDict(extra="allow")

    id: str
    model: str
    results: list[ModerationResult]
    raw: dict[str, Any] | None = None


# When a future any-llm-sdk release ships native moderation types, prefer
# those to stay wire-compatible. This runs at import time and is a no-op
# on older SDK releases.
try:  # pragma: no cover - exercised only when SDK exposes moderation types
    from any_llm.types.moderation import (  # type: ignore[import-not-found,no-redef]
        ModerationResponse,  # noqa: F811
        ModerationResult,  # noqa: F811
    )
except ImportError:
    pass


__all__ = ["ModerationResponse", "ModerationResult"]
