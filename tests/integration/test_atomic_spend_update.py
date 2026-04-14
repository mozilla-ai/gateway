"""Tests for atomic spend update via SQL expression."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes.chat import log_usage
from gateway.models.entities import ModelPricing, User
from gateway.services.log_writer import SingleLogWriter


@pytest.mark.asyncio
async def test_spend_update_uses_sql_expression(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _log_usage updates spend atomically via SQL, not Python read-modify-write."""
    # Set up user with initial spend
    user = User(user_id="atomic-user", spend=5.0)
    async_db.add(user)

    pricing = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=2.5,
        output_price_per_million=10.0,
    )
    async_db.add(pricing)
    await async_db.commit()

    usage = CompletionUsage(prompt_tokens=1_000_000, completion_tokens=100_000, total_tokens=1_100_000)

    writer = SingleLogWriter()

    @asynccontextmanager
    async def _session_cm() -> AsyncGenerator[AsyncSession, None]:
        yield async_db

    monkeypatch.setattr("gateway.services.log_writer.create_session", lambda: _session_cm())

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        user_id="atomic-user",
        usage_override=usage,
    )

    await async_db.refresh(user)
    updated_user = user
    assert updated_user is not None

    # Expected cost: (1M / 1M) * 2.5 + (100K / 1M) * 10.0 = 2.5 + 1.0 = 3.5
    expected_new_spend = 5.0 + 3.5
    assert abs(updated_user.spend - expected_new_spend) < 0.001, (
        f"Expected spend {expected_new_spend}, got {updated_user.spend}"
    )


@pytest.mark.asyncio
async def test_multiple_spend_updates_accumulate(async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that multiple sequential spend updates accumulate correctly."""
    user = User(user_id="multi-spend-user", spend=0.0)
    async_db.add(user)

    pricing = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=10.0,
        output_price_per_million=10.0,
    )
    async_db.add(pricing)
    await async_db.commit()

    writer = SingleLogWriter()

    @asynccontextmanager
    async def _session_cm() -> AsyncGenerator[AsyncSession, None]:
        yield async_db

    monkeypatch.setattr("gateway.services.log_writer.create_session", lambda: _session_cm())

    for _ in range(3):
        usage = CompletionUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        await log_usage(
            db=async_db,
            log_writer=writer,
            api_key_id=None,
            model="gpt-4o",
            provider="openai",
            endpoint="/v1/chat/completions",
            user_id="multi-spend-user",
            usage_override=usage,
        )

    await async_db.refresh(user)
    updated_user = user
    assert updated_user is not None

    # Each call: (1M/1M)*10 + (1M/1M)*10 = 20.0, x3 = 60.0
    assert abs(updated_user.spend - 60.0) < 0.001, f"Expected spend 60.0, got {updated_user.spend}"
