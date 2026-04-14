"""Tests for usage logging via the log writer abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from any_llm.types.completion import CompletionUsage
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes.chat import log_usage
from gateway.models.entities import ModelPricing, UsageLog


@dataclass
class StubLogWriter:
    logs: list[UsageLog]

    def __init__(self) -> None:
        self.logs = []

    async def put(self, log: UsageLog) -> None:
        self.logs.append(log)

    async def start(self) -> None:  # pragma: no cover - not used
        return None

    async def stop(self) -> None:  # pragma: no cover - not used
        return None


@pytest.mark.asyncio
async def test_log_usage_records_usage_data(async_db: AsyncSession) -> None:
    pricing = ModelPricing(model_key="openai:gpt-4o", input_price_per_million=2.0, output_price_per_million=4.0)
    async_db.add(pricing)
    await async_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    writer = StubLogWriter()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.prompt_tokens == 1000
    assert log.completion_tokens == 500
    assert log.cost == pytest.approx((1000 / 1_000_000) * 2.0 + (500 / 1_000_000) * 4.0)
    assert log.status == "success"


@pytest.mark.asyncio
async def test_log_usage_records_error(async_db: AsyncSession) -> None:
    writer = StubLogWriter()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="Provider timeout",
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    assert log.status == "error"
    assert log.error_message == "Provider timeout"
