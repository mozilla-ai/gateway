"""Tests for timezone-aware datetime consistency."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes.chat import log_usage
from gateway.models.entities import UsageLog


@pytest.mark.asyncio
async def test_usage_log_timestamp_is_timezone_aware(async_db: AsyncSession) -> None:
    """Test that usage log timestamps are stored with timezone info."""

    class _Writer:
        def __init__(self) -> None:
            self.logs: list[UsageLog] = []

        async def put(self, log: UsageLog) -> None:
            self.logs.append(log)

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    writer = _Writer()
    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="test error",
    )

    assert writer.logs
    log = writer.logs[0]
    assert log is not None
    assert log.timestamp is not None
    # The timestamp should be timezone-aware (has tzinfo)
    assert log.timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
