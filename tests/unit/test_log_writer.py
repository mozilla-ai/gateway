from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.models.entities import UsageLog
from gateway.services.log_writer import SingleLogWriter


@pytest.mark.asyncio
async def test_single_log_writer_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")

    @asynccontextmanager
    async def _session_cm() -> AsyncGenerator[AsyncMock, None]:
        yield session

    monkeypatch.setattr("gateway.services.log_writer.create_session", lambda: _session_cm())
    writer = SingleLogWriter()

    log = UsageLog(id="log", model="test-model", endpoint="/v1/test", status="success")
    await writer.put(log)

    session.rollback.assert_awaited()
