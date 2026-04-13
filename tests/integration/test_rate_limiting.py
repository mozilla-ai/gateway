"""Integration tests for request-level rate limiting behavior."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.db import Base, get_db
from gateway.main import create_app

from .conftest import _run_alembic_migrations, build_async_session_override


class _MockCompletionError(Exception):
    """Raised to short-circuit acompletion in integration tests."""


def _make_rate_limit_client(
    postgres_url: str,
    rate_limit_rpm: int | None,
) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        rate_limit_rpm=rate_limit_rpm,
    )
    _run_alembic_migrations(postgres_url)
    engine = create_engine(postgres_url, pool_pre_ping=True)
    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(postgres_url)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        dispose_override()
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


@pytest.fixture
def rate_limit_client(postgres_url: str) -> Generator[TestClient]:
    yield from _make_rate_limit_client(postgres_url, rate_limit_rpm=3)


@pytest.fixture
def no_rate_limit_client(postgres_url: str) -> Generator[TestClient]:
    yield from _make_rate_limit_client(postgres_url, rate_limit_rpm=None)


def _create_test_user(client: TestClient, master_key: str = "test-master-key") -> str:
    header = {API_KEY_HEADER: f"Bearer {master_key}"}
    resp = client.post("/v1/users", json={"user_id": "rl-test-user", "alias": "RL"}, headers=header)
    assert resp.status_code == 200
    result: str = resp.json()["user_id"]
    return result


def _chat_request(client: TestClient, user_id: str, master_key: str = "test-master-key") -> Any:
    header = {API_KEY_HEADER: f"Bearer {master_key}"}
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "user": user_id,
        },
        headers=header,
    )


def test_rate_limit_headers_on_success(rate_limit_client: TestClient) -> None:
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionMessage,
        Choice,
        CompletionUsage,
    )

    user_id = _create_test_user(rate_limit_client)

    mock_response = ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return mock_response

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _chat_request(rate_limit_client, user_id)

    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "3"
    assert resp.headers["X-RateLimit-Remaining"] == "2"
    assert "X-RateLimit-Reset" in resp.headers
    reset = int(resp.headers["X-RateLimit-Reset"])
    assert reset > 1_000_000_000


def test_rate_limit_returns_429_with_retry_after(rate_limit_client: TestClient) -> None:
    user_id = _create_test_user(rate_limit_client)

    async def mock_acompletion(**kwargs: Any) -> None:
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        for _ in range(3):
            _chat_request(rate_limit_client, user_id)

        resp = _chat_request(rate_limit_client, user_id)

    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_no_rate_limit_allows_unlimited(no_rate_limit_client: TestClient) -> None:
    user_id = _create_test_user(no_rate_limit_client)

    async def mock_acompletion(**kwargs: Any) -> None:
        raise _MockCompletionError

    with patch("gateway.api.routes.chat.acompletion", new=mock_acompletion):
        for _ in range(10):
            resp = _chat_request(no_rate_limit_client, user_id)
            assert resp.status_code != 429
