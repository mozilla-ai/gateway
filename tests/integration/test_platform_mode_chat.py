from collections.abc import Generator
from typing import Any

import httpx
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_PLATFORM_TOKEN", "gw_test_token")
    app = create_app(
        GatewayConfig(
            mode="platform",
            platform={"base_url": "http://platform.test/api/v1"},
        )
    )

    with TestClient(app) as client:
        yield client

    reset_config()
    reset_db()


def test_platform_mode_requires_authorization_header(platform_client: TestClient) -> None:
    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing authentication token"}


def test_platform_mode_maps_resolve_unauthorized(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid user token"})

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid user token"}


def test_platform_mode_sets_correlation_id_and_reports_usage(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        assert kwargs["model"] == "openai:gpt-4o-mini"
        assert kwargs["api_key"] == "sk-platform-key"
        return ChatCompletion(
            id="chatcmpl-platform",
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
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
        )

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68"
    assert usage_reports == [
        {
            "correlation_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
            "status": "success",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17,
            },
        }
    ]


def test_platform_mode_accepts_legacy_resolve_shape(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older otari (pre-fallback) returns a flat resolve payload.

    Gateway must accept it and treat it as a single-attempt route so deployments
    where the platform side hasn't been upgraded yet still work.
    """
    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                    "correlation_id": "9b2cce4a-5e91-4c19-9ad5-17a83f72b001",
                },
            )

        usage_reports.append(body)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-legacy",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    # Gateway maps the legacy correlation_id onto attempt_id, so X-Correlation-ID
    # still carries the same value as before.
    assert response.headers["X-Correlation-ID"] == "9b2cce4a-5e91-4c19-9ad5-17a83f72b001"
    assert usage_reports[0]["correlation_id"] == "9b2cce4a-5e91-4c19-9ad5-17a83f72b001"
    assert usage_reports[0]["status"] == "success"


def test_platform_mode_maps_provider_timeout(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "41a9667f-0af7-4ddf-8468-65c5f5c2af57",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "41a9667f-0af7-4ddf-8468-65c5f5c2af57",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        raise TimeoutError("provider timeout")

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "LLM provider timeout"}


def test_platform_mode_propagates_resolve_rate_limit_retry_after(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Rate limited"}, headers={"Retry-After": "11"})

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "11"
    assert response.json() == {"detail": "Rate limited"}


def test_platform_mode_usage_retries_only_transient_failures(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_calls: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "e655dc9a-6d90-4207-b371-f58d521a7a81",
                    "fallback_enabled": False,
                    "attempts": [
                        {
                            "attempt_id": "e655dc9a-6d90-4207-b371-f58d521a7a81",
                            "position": 0,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-platform-key",
                            "api_base": "https://api.openai.com/v1",
                            "managed": True,
                        }
                    ],
                },
            )

        usage_calls.append(body)
        if len(usage_calls) == 1:
            return httpx.Response(500)
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-platform",
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
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )
    assert response.status_code == 200
    assert len(usage_calls) == 2


def test_platform_mode_maps_resolve_validation_error_to_bad_gateway(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        return httpx.Response(422, json={"detail": "missing headers"})

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)

    response = platform_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Authorization service unavailable"}


# ---------------------------------------------------------------------------
# Streaming fallback (v1.1)
# ---------------------------------------------------------------------------


def test_platform_mode_streaming_falls_through_on_first_attempt_failure(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming request whose first attempt errors before any chunk → falls
    through to the second attempt; client sees a clean 200 SSE stream from
    the second provider."""
    from collections.abc import AsyncIterator

    from any_llm.types.completion import ChatCompletionChunk

    usage_reports: list[dict[str, Any]] = []

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "stream-req-1",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "stream-att-anthropic",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-ant-broken",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "stream-att-openai",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-openai-real",
                            "api_base": "https://api.openai.com/v1",
                            "managed": False,
                        },
                    ],
                },
            )
        usage_reports.append(body)
        return httpx.Response(204)

    calls: list[str] = []

    class _FakeApiStatusError(Exception):
        # status_code on the exception is what _classify_upstream_error reads;
        # 401 is in _FALLBACK_RETRYABLE_STATUS_CODES so the gateway will move
        # on to the next attempt.
        status_code = 401

    async def fake_acompletion(**kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        calls.append(model)
        if "anthropic" in model:
            raise _FakeApiStatusError("simulated upstream 401")

        async def _success_stream() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
                }
            )
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chunk-2",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            )

        return _success_stream()

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "stream-att-openai"
    # StreamingResponse builds its own response object, so X-Otari-Request-ID
    # has to be set in the StreamingResponse headers directly — assigning to
    # the dependency-injected Response object doesn't propagate.
    assert response.headers["X-Otari-Request-ID"] == "stream-req-1"
    # Both attempts were tried in order — anthropic first, then openai succeeded.
    assert [m for m in calls if "anthropic" in m or "openai" in m] == [
        "anthropic:claude-haiku-4-5",
        "openai:gpt-4o-mini",
    ]
    # The body should be a valid SSE stream from openai.
    body = response.text
    assert "data:" in body
    assert "hi" in body

    # The failed anthropic attempt should have reported an error to the platform.
    error_reports = [r for r in usage_reports if r.get("status") == "error"]
    assert len(error_reports) == 1
    assert error_reports[0]["correlation_id"] == "stream-att-anthropic"


def test_platform_mode_streaming_returns_502_when_all_attempts_fail(
    platform_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every attempt fails before yielding, the gateway returns 502 with
    the multi-attempt error wording instead of starting an SSE stream."""

    async def fake_post_platform(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_seconds: float,
    ) -> httpx.Response:
        if url.endswith("/gateway/provider-keys/resolve"):
            return httpx.Response(
                200,
                json={
                    "request_id": "stream-req-fail",
                    "fallback_enabled": True,
                    "attempts": [
                        {
                            "attempt_id": "att-a",
                            "position": 0,
                            "provider": "anthropic",
                            "model": "claude-haiku-4-5",
                            "api_key": "sk-ant-broken",
                            "api_base": None,
                            "managed": False,
                        },
                        {
                            "attempt_id": "att-b",
                            "position": 1,
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-openai-broken",
                            "api_base": None,
                            "managed": False,
                        },
                    ],
                },
            )
        return httpx.Response(204)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr("gateway.api.routes.chat._post_platform", fake_post_platform)
    monkeypatch.setattr("gateway.api.routes.chat.acompletion", fake_acompletion)

    response = platform_client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"Authorization": "Bearer user_test_token"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "All upstream providers failed"}
