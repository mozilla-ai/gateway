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
    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "gw_test_token")
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
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                    "correlation_id": "7af2c39d-4eb8-4b3f-8242-46a97f7d5e68",
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
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                    "correlation_id": "41a9667f-0af7-4ddf-8468-65c5f5c2af57",
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
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-platform-key",
                    "api_base": "https://api.openai.com/v1",
                    "managed": True,
                    "correlation_id": "e655dc9a-6d90-4207-b371-f58d521a7a81",
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
