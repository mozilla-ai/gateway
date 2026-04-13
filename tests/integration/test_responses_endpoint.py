"""Tests for the /v1/responses gateway endpoint."""

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int | None = None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens or input_tokens + output_tokens


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], usage: _FakeUsage | None = None) -> None:
        self._payload = payload
        self.usage = usage

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return self._payload


class _FakeStreamEvent:
    def __init__(self, event_type: str, payload: dict[str, Any], response: _FakeResponse | None = None) -> None:
        self.type = event_type
        self._payload = payload
        self.response = response

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        return json.dumps(self._payload)


@pytest.fixture
def responses_request_body(test_user: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "openai:gpt-4o-mini",
        "input": {"type": "text", "text": "Hello"},
        "user": test_user["user_id"],
    }


def _make_stream(events: list[_FakeStreamEvent]) -> AsyncIterator[_FakeStreamEvent]:
    async def _generator() -> AsyncIterator[_FakeStreamEvent]:
        for event in events:
            yield event

    return _generator()


def _make_failing_stream(
    events: list[_FakeStreamEvent],
    exc: Exception,
) -> AsyncIterator[_FakeStreamEvent]:
    async def _generator() -> AsyncIterator[_FakeStreamEvent]:
        for event in events:
            yield event
        raise exc

    return _generator()


def test_responses_endpoint_basic_completion(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Test basic non-streaming Responses call."""

    fake_usage = _FakeUsage(input_tokens=10, output_tokens=20)
    fake_payload = {"id": "resp_123", "output": [{"type": "message", "content": "Hello"}]}

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        return_value=_FakeResponse(fake_payload, fake_usage),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 200
    data = result.json()
    assert data["id"] == "resp_123"
    assert data["output"][0]["content"] == "Hello"


def test_responses_endpoint_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Master key authentication requires explicit user field."""

    responses_request_body.pop("user")

    with patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock) as mock_call:
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 400
    assert result.json()["detail"] == "When using master key, 'user' field is required in request body"
    mock_call.assert_not_called()


def test_responses_endpoint_rejects_unsupported_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Requests to providers without Responses support are rejected."""

    responses_request_body["model"] = "anthropic:claude-3-5"

    class _UnsupportedProvider:
        SUPPORTS_RESPONSES = False

    with (
        patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock) as mock_call,
        patch("gateway.api.routes.responses.AnyLLM.get_provider_class", return_value=_UnsupportedProvider),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 400
    assert "does not support" in result.json()["detail"]
    mock_call.assert_not_called()


def test_responses_endpoint_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Streaming responses emit SSE events and done marker."""

    responses_request_body["stream"] = True
    events = [
        _FakeStreamEvent("response.created", {"event": "created"}),
        _FakeStreamEvent(
            "response.completed",
            {"event": "completed"},
            response=_FakeResponse({}, usage=_FakeUsage(5, 7)),
        ),
    ]

    with patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=_make_stream(events)):
        resp = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = list(resp.iter_lines())

    assert "event: response.created" in lines
    assert "event: response.completed" in lines
    data_lines = [line for line in lines if line.startswith("data: ") and line != "data: [DONE]"]
    assert data_lines, "Expected at least one payload data line"
    json.loads(data_lines[0][6:])
    assert "data: [DONE]" in lines


def test_responses_endpoint_preserves_encrypted_reasoning_fields(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Ensure reasoning fields pass through untouched."""

    responses_request_body.update(
        {
            "reasoning": {"effort": "medium"},
            "include": ["reasoning.encrypted_content"],
        }
    )

    payload = {"id": "resp_reasoning", "reasoning": {"encrypted_content": "***"}}

    with patch(
        "gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=_FakeResponse(payload)
    ) as mock_call:
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 200
    assert result.json()["reasoning"]["encrypted_content"] == "***"

    kwargs = mock_call.await_args.kwargs
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert kwargs["reasoning"] == {"effort": "medium"}


def test_responses_endpoint_bearer_auth(
    client: TestClient,
    api_key_obj: dict[str, Any],
    responses_request_body: dict[str, Any],
) -> None:
    """Standard Bearer auth header is accepted."""

    headers = {API_KEY_HEADER: f"Bearer {api_key_obj['key']}"}

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        return_value=_FakeResponse({"id": "resp_token"}),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=headers)

    assert result.status_code == 200
    assert result.json()["id"] == "resp_token"


def test_responses_endpoint_provider_error_non_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Provider failures bubble up as HTTP 502 errors."""

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 502
    assert result.json() == {"detail": "LLM provider error"}


def test_responses_endpoint_provider_error_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Provider errors before the stream starts return HTTP error."""

    responses_request_body["stream"] = True

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 502
    assert result.json() == {"detail": "LLM provider error"}


def test_responses_endpoint_no_usage_data(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Responses without usage data still succeed."""

    payload = {"id": "resp_no_usage", "output": []}

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        return_value=_FakeResponse(payload, usage=None),
    ):
        result = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert result.status_code == 200
    assert result.json()["id"] == "resp_no_usage"


def test_responses_endpoint_streaming_mid_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """Streaming errors emit SSE error events without DONE marker."""

    responses_request_body["stream"] = True
    events = [_FakeStreamEvent("response.created", {"event": "created"})]

    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        return_value=_make_failing_stream(events, RuntimeError("stream boom")),
    ):
        resp = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert resp.status_code == 200
    lines = list(resp.iter_lines())
    assert "event: response.created" in lines
    assert "event: error" in lines
    assert any(line.startswith('data: {"error"') for line in lines)
    assert "data: [DONE]" not in lines
