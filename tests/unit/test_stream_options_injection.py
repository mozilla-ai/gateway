"""Unit tests for automatic stream_options injection behavior."""

from typing import Any

from gateway.api.routes.chat import ChatCompletionRequest


def test_stream_options_accepted_by_request_model() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    assert request.stream_options == {"include_usage": True}


def test_stream_options_defaults_to_none() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert request.stream_options is None


def test_stream_options_excluded_when_unset() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    dumped = request.model_dump(exclude_unset=True)
    assert "stream_options" not in dumped


def test_stream_options_included_when_set() -> None:
    request = ChatCompletionRequest(
        model="openai:gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": False},
    )
    dumped = request.model_dump(exclude_unset=True)
    assert dumped["stream_options"] == {"include_usage": False}


def _build_completion_kwargs(
    request_body: dict[str, Any],
    provider_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = ChatCompletionRequest(**request_body)
    request_fields = request.model_dump(exclude_unset=True)
    completion_kwargs: dict[str, Any] = {**(provider_kwargs or {}), **request_fields}

    if request.stream and completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}

    return completion_kwargs


def test_auto_injects_stream_options_for_streaming() -> None:
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
    )
    assert kwargs["stream_options"] == {"include_usage": True}


def test_no_injection_for_non_streaming() -> None:
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    assert "stream_options" not in kwargs


def test_preserves_client_stream_options() -> None:
    custom = {"include_usage": False, "custom_field": "value"}
    kwargs = _build_completion_kwargs(
        {
            "model": "openai:gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": custom,
        }
    )
    assert kwargs["stream_options"] == custom
