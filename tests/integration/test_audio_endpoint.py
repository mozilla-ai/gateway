"""Tests for the POST /v1/audio/transcriptions and POST /v1/audio/speech endpoints."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.audio import Transcription
from fastapi.testclient import TestClient


def _mock_transcription_response() -> Transcription:
    """Build a real Transcription for testing."""
    return Transcription(text="Hello, world!")


FAKE_AUDIO_BYTES = b"fake-audio-content-mp3"


# === Transcription tests ===


def test_transcription_requires_auth(client: TestClient) -> None:
    """POST /v1/audio/transcriptions requires authentication."""
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
        data={"model": "openai:whisper-1"},
    )
    assert resp.status_code == 401


def test_transcription_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/transcriptions works with API key authentication."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello, world!"


def test_transcription_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/transcriptions with master key requires 'user' field."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_transcription_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/audio/transcriptions with master key + user field succeeds."""
    mock_resp = _mock_transcription_response()
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", "user": test_user["user_id"]},
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_transcription_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/transcriptions returns 500 when the provider fails."""
    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 500
    assert "provider" in resp.json()["detail"].lower()


def test_transcription_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/audio/transcriptions creates a usage log entry."""
    mock_resp = _mock_transcription_response()
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    transcription_logs = [log for log in logs if log["endpoint"] == "/v1/audio/transcriptions"]
    assert len(transcription_logs) >= 1
    assert transcription_logs[0]["status"] == "success"
    assert transcription_logs[0]["prompt_tokens"] == 0


def test_transcription_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/audio/transcriptions logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.audio.atranscription",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1"},
            headers=api_key_header,
        )
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/audio/transcriptions" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["language", "prompt", "response_format", "temperature"])
def test_transcription_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /v1/audio/transcriptions forwards optional fields."""
    mock_resp = _mock_transcription_response()
    values = {"language": "en", "prompt": "Previous context", "response_format": "verbose_json", "temperature": "0.2"}
    with patch("gateway.api.routes.audio.atranscription", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.mp3", b"audio-data", "audio/mpeg")},
            data={"model": "openai:whisper-1", extra_field: values[extra_field]},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs


# === Speech tests ===


def test_speech_requires_auth(client: TestClient) -> None:
    """POST /v1/audio/speech requires authentication."""
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
    )
    assert resp.status_code == 401


def test_speech_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/speech works with API key authentication."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == FAKE_AUDIO_BYTES


def test_speech_content_type_matches_format(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/speech returns correct Content-Type for each format."""
    format_types = {
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    for fmt, expected_type in format_types.items():
        with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "openai:tts-1", "input": "Hi", "voice": "alloy", "response_format": fmt},
                headers=api_key_header,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == expected_type, f"Expected {expected_type} for {fmt}"


def test_speech_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/speech with master key requires 'user' field."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_speech_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/audio/speech with master key + user field succeeds."""
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "openai:tts-1",
                "input": "Hello",
                "voice": "alloy",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_speech_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/audio/speech returns 500 when the provider fails."""
    with patch(
        "gateway.api.routes.audio.aspeech",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 500
    assert "provider" in resp.json()["detail"].lower()


def test_speech_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/audio/speech creates a usage log entry."""
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    speech_logs = [log for log in logs if log["endpoint"] == "/v1/audio/speech"]
    assert len(speech_logs) >= 1
    assert speech_logs[0]["status"] == "success"
    assert speech_logs[0]["prompt_tokens"] == 0


def test_speech_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/audio/speech logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.audio.aspeech",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "openai:tts-1", "input": "Hello", "voice": "alloy"},
            headers=api_key_header,
        )
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/audio/speech" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["response_format", "speed", "instructions"])
def test_speech_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /v1/audio/speech forwards optional fields."""
    values = {"response_format": "opus", "speed": 1.5, "instructions": "Speak slowly"}
    with patch("gateway.api.routes.audio.aspeech", new_callable=AsyncMock, return_value=FAKE_AUDIO_BYTES) as mock:
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "openai:tts-1",
                "input": "Hello",
                "voice": "alloy",
                extra_field: values[extra_field],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs
    assert call_kwargs[extra_field] == values[extra_field]
