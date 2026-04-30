"""Tests for the POST /v1/moderations endpoint."""

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from gateway.types.moderation import ModerationResponse, ModerationResult


def _mock_moderation_response() -> ModerationResponse:
    """Build a real ModerationResponse for testing."""
    return ModerationResponse(
        id="modr-test",
        model="omni-moderation-latest",
        results=[
            ModerationResult(
                flagged=True,
                categories={"violence": True},
                category_scores={"violence": 0.93},
            )
        ],
    )


def test_moderations_requires_auth(client: TestClient) -> None:
    """POST /v1/moderations requires authentication."""
    resp = client.post(
        "/v1/moderations",
        json={"model": "openai:omni-moderation-latest", "input": "hello"},
    )
    assert resp.status_code == 401


def test_moderations_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations works with API key authentication."""
    mock_resp = _mock_moderation_response()
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "modr-test"
    assert len(data["results"]) == 1
    assert data["results"][0]["flagged"] is True


def test_moderations_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations with master key requires 'user' field."""
    mock_resp = _mock_moderation_response()
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_moderations_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/moderations with master key + user field succeeds."""
    mock_resp = _mock_moderation_response()
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={
                "model": "openai:omni-moderation-latest",
                "input": "hello",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_moderations_list_input(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations accepts a list of strings as input."""
    mock_resp = _mock_moderation_response()
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={
                "model": "openai:omni-moderation-latest",
                "input": ["hello", "world"],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200


def test_moderations_multimodal_input(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations accepts multimodal content-part dicts."""
    mock_resp = _mock_moderation_response()
    multimodal_input = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ) as mock:
        resp = client.post(
            "/v1/moderations",
            json={
                "model": "openai:omni-moderation-latest",
                "input": multimodal_input,
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert mock.call_args.kwargs["input"] == multimodal_input


def test_moderations_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations returns 500 when the provider fails."""
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 500
    assert "could not be completed" in resp.json()["detail"].lower()


def test_moderations_unsupported_provider_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Unsupported provider (NotImplementedError) returns 400 with the locked phrasing."""
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        side_effect=NotImplementedError("Provider anthropic does not support moderation"),
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "anthropic:claude-3-5-sonnet", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 400
    # Assert literal string — cross-repo error contract.
    assert resp.json() == {"detail": "Provider anthropic does not support moderation"}


def test_moderations_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/moderations creates a usage log entry on success."""
    mock_resp = _mock_moderation_response()
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    moderation_logs = [log for log in logs if log["endpoint"] == "/v1/moderations"]
    assert len(moderation_logs) >= 1
    assert moderation_logs[0]["status"] == "success"
    assert moderation_logs[0]["prompt_tokens"] is None
    assert moderation_logs[0]["completion_tokens"] == 0
    assert moderation_logs[0]["total_tokens"] is None
    assert moderation_logs[0]["provider"] == "openai"
    assert moderation_logs[0]["model"] == "omni-moderation-latest"


def test_moderations_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/moderations logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/moderations" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


def test_moderations_include_raw_opts_in(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/moderations forwards the include_raw query param to amoderation."""
    mock_resp = _mock_moderation_response()
    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ) as mock:
        resp = client.post(
            "/v1/moderations?include_raw=true",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert mock.call_args.kwargs["include_raw"] is True

    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ) as mock:
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert mock.call_args.kwargs["include_raw"] is False


def test_moderations_cost_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/moderations records a non-zero cost when pricing exists."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:omni-moderation-latest",
            "input_price_per_million": 2.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_moderation_response()
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.moderations.amoderation",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ):
        resp = client.post(
            "/v1/moderations",
            json={"model": "openai:omni-moderation-latest", "input": "hello"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    logs = usage_resp.json()
    moderation_logs = [log for log in logs if log["endpoint"] == "/v1/moderations"]
    assert len(moderation_logs) >= 1
    assert moderation_logs[0]["cost"] is not None
    assert moderation_logs[0]["cost"] > 0


def test_moderations_no_warning_when_pricing_missing(
    client: TestClient,
    api_key_header: dict[str, str],
    caplog: Any,
) -> None:
    """POST /v1/moderations does not emit the 'No pricing configured' warning."""
    mock_resp = _mock_moderation_response()
    with caplog.at_level(logging.WARNING, logger="gateway"):
        with patch(
            "gateway.api.routes.moderations.amoderation",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            resp = client.post(
                "/v1/moderations",
                json={"model": "openai:omni-moderation-latest", "input": "hello"},
                headers=api_key_header,
            )
    assert resp.status_code == 200
    offending = [r for r in caplog.records if "No pricing configured" in r.getMessage()]
    assert offending == []
