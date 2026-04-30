"""Integration tests for the POST /v1/rerank endpoint."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel


class _RerankUsage(BaseModel):
    """Mock RerankUsage for testing until SDK ships rerank types."""

    total_tokens: int | None = None


class _RerankResult(BaseModel):
    """Mock RerankResult for testing."""

    index: int
    relevance_score: float


class _RerankMeta(BaseModel):
    """Mock RerankMeta for testing."""

    billed_units: dict[str, float] | None = None
    tokens: dict[str, int] | None = None


class _RerankResponse(BaseModel):
    """Mock RerankResponse for testing."""

    id: str
    results: list[_RerankResult]
    meta: _RerankMeta | None = None
    usage: _RerankUsage | None = None


def _mock_rerank_response() -> _RerankResponse:
    """Create a mock RerankResponse for testing."""
    return _RerankResponse(
        id="rerank-test-123",
        results=[
            _RerankResult(index=0, relevance_score=0.95),
            _RerankResult(index=2, relevance_score=0.80),
            _RerankResult(index=1, relevance_score=0.30),
        ],
        meta=_RerankMeta(
            billed_units={"search_units": 1.0},
            tokens={"input_tokens": 100},
        ),
        usage=_RerankUsage(total_tokens=100),
    )


RERANK_PAYLOAD: dict[str, Any] = {
    "model": "cohere:rerank-v3.5",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital.", "Berlin is in Germany."],
}


def test_rerank_requires_auth(client: TestClient) -> None:
    """POST /v1/rerank requires authentication."""
    resp = client.post("/v1/rerank", json=RERANK_PAYLOAD)
    assert resp.status_code == 401


def test_rerank_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank works with API key authentication."""
    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        return_value=_mock_rerank_response(),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["relevance_score"] == 0.95


def test_rerank_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank with master key requires 'user' field."""
    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        return_value=_mock_rerank_response(),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=master_key_header)
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_rerank_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/rerank with master key + user field succeeds."""
    payload = {**RERANK_PAYLOAD, "user": test_user["user_id"]}
    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        return_value=_mock_rerank_response(),
    ):
        resp = client.post("/v1/rerank", json=payload, headers=master_key_header)
    assert resp.status_code == 200


def test_rerank_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank returns 500 when the provider fails."""
    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 500
    assert "provider" in resp.json()["detail"].lower()


def test_rerank_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/rerank creates a usage log entry."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        return_value=_mock_rerank_response(),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    rerank_logs = [log for log in logs if log["endpoint"] == "/v1/rerank"]
    assert len(rerank_logs) >= 1
    assert rerank_logs[0]["status"] == "success"
    assert rerank_logs[0]["prompt_tokens"] == 100
    assert rerank_logs[0]["completion_tokens"] == 0


def test_rerank_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/rerank logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/rerank" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


def test_rerank_empty_documents_422(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank returns 422 when documents list is empty."""
    payload = {**RERANK_PAYLOAD, "documents": []}
    resp = client.post("/v1/rerank", json=payload, headers=api_key_header)
    assert resp.status_code == 422


def test_rerank_top_n_zero_422(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank returns 422 when top_n is 0."""
    payload = {**RERANK_PAYLOAD, "top_n": 0}
    resp = client.post("/v1/rerank", json=payload, headers=api_key_header)
    assert resp.status_code == 422


def test_rerank_top_n_negative_422(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/rerank returns 422 when top_n is negative."""
    payload = {**RERANK_PAYLOAD, "top_n": -1}
    resp = client.post("/v1/rerank", json=payload, headers=api_key_header)
    assert resp.status_code == 422


@pytest.mark.parametrize("extra_field,value", [("top_n", 2), ("max_tokens_per_doc", 512)])
def test_rerank_optional_fields_forwarded(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
    value: int,
) -> None:
    """POST /v1/rerank forwards top_n and max_tokens_per_doc to the SDK."""
    mock = AsyncMock(return_value=_mock_rerank_response())
    payload = {**RERANK_PAYLOAD, extra_field: value}
    with patch("gateway.api.routes.rerank.arerank", mock):
        resp = client.post("/v1/rerank", json=payload, headers=api_key_header)
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get(extra_field) == value


def test_rerank_cost_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/rerank calculates cost when model pricing exists."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "cohere:rerank-v3.5",
            "input_price_per_million": 2.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.rerank.arerank",
        new_callable=AsyncMock,
        return_value=_mock_rerank_response(),
    ):
        resp = client.post("/v1/rerank", json=RERANK_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    logs = usage_resp.json()
    rerank_logs = [log for log in logs if log["endpoint"] == "/v1/rerank"]
    assert len(rerank_logs) >= 1
    assert rerank_logs[0]["cost"] is not None
    assert rerank_logs[0]["cost"] > 0
