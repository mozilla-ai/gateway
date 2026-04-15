"""Tests for the /v1/batches batch API endpoints."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.batch import BatchResult, BatchResultError, BatchResultItem
from fastapi.testclient import TestClient
from openai.types.batch import Batch
from openai.types.batch_request_counts import BatchRequestCounts


def _mock_batch(**overrides: Any) -> Batch:
    """Build a mock Batch object for testing."""
    defaults: dict[str, Any] = {
        "id": "batch_abc123",
        "completion_window": "24h",
        "created_at": 1714502400,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc123",
        "object": "batch",
        "status": "validating",
        "metadata": {"team": "ml-ops"},
        "request_counts": BatchRequestCounts(total=1, completed=0, failed=0),
    }
    defaults.update(overrides)
    return Batch(**defaults)


def _create_batch_body(**overrides: Any) -> dict[str, Any]:
    """Build a valid create batch request body."""
    defaults: dict[str, Any] = {
        "model": "openai:gpt-4o-mini",
        "requests": [
            {
                "custom_id": "req-1",
                "body": {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                },
            },
        ],
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# POST /v1/batches — Create batch
# ---------------------------------------------------------------------------


def test_create_batch_auth_required(client: TestClient) -> None:
    """POST /v1/batches requires authentication."""
    resp = client.post("/v1/batches", json=_create_batch_body())
    assert resp.status_code == 401


def test_create_batch_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches works with API key and returns provider field."""
    mock_batch = _mock_batch()

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    with (
        patch("gateway.api.routes.batches.acreate_batch", new_callable=AsyncMock, return_value=mock_batch) as mock_call,
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=api_key_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "batch_abc123"
    assert data["provider"] == "openai"
    assert data["status"] == "validating"
    assert data["endpoint"] == "/v1/chat/completions"

    # Verify SDK was called with correct params
    mock_call.assert_awaited_once()
    call_kwargs = mock_call.call_args
    assert call_kwargs.kwargs.get("provider") is not None or call_kwargs.args[0] is not None
    assert call_kwargs.kwargs.get("endpoint") == "/v1/chat/completions"


def test_create_batch_with_master_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/batches works with master key."""
    mock_batch = _mock_batch()

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    with (
        patch("gateway.api.routes.batches.acreate_batch", new_callable=AsyncMock, return_value=mock_batch),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=master_key_header)

    assert resp.status_code == 200
    assert resp.json()["id"] == "batch_abc123"


def test_create_batch_unsupported_provider(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches returns 422 for unsupported provider."""

    class _NoBatch:
        SUPPORTS_BATCH = False

    with patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_NoBatch):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=api_key_header)

    assert resp.status_code == 422
    assert "does not support batch operations" in resp.json()["detail"]


def test_create_batch_empty_requests(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches returns 422 for empty requests array."""
    resp = client.post(
        "/v1/batches",
        json=_create_batch_body(requests=[]),
        headers=api_key_header,
    )
    assert resp.status_code == 422


def test_create_batch_invalid_model_format(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches returns 400 when model has no provider prefix."""
    with patch(
        "gateway.api.routes.batches.AnyLLM.split_model_provider",
        side_effect=ValueError("Model must be in 'provider:model' format"),
    ):
        resp = client.post(
            "/v1/batches",
            json=_create_batch_body(model="gpt-4o-mini"),
            headers=api_key_header,
        )
    assert resp.status_code == 400
    assert "Invalid request" in resp.json()["detail"]


def test_create_batch_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches returns 502 when provider fails."""

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    with (
        patch(
            "gateway.api.routes.batches.acreate_batch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("provider down"),
        ),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=api_key_header)

    assert resp.status_code == 502
    assert resp.json()["detail"] == "LLM provider error"


def test_create_batch_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/batches creates a usage log entry."""
    mock_batch = _mock_batch()
    user_id = api_key_obj["user_id"]

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    with (
        patch("gateway.api.routes.batches.acreate_batch", new_callable=AsyncMock, return_value=mock_batch),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
    ):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=api_key_header)

    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    batch_logs = [log for log in logs if log["endpoint"] == "/v1/batches"]
    assert len(batch_logs) >= 1
    assert batch_logs[0]["status"] == "success"


def test_create_batch_temp_file_cleanup(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches cleans up temp file even on error."""
    created_files: list[str] = []

    class _SupportsBatch:
        SUPPORTS_BATCH = True

    original_unlink = os.unlink

    def tracking_unlink(path: str) -> None:
        created_files.append(path)
        original_unlink(path)

    with (
        patch(
            "gateway.api.routes.batches.acreate_batch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("provider down"),
        ),
        patch("gateway.api.routes.batches.AnyLLM.get_provider_class", return_value=_SupportsBatch),
        patch("gateway.api.routes.batches.os.unlink", side_effect=tracking_unlink),
    ):
        resp = client.post("/v1/batches", json=_create_batch_body(), headers=api_key_header)

    assert resp.status_code == 502
    # Temp file should have been cleaned up
    assert len(created_files) == 1
    assert created_files[0].endswith(".jsonl")


# ---------------------------------------------------------------------------
# GET /v1/batches/{batch_id} — Retrieve batch
# ---------------------------------------------------------------------------


def test_retrieve_batch(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/batches/{batch_id} retrieves batch status."""
    mock_batch = _mock_batch(status="in_progress")

    with patch("gateway.api.routes.batches.aretrieve_batch", new_callable=AsyncMock, return_value=mock_batch):
        resp = client.get("/v1/batches/batch_abc123?provider=openai", headers=api_key_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "batch_abc123"
    assert data["status"] == "in_progress"
    assert data["provider"] == "openai"


def test_retrieve_batch_missing_provider(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/batches/{batch_id} without provider returns 422."""
    resp = client.get("/v1/batches/batch_abc123", headers=api_key_header)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/batches/{batch_id}/cancel — Cancel batch
# ---------------------------------------------------------------------------


def test_cancel_batch(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/batches/{batch_id}/cancel cancels a batch."""
    mock_batch = _mock_batch(status="cancelling")

    with patch("gateway.api.routes.batches.acancel_batch", new_callable=AsyncMock, return_value=mock_batch):
        resp = client.post("/v1/batches/batch_abc123/cancel?provider=openai", headers=api_key_header)

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "batch_abc123"
    assert data["status"] == "cancelling"
    assert data["provider"] == "openai"


# ---------------------------------------------------------------------------
# GET /v1/batches — List batches
# ---------------------------------------------------------------------------


def test_list_batches(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/batches lists batches for a provider."""
    mock_batches = [_mock_batch(id="batch_1"), _mock_batch(id="batch_2")]

    with patch(
        "gateway.api.routes.batches.alist_batches", new_callable=AsyncMock, return_value=mock_batches
    ) as mock_call:
        resp = client.get("/v1/batches?provider=openai&limit=10&after=cursor_abc", headers=api_key_header)

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "batch_1"
    assert data["data"][0]["provider"] == "openai"
    assert data["data"][1]["id"] == "batch_2"
    assert data["data"][1]["provider"] == "openai"

    # Verify pagination params were forwarded
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs.get("limit") == 10
    assert call_kwargs.get("after") == "cursor_abc"


# ---------------------------------------------------------------------------
# GET /v1/batches/{batch_id}/results — Retrieve batch results
# ---------------------------------------------------------------------------


def test_retrieve_batch_results(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/batches/{batch_id}/results returns per-request results."""
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {"id": "chatcmpl-1", "choices": []}

    mock_result = BatchResult(
        results=[
            BatchResultItem(custom_id="req-1", result=mock_completion, error=None),
            BatchResultItem(
                custom_id="req-2", result=None, error=BatchResultError(code="rate_limit", message="Rate limit exceeded")
            ),
        ]
    )

    with patch("gateway.api.routes.batches.aretrieve_batch_results", new_callable=AsyncMock, return_value=mock_result):
        resp = client.get("/v1/batches/batch_abc123/results?provider=openai", headers=api_key_header)

    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["custom_id"] == "req-1"
    assert data["results"][0]["result"] == {"id": "chatcmpl-1", "choices": []}
    assert data["results"][0]["error"] is None
    assert data["results"][1]["custom_id"] == "req-2"
    assert data["results"][1]["result"] is None
    assert data["results"][1]["error"]["code"] == "rate_limit"
    assert data["results"][1]["error"]["message"] == "Rate limit exceeded"


def test_retrieve_batch_results_not_complete(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """GET /v1/batches/{batch_id}/results returns 409 when batch is not complete."""
    with patch(
        "gateway.api.routes.batches.aretrieve_batch_results",
        new_callable=AsyncMock,
        side_effect=BatchNotCompleteError(batch_id="batch_abc123", status="in_progress"),
    ):
        resp = client.get("/v1/batches/batch_abc123/results?provider=openai", headers=api_key_header)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "not yet complete" in detail
    assert "in_progress" in detail
    assert "batch_abc123" in detail


def test_retrieve_batch_results_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """GET /v1/batches/{batch_id}/results creates a usage log entry."""
    mock_result = BatchResult(results=[])
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.batches.aretrieve_batch_results", new_callable=AsyncMock, return_value=mock_result):
        resp = client.get("/v1/batches/batch_abc123/results?provider=openai", headers=api_key_header)

    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    results_logs = [log for log in logs if log["endpoint"] == "/v1/batches/results"]
    assert len(results_logs) >= 1
    assert results_logs[0]["status"] == "success"


# ---------------------------------------------------------------------------
# Platform mode — batch endpoints not registered
# ---------------------------------------------------------------------------


def test_batch_endpoints_not_in_platform_mode(
    postgres_url: str,
    monkeypatch: Any,
) -> None:
    """Batch endpoints return 404 in platform mode."""
    from gateway.core.config import GatewayConfig
    from gateway.main import create_app

    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "test-platform-token")

    platform_config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        platform={"base_url": "http://localhost:9000"},
    )

    app = create_app(platform_config)
    with TestClient(app) as platform_client:
        resp = platform_client.post(
            "/v1/batches",
            json=_create_batch_body(),
            headers={"Authorization": "Bearer test-master-key"},
        )
        # In platform mode, batch routes are not registered, so we get 404
        # (chat is always registered but batches are standalone-only)
        assert resp.status_code == 404
