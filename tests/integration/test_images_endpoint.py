"""Tests for the POST /v1/images/generations endpoint."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.image import Image, ImagesResponse
from fastapi.testclient import TestClient


def _mock_images_response(n: int = 1) -> ImagesResponse:
    """Build a real ImagesResponse for testing."""
    return ImagesResponse(
        created=1700000000,
        data=[
            Image(
                url=f"https://example.com/image_{i}.png",
                revised_prompt="a cute cat sitting on a windowsill",
            )
            for i in range(n)
        ],
    )


def test_images_requires_auth(client: TestClient) -> None:
    """POST /v1/images/generations requires authentication."""
    resp = client.post(
        "/v1/images/generations",
        json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
    )
    assert resp.status_code == 401


def test_images_with_api_key(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/images/generations works with API key authentication."""
    mock_resp = _mock_images_response()
    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["url"].startswith("https://")


def test_images_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/images/generations with master key requires 'user' field."""
    mock_resp = _mock_images_response()
    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
            headers=master_key_header,
        )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_images_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/images/generations with master key + user field succeeds."""
    mock_resp = _mock_images_response()
    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/images/generations",
            json={
                "model": "openai:dall-e-3",
                "prompt": "a cute cat",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )
    assert resp.status_code == 200


def test_images_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """POST /v1/images/generations returns 500 when the provider fails."""
    with patch(
        "gateway.api.routes.images.aimage_generation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
            headers=api_key_header,
        )
    assert resp.status_code == 500
    assert "provider" in resp.json()["detail"].lower()


def test_images_logs_usage(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/images/generations creates a usage log entry."""
    mock_resp = _mock_images_response()
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    image_logs = [log for log in logs if log["endpoint"] == "/v1/images/generations"]
    assert len(image_logs) >= 1
    assert image_logs[0]["status"] == "success"
    assert image_logs[0]["prompt_tokens"] == 0
    assert image_logs[0]["completion_tokens"] == 0


def test_images_logs_error_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/images/generations logs an error entry when the provider fails."""
    user_id = api_key_obj["user_id"]

    with patch(
        "gateway.api.routes.images.aimage_generation",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat"},
            headers=api_key_header,
        )
    assert resp.status_code == 500

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    assert usage_resp.status_code == 200
    logs = usage_resp.json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/images/generations" and log["status"] == "error"]
    assert len(error_logs) >= 1
    assert "provider down" in error_logs[0]["error_message"]


@pytest.mark.parametrize("extra_field", ["n", "size", "quality", "style"])
def test_images_optional_fields(
    client: TestClient,
    api_key_header: dict[str, str],
    extra_field: str,
) -> None:
    """POST /v1/images/generations forwards optional fields."""
    mock_resp = _mock_images_response()
    values = {"n": 2, "size": "1792x1024", "quality": "hd", "style": "natural"}
    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp) as mock:
        resp = client.post(
            "/v1/images/generations",
            json={
                "model": "openai:dall-e-3",
                "prompt": "a cute cat",
                extra_field: values[extra_field],
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    call_kwargs = mock.call_args.kwargs
    assert extra_field in call_kwargs
    assert call_kwargs[extra_field] == values[extra_field]


def test_images_cost_tracked_with_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """POST /v1/images/generations calculates cost when model pricing exists."""
    # Set up pricing: input_price_per_million is repurposed as price-per-image
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:dall-e-3",
            "input_price_per_million": 0.04,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )

    mock_resp = _mock_images_response(n=2)
    user_id = api_key_obj["user_id"]

    with patch("gateway.api.routes.images.aimage_generation", new_callable=AsyncMock, return_value=mock_resp):
        resp = client.post(
            "/v1/images/generations",
            json={"model": "openai:dall-e-3", "prompt": "a cute cat", "n": 2},
            headers=api_key_header,
        )
    assert resp.status_code == 200

    usage_resp = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header)
    logs = usage_resp.json()
    image_logs = [log for log in logs if log["endpoint"] == "/v1/images/generations"]
    assert len(image_logs) >= 1
    assert image_logs[0]["cost"] is not None
    assert image_logs[0]["cost"] > 0
    # Cost should be n_images * input_price_per_million = 2 * 0.04 = 0.08
    assert image_logs[0]["cost"] == pytest.approx(0.08)
