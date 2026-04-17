"""Unit tests for the gateway's Bearer-token extraction.

These tests exercise ``_extract_bearer_token`` directly — a pure function over
``Request.headers`` — so they cover all precedence and malformed-input branches
without needing the full FastAPI stack or a database.
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api.deps import _extract_bearer_token
from gateway.core.config import API_KEY_HEADER, LEGACY_API_KEY_HEADER, GatewayConfig


def _make_request(headers: dict[str, str]) -> Request:
    """Build a minimal ASGI ``Request`` with the provided headers."""
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers.items()],
        "state": {},
    }
    return Request(scope)


@pytest.fixture
def config() -> GatewayConfig:
    """A minimal config object; ``_extract_bearer_token`` does not read fields from it."""
    return GatewayConfig(master_key="test-master-key", auto_migrate=False)


def test_canonical_header_returns_token(config: GatewayConfig) -> None:
    request = _make_request({API_KEY_HEADER: "Bearer token-canonical"})

    assert _extract_bearer_token(request, config) == "token-canonical"


def test_authorization_header_returns_token(config: GatewayConfig) -> None:
    request = _make_request({"Authorization": "Bearer token-auth"})

    assert _extract_bearer_token(request, config) == "token-auth"


def test_canonical_takes_precedence_over_authorization(config: GatewayConfig) -> None:
    request = _make_request(
        {
            API_KEY_HEADER: "Bearer canonical-wins",
            "Authorization": "Bearer authorization-loses",
        }
    )

    assert _extract_bearer_token(request, config) == "canonical-wins"


def test_legacy_header_returns_token(config: GatewayConfig) -> None:
    request = _make_request({LEGACY_API_KEY_HEADER: "Bearer token-legacy"})

    assert _extract_bearer_token(request, config) == "token-legacy"


def test_canonical_takes_precedence_over_legacy(config: GatewayConfig) -> None:
    request = _make_request(
        {
            API_KEY_HEADER: "Bearer canonical-wins",
            LEGACY_API_KEY_HEADER: "Bearer legacy-loses",
        }
    )

    assert _extract_bearer_token(request, config) == "canonical-wins"


def test_legacy_takes_precedence_over_authorization(config: GatewayConfig) -> None:
    request = _make_request(
        {
            LEGACY_API_KEY_HEADER: "Bearer legacy-wins",
            "Authorization": "Bearer authorization-loses",
        }
    )

    assert _extract_bearer_token(request, config) == "legacy-wins"


def test_malformed_legacy_header_raises_401(config: GatewayConfig) -> None:
    request = _make_request({LEGACY_API_KEY_HEADER: "NotBearer token-bad"})

    with pytest.raises(HTTPException) as exc_info:
        _extract_bearer_token(request, config)

    assert exc_info.value.status_code == 401
    assert "Bearer" in exc_info.value.detail


def test_malformed_canonical_header_raises_401(config: GatewayConfig) -> None:
    request = _make_request({API_KEY_HEADER: "NotBearer token-bad"})

    with pytest.raises(HTTPException) as exc_info:
        _extract_bearer_token(request, config)

    assert exc_info.value.status_code == 401
    assert "Bearer" in exc_info.value.detail


def test_malformed_authorization_header_raises_401(config: GatewayConfig) -> None:
    request = _make_request({"Authorization": "Basic abc123"})

    with pytest.raises(HTTPException) as exc_info:
        _extract_bearer_token(request, config)

    assert exc_info.value.status_code == 401


def test_missing_credentials_raises_401(config: GatewayConfig) -> None:
    request = _make_request({})

    with pytest.raises(HTTPException) as exc_info:
        _extract_bearer_token(request, config)

    assert exc_info.value.status_code == 401
    assert API_KEY_HEADER in exc_info.value.detail
