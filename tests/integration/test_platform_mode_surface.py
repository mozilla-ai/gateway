import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


def test_platform_mode_starts_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="platform",
        database_url="postgresql://127.0.0.1:1/does-not-exist",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "mode": "platform"}

    reset_config()
    reset_db()


def test_platform_mode_disables_local_management_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_PLATFORM_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="platform",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        users_response = client.post("/v1/users", json={"user_id": "u1"})
        keys_response = client.get("/v1/keys")
        budgets_response = client.get("/v1/budgets")
        spend_response = client.get("/v1/spend")

    expected = {"detail": "This endpoint is not available in platform mode. Manage this resource via the platform UI."}
    assert users_response.status_code == 404
    assert users_response.json() == expected
    assert keys_response.status_code == 404
    assert keys_response.json() == expected
    assert budgets_response.status_code == 404
    assert budgets_response.json() == expected
    assert spend_response.status_code == 404
    assert spend_response.json() == expected

    reset_config()
    reset_db()
