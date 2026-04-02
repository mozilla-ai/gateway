"""Integration tests for helper wiring through API endpoints."""

from fastapi.testclient import TestClient


def test_get_active_user_via_budget_endpoint(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    client.post(
        "/v1/users",
        json={"user_id": "helper-test-user", "alias": "Helper Test"},
        headers=master_key_header,
    )
    resp = client.get("/v1/users/helper-test-user", headers=master_key_header)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "helper-test-user"


def test_get_active_user_returns_none_for_deleted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    client.post(
        "/v1/users",
        json={"user_id": "to-delete-user"},
        headers=master_key_header,
    )
    client.delete("/v1/users/to-delete-user", headers=master_key_header)
    resp = client.get("/v1/users/to-delete-user", headers=master_key_header)
    assert resp.status_code == 404


def test_get_active_user_returns_none_for_nonexistent(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.get("/v1/users/nonexistent-user", headers=master_key_header)
    assert resp.status_code == 404


def test_budget_from_model_roundtrip(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/budgets",
        json={"max_budget": 50.0, "budget_duration_sec": 3600},
        headers=master_key_header,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["max_budget"] == 50.0
    assert data["budget_duration_sec"] == 3600


def test_pricing_from_model_roundtrip(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_key"] == "openai:gpt-4o"
    assert data["input_price_per_million"] == 2.5
    assert data["output_price_per_million"] == 10.0


def test_resolve_user_id_chat_master_key_requires_user(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_resolve_user_id_messages_master_key_requires_user(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
        headers=master_key_header,
    )
    assert resp.status_code == 400
