"""Tests for transaction safety: rollback on commit failure and narrowed exception handling."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER
from gateway.models.entities import Budget, User
from gateway.services.budget_service import _is_model_free, reset_user_budget


def test_create_user_rollback_on_commit_failure(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """create_user rolls back and returns 500 when commit fails."""
    with patch(
        "gateway.api.routes.users.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.post(
            "/v1/users",
            json={"user_id": "fail-user"},
            headers=master_key_header,
        )
    assert resp.status_code == 500


def test_delete_user_rollback_on_commit_failure(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """delete_user rolls back both the API key deactivation and soft-delete on commit failure."""
    client.post("/v1/users", json={"user_id": "del-fail-user"}, headers=master_key_header)

    with patch(
        "gateway.api.routes.users.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.delete("/v1/users/del-fail-user", headers=master_key_header)
    assert resp.status_code == 500

    # User should still be active because the commit was rolled back
    resp = client.get("/v1/users/del-fail-user", headers=master_key_header)
    assert resp.status_code == 200


def test_create_key_rollback_on_commit_failure(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """create_key rolls back on commit failure."""
    with patch(
        "gateway.api.routes.keys.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.post(
            "/v1/keys",
            json={"key_name": "fail-key"},
            headers=master_key_header,
        )
    assert resp.status_code == 500


def test_create_budget_rollback_on_commit_failure(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """create_budget rolls back on commit failure."""
    with patch(
        "gateway.api.routes.budgets.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.post(
            "/v1/budgets",
            json={"max_budget": 100.0},
            headers=master_key_header,
        )
    assert resp.status_code == 500


def test_set_pricing_rollback_on_commit_failure(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """set_pricing rolls back on commit failure."""
    with patch(
        "gateway.api.routes.pricing.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.post(
            "/v1/pricing",
            json={
                "model_key": "openai:gpt-4o",
                "input_price_per_million": 2.5,
                "output_price_per_million": 10.0,
            },
            headers=master_key_header,
        )
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_reset_user_budget_rollback_on_commit_failure(async_db: Session) -> None:
    """reset_user_budget rolls back and re-raises when commit fails."""
    from datetime import UTC, datetime

    user = User(user_id="reset-fail-user", spend=50.0)
    budget = Budget(max_budget=100.0, budget_duration_sec=3600)
    async_db.add_all([user, budget])
    await async_db.commit()

    now = datetime.now(UTC)

    with (
        patch.object(async_db, "commit", side_effect=OperationalError("db", {}, Exception("disk full"))),
        patch.object(async_db, "rollback", wraps=async_db.rollback) as mock_rollback,
    ):
        with pytest.raises(OperationalError):
            await reset_user_budget(async_db, user, budget, now)

        mock_rollback.assert_called_once()


@pytest.mark.asyncio
async def test_is_model_free_catches_value_error(async_db: Session) -> None:
    """_is_model_free returns False on ValueError from split_model_provider."""
    result = await _is_model_free(async_db, "completely-invalid-model-string-no-provider")
    assert result is False


@pytest.mark.asyncio
async def test_is_model_free_catches_unsupported_provider_error(async_db: Session) -> None:
    """_is_model_free returns False on UnsupportedProviderError from split_model_provider."""
    result = await _is_model_free(async_db, "unknown:some-model")
    assert result is False


@pytest.mark.asyncio
async def test_is_model_free_catches_sqlalchemy_error(async_db: Session) -> None:
    """_is_model_free returns False on SQLAlchemy errors during pricing lookup."""
    with patch(
        "gateway.services.budget_service.find_model_pricing",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        result = await _is_model_free(async_db, "openai:gpt-4o")
        assert result is False


@pytest.mark.asyncio
async def test_is_model_free_does_not_catch_unexpected_errors(async_db: Session) -> None:
    """_is_model_free does not swallow unexpected non-DB, non-ValueError exceptions."""
    with (
        patch("gateway.services.budget_service.find_model_pricing", side_effect=RuntimeError("unexpected")),
        pytest.raises(RuntimeError, match="unexpected"),
    ):
        await _is_model_free(async_db, "openai:gpt-4o")


def test_auth_commit_failure_does_not_break_verification(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """API key verification succeeds even if the last_used_at commit fails."""
    key_resp = client.post(
        "/v1/keys",
        json={"key_name": "auth-fail-key"},
        headers=master_key_header,
    )
    api_key = key_resp.json()["key"]

    with patch(
        "gateway.api.deps.AsyncSession.commit",
        side_effect=OperationalError("db", {}, Exception("connection lost")),
    ):
        resp = client.get(
            "/v1/users",
            headers={API_KEY_HEADER: f"Bearer {api_key}"},
        )
        # Auth should not crash with 500 from the commit failure.
        # The /v1/users endpoint requires master key, so we may get 401
        # (API key not accepted as master key), but crucially not 500.
        assert resp.status_code != 500
