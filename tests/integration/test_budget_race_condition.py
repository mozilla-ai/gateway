"""Tests for budget enforcement behavior."""

from typing import Any
from unittest.mock import patch

import pytest

from gateway.models.entities import Budget, User
from gateway.repositories.users_repository import get_active_user
from gateway.services.budget_service import validate_user_budget


@pytest.mark.asyncio
async def test_validate_user_budget_reads_user_without_locking(
    async_db: Any,
) -> None:
    """validate_user_budget should allow under-limit users without lock contention."""
    budget = Budget(
        budget_id="race-budget",
        max_budget=10.0,
    )
    async_db.add(budget)

    user = User(
        user_id="race-user",
        spend=9.99,
        budget_id="race-budget",
    )
    async_db.add(user)
    await async_db.commit()

    with patch(
        "gateway.services.budget_service.get_active_user",
        wraps=get_active_user,
    ) as mock_get_active_user:
        result = await validate_user_budget(async_db, "race-user", strategy="cas")

    assert result.user_id == "race-user"
    assert mock_get_active_user.call_args.kwargs.get("for_update", False) is False


@pytest.mark.asyncio
async def test_budget_check_rejects_at_limit(
    async_db: Any,
) -> None:
    """Test that a user at or over budget limit is rejected."""
    from fastapi import HTTPException

    budget = Budget(
        budget_id="full-budget",
        max_budget=10.0,
    )
    async_db.add(budget)

    user = User(
        user_id="full-user",
        spend=10.0,
        budget_id="full-budget",
    )
    async_db.add(user)
    await async_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await validate_user_budget(async_db, "full-user")
    assert exc_info.value.status_code == 403
