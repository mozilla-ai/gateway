"""Unit tests for pure helper behavior shared by route handlers."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes.budgets import BudgetResponse
from gateway.api.routes.pricing import PricingResponse


def _make_error(detail: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def test_resolve_user_id_master_key_with_user() -> None:
    user_id = resolve_user_id(
        user_id_from_request="user-1",
        api_key=None,
        is_master_key=True,
        master_key_error=_make_error("master key requires user"),
        no_api_key_error=_make_error("no api key"),
        no_user_error=_make_error("no user"),
    )
    assert user_id == "user-1"


def test_resolve_user_id_master_key_without_user() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id(
            user_id_from_request=None,
            api_key=None,
            is_master_key=True,
            master_key_error=_make_error("master key requires user"),
            no_api_key_error=_make_error("no api key"),
            no_user_error=_make_error("no user"),
        )
    assert exc_info.value.detail == "master key requires user"


def test_resolve_user_id_api_key_with_request_user() -> None:
    api_key = MagicMock()
    api_key.user_id = "key-user"
    user_id = resolve_user_id(
        user_id_from_request="explicit-user",
        api_key=api_key,
        is_master_key=False,
        master_key_error=_make_error("master key requires user"),
        no_api_key_error=_make_error("no api key"),
        no_user_error=_make_error("no user"),
    )
    assert user_id == "explicit-user"


def test_resolve_user_id_falls_back_to_api_key() -> None:
    api_key = MagicMock()
    api_key.user_id = "key-user"
    user_id = resolve_user_id(
        user_id_from_request=None,
        api_key=api_key,
        is_master_key=False,
        master_key_error=_make_error("master key requires user"),
        no_api_key_error=_make_error("no api key"),
        no_user_error=_make_error("no user"),
    )
    assert user_id == "key-user"


def test_resolve_user_id_no_api_key() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id(
            user_id_from_request=None,
            api_key=None,
            is_master_key=False,
            master_key_error=_make_error("master key requires user"),
            no_api_key_error=_make_error("no api key", 500),
            no_user_error=_make_error("no user"),
        )
    assert exc_info.value.detail == "no api key"
    assert exc_info.value.status_code == 500


def test_resolve_user_id_api_key_without_user() -> None:
    api_key = MagicMock()
    api_key.user_id = None
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id(
            user_id_from_request=None,
            api_key=api_key,
            is_master_key=False,
            master_key_error=_make_error("master key requires user"),
            no_api_key_error=_make_error("no api key"),
            no_user_error=_make_error("no user", 500),
        )
    assert exc_info.value.detail == "no user"
    assert exc_info.value.status_code == 500


def test_resolve_user_id_empty_string_treated_as_missing() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_user_id(
            user_id_from_request="",
            api_key=None,
            is_master_key=True,
            master_key_error=_make_error("master key requires user"),
            no_api_key_error=_make_error("no api key"),
            no_user_error=_make_error("no user"),
        )
    assert exc_info.value.detail == "master key requires user"


def test_budget_response_from_model() -> None:
    budget = MagicMock()
    budget.budget_id = "budget-1"
    budget.max_budget = 100.0
    budget.budget_duration_sec = 86400
    budget.created_at = datetime(2025, 1, 1, tzinfo=UTC)
    budget.updated_at = datetime(2025, 1, 2, tzinfo=UTC)

    resp = BudgetResponse.from_model(budget)
    assert resp.budget_id == "budget-1"
    assert resp.max_budget == 100.0
    assert resp.budget_duration_sec == 86400
    assert resp.created_at == "2025-01-01T00:00:00+00:00"
    assert resp.updated_at == "2025-01-02T00:00:00+00:00"


def test_budget_response_from_model_nullable_fields() -> None:
    budget = MagicMock()
    budget.budget_id = "budget-2"
    budget.max_budget = None
    budget.budget_duration_sec = None
    budget.created_at = datetime(2025, 6, 15, tzinfo=UTC)
    budget.updated_at = datetime(2025, 6, 15, tzinfo=UTC)

    resp = BudgetResponse.from_model(budget)
    assert resp.max_budget is None
    assert resp.budget_duration_sec is None


def test_pricing_response_from_model() -> None:
    pricing = MagicMock()
    pricing.model_key = "openai:gpt-4"
    pricing.input_price_per_million = 30.0
    pricing.output_price_per_million = 60.0
    pricing.created_at = datetime(2025, 3, 1, tzinfo=UTC)
    pricing.updated_at = datetime(2025, 3, 2, tzinfo=UTC)

    resp = PricingResponse.from_model(pricing)
    assert resp.model_key == "openai:gpt-4"
    assert resp.input_price_per_million == 30.0
    assert resp.output_price_per_million == 60.0
    assert resp.created_at == "2025-03-01T00:00:00+00:00"
    assert resp.updated_at == "2025-03-02T00:00:00+00:00"
