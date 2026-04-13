"""Tests for pricing configuration from config file."""

from typing import Any

import pytest
from any_llm.types.completion import CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.api.routes.chat import log_usage
from gateway.core.config import GatewayConfig, PricingConfig
from gateway.db import ModelPricing, get_db
from gateway.main import create_app
from gateway.models.entities import UsageLog

from .conftest import build_async_session_override


def test_pricing_loaded_from_config(postgres_url: str, test_db: Session) -> None:
    """Test that pricing is loaded from config file on startup."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai:gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
            "openai:gpt-3.5-turbo": PricingConfig(
                input_price_per_million=0.5,
                output_price_per_million=1.5,
            ),
        },
    )

    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(postgres_url)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app):
            # Check GPT-4 pricing
            pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
            assert pricing is not None, "GPT-4 pricing should be loaded from config"
            assert pricing.input_price_per_million == 30.0
            assert pricing.output_price_per_million == 60.0

            # Check GPT-3.5 pricing
            pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-3.5-turbo").first()
            assert pricing is not None, "GPT-3.5-turbo pricing should be loaded from config"
            assert pricing.input_price_per_million == 0.5
            assert pricing.output_price_per_million == 1.5
    finally:
        dispose_override()


def test_database_pricing_takes_precedence(postgres_url: str, test_db: Session) -> None:
    """Test that existing database pricing is not overwritten by config."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai:gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
        },
    )

    # Pre-populate database with different pricing
    existing_pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=25.0,
        output_price_per_million=50.0,
    )
    test_db.add(existing_pricing)
    test_db.commit()

    # Create app (which loads config pricing)
    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(postgres_url)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app):
            # Check that database pricing was preserved
            pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
            assert pricing is not None
            # Should keep database values, not config values
            assert pricing.input_price_per_million == 25.0
            assert pricing.output_price_per_million == 50.0
    finally:
        dispose_override()


def test_pricing_validation_requires_configured_provider(postgres_url: str) -> None:
    """Test that pricing initialization fails if provider is not configured."""
    # Config with pricing for a provider that's not in providers list
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "anthropic:claude-3-opus": PricingConfig(
                input_price_per_million=15.0,
                output_price_per_million=75.0,
            ),
        },
    )

    # Should raise ValueError when trying to initialize pricing
    app = create_app(config)

    with pytest.raises(ValueError, match="provider 'anthropic' is not configured"):
        with TestClient(app):
            pass


def test_pricing_loaded_from_config_normalizes_legacy_slash_format(postgres_url: str, test_db: Session) -> None:
    """Test that pricing configured with legacy slash format is normalized to colon format."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai/gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
        },
    )

    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(postgres_url)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app):
            # Pricing should be stored with canonical colon format, not slash
            pricing_slash = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai/gpt-4").first()
            assert pricing_slash is None, "Pricing should not be stored with legacy slash format"

            pricing_colon = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
            assert pricing_colon is not None, "Pricing should be stored with canonical colon format"
            assert pricing_colon.input_price_per_million == 30.0
            assert pricing_colon.output_price_per_million == 60.0
    finally:
        dispose_override()


def test_set_pricing_api_normalizes_legacy_slash_format(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that the pricing API normalizes legacy slash format to colon format."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "gemini/gemini-2.5-flash",
            "input_price_per_million": 0.075,
            "output_price_per_million": 0.30,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "gemini:gemini-2.5-flash", "API should normalize slash to colon format"


def test_pricing_initialization_with_no_config(postgres_url: str, test_db: Session) -> None:
    """Test that app starts successfully when no pricing is configured."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={},  # Empty pricing
    )

    app = create_app(config)
    override_get_db, dispose_override = build_async_session_override(postgres_url)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app):
            # App should start successfully
            # No pricing should be in database
            pricing_count = test_db.query(ModelPricing).count()
            assert pricing_count == 0, "No pricing should be loaded when config is empty"
    finally:
        dispose_override()


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_legacy_slash_format(async_db) -> None:
    """Test that log_usage falls back to legacy slash format when colon format is absent."""

    class _Writer:
        def __init__(self) -> None:
            self.logs: list[UsageLog] = []

        async def put(self, log: UsageLog) -> None:
            self.logs.append(log)

    legacy_pricing = ModelPricing(
        model_key="openai/gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    async_db.add(legacy_pricing)
    await async_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    writer = _Writer()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert log.cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_colon_format(async_db) -> None:
    """Test that log_usage uses canonical colon pricing when available."""

    class _Writer:
        def __init__(self) -> None:
            self.logs: list[UsageLog] = []

        async def put(self, log: UsageLog) -> None:
            self.logs.append(log)

    pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    async_db.add(pricing)
    await async_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    writer = _Writer()

    await log_usage(
        db=async_db,
        log_writer=writer,
        api_key_id=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    assert len(writer.logs) == 1
    log = writer.logs[0]
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert log.cost == pytest.approx(expected_cost)
