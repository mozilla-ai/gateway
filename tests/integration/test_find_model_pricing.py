"""Tests for the shared find_model_pricing helper."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.db import ModelPricing
from gateway.services.pricing_service import find_model_pricing


@pytest.mark.asyncio
async def test_find_pricing_colon_format(async_db: AsyncSession) -> None:
    """Test lookup with canonical colon-separated key."""
    async_db.add(ModelPricing(model_key="openai:gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.input_price_per_million == 30.0


@pytest.mark.asyncio
async def test_find_pricing_legacy_slash_fallback(async_db: AsyncSession) -> None:
    """Test fallback to legacy slash-separated key when colon key is missing."""
    async_db.add(ModelPricing(model_key="openai/gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai/gpt-4"


@pytest.mark.asyncio
async def test_find_pricing_no_provider(async_db: AsyncSession) -> None:
    """Test lookup without a provider uses model name directly."""
    async_db.add(ModelPricing(model_key="gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    await async_db.commit()

    pricing = await find_model_pricing(async_db, None, "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "gpt-4"


@pytest.mark.asyncio
async def test_find_pricing_not_found(async_db: AsyncSession) -> None:
    """Test that None is returned when no pricing exists."""
    pricing = await find_model_pricing(async_db, "openai", "nonexistent-model")
    assert pricing is None


@pytest.mark.asyncio
async def test_find_pricing_colon_preferred_over_slash(async_db: AsyncSession) -> None:
    """Test that colon format is returned when both formats exist."""
    async_db.add(ModelPricing(model_key="openai:gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    async_db.add(ModelPricing(model_key="openai/gpt-4", input_price_per_million=10.0, output_price_per_million=20.0))
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4"
    assert pricing.input_price_per_million == 30.0
