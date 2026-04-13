"""Shared pricing lookup utilities."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import ModelPricing


async def find_model_pricing(db: AsyncSession, provider: str | None, model: str) -> ModelPricing | None:
    """Look up model pricing, falling back to legacy slash-separated key format."""

    model_key = f"{provider}:{model}" if provider else model
    result = await db.execute(select(ModelPricing).where(ModelPricing.model_key == model_key))
    pricing = result.scalar_one_or_none()
    if pricing or not provider:
        return pricing

    legacy_key = f"{provider}/{model}"
    legacy_result = await db.execute(select(ModelPricing).where(ModelPricing.model_key == legacy_key))
    return legacy_result.scalar_one_or_none()
