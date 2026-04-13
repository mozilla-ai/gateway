"""Shared pricing lookup utilities."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import ModelPricing


def normalize_effective_at(value: datetime | None) -> datetime:
    """Normalize a datetime to an aware UTC timestamp, defaulting to now."""

    normalized = value or datetime.now(UTC)
    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=UTC)
    return normalized.astimezone(UTC)


async def _find_by_model_key(db: AsyncSession, model_key: str, as_of: datetime) -> ModelPricing | None:
    stmt = (
        select(ModelPricing)
        .where(
            ModelPricing.model_key == model_key,
            ModelPricing.effective_at <= as_of,
        )
        .order_by(ModelPricing.effective_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def find_model_pricing(
    db: AsyncSession,
    provider: str | None,
    model: str,
    *,
    as_of: datetime | None = None,
) -> ModelPricing | None:
    """Look up model pricing as of a timestamp, with legacy key fallback."""

    lookup_time = normalize_effective_at(as_of)
    model_key = f"{provider}:{model}" if provider else model
    pricing = await _find_by_model_key(db, model_key, lookup_time)
    if pricing or not provider:
        return pricing

    legacy_key = f"{provider}/{model}"
    return await _find_by_model_key(db, legacy_key, lookup_time)
