"""Rerank endpoint — reorder documents by relevance to a query."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM

try:
    from any_llm import arerank  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover – available once any-llm ships rerank support
    arerank = None
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes.chat import get_provider_kwargs, rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing

router = APIRouter(prefix="/v1", tags=["rerank"])


class RerankRequest(BaseModel):
    """Rerank request."""

    model: str = Field(description="Provider-prefixed model ID, e.g. 'cohere:rerank-v3.5'")
    query: str = Field(description="The search query to rerank documents against")
    documents: list[str] = Field(description="List of document strings to rerank", min_length=1)
    top_n: int | None = Field(default=None, description="Maximum number of results to return", gt=0)
    max_tokens_per_doc: int | None = Field(default=None, description="Per-document truncation limit", gt=0)
    user: str | None = Field(default=None, description="User ID for usage attribution")


@router.post("/rerank", response_model=None)
async def create_rerank(
    raw_request: Request,
    response: Response,
    request: RerankRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> Any:
    """Rerank documents by relevance to a query.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    rerank_kwargs: dict[str, Any] = {
        "model": model,
        "query": request.query,
        "documents": request.documents,
        "provider": provider,
        **provider_kwargs,
    }
    if request.top_n is not None:
        rerank_kwargs["top_n"] = request.top_n
    if request.max_tokens_per_doc is not None:
        rerank_kwargs["max_tokens_per_doc"] = request.max_tokens_per_doc

    try:
        result = await arerank(**rerank_kwargs)

        total_tokens = result.usage.total_tokens if result.usage and result.usage.total_tokens else 0

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/rerank",
            status="success",
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        )

        if total_tokens:
            pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
            if pricing:
                cost = (total_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost
            else:
                model_ref = f"{provider}:{model}" if provider else model
                logger.warning(
                    "No pricing configured for '%s'. Cost will not be tracked for this rerank request.",
                    model_ref,
                )

        await log_writer.put(usage_log)

    except HTTPException:
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/rerank",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result
