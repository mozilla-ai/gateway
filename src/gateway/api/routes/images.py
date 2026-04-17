"""OpenAI-compatible image generation endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import AnyLLM, aimage_generation
from any_llm.types.image import ImagesResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
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

router = APIRouter(prefix="/v1", tags=["images"])


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    model: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    response_format: str | None = None
    user: str | None = None


@router.post("/images/generations", response_model=None)
async def create_image(
    raw_request: Request,
    response: Response,
    request: ImageGenerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible image generation endpoint.

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

    image_kwargs: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "provider": provider,
        **provider_kwargs,
    }
    if request.n is not None:
        image_kwargs["n"] = request.n
    if request.size is not None:
        image_kwargs["size"] = request.size
    if request.quality is not None:
        image_kwargs["quality"] = request.quality
    if request.style is not None:
        image_kwargs["style"] = request.style
    if request.response_format is not None:
        image_kwargs["response_format"] = request.response_format

    try:
        result: ImagesResponse = await aimage_generation(**image_kwargs)

        n_images = len(result.data) if result.data else 1

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/images/generations",
            status="success",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Image pricing: repurpose input_price_per_million as price-per-image
        pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
        if pricing:
            cost = n_images * pricing.input_price_per_million
            usage_log.cost = cost
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning("No pricing configured for '%s'. Usage will be tracked without cost.", model_ref)

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
            endpoint="/v1/images/generations",
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

    return result.model_dump()
