from typing import Annotated, Any

from any_llm import AnyLLM, aresponses
from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import ResponseStreamEvent
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes.chat import get_provider_kwargs, log_usage, rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"


class ResponsesRequest(BaseModel):
    """OpenAI Responses API-compatible request."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
    stream: bool = False
    user: str | None = None


def _usage_to_completion_usage(
    usage: ResponseUsage | OpenResponsesUsage | None,
) -> CompletionUsage | None:
    if usage is None:
        return None
    return CompletionUsage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: FastAPIResponse,
    request_body: ResponsesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint."""

    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request_body.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_MASTER_KEY_USER_REQUIRED,
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_VALIDATION_FAILED,
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_NO_USER,
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request_body.model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model = AnyLLM.split_model_provider(request_body.model)
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.value}' does not support the Responses API",
        )

    provider_kwargs = get_provider_kwargs(config, provider)

    request_fields = request_body.model_dump(exclude_none=True)
    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    request_fields["user"] = user_id

    call_kwargs: dict[str, Any] = {**provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = model
    call_kwargs["provider"] = provider
    call_kwargs["input_data"] = input_payload

    try:
        if stream:
            call_kwargs["stream"] = True

            def _format_chunk(event: ResponseStreamEvent) -> str:
                return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

            def _extract_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
                response_obj = getattr(event, "response", None)
                if response_obj and getattr(response_obj, "usage", None):
                    return _usage_to_completion_usage(response_obj.usage)
                return None

            async def _on_complete(usage_data: CompletionUsage) -> None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    usage_override=usage_data,
                )

            async def _on_error(error: str) -> None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    error=error,
                )

            stream_result = await aresponses(**call_kwargs)
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(
                streaming_generator(
                    stream=stream_result,  # type: ignore[arg-type]
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=RESPONSES_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        result = await aresponses(**call_kwargs)
        usage_data = _usage_to_completion_usage(getattr(result, "usage", None))
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            usage_override=usage_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump(exclude_none=True)  # type: ignore[union-attr]
