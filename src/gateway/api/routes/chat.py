import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.auth.vertex_auth import setup_vertex_environment
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing
from gateway.streaming import OPENAI_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1/chat", tags=["chat"])

_USAGE_NON_RETRYABLE_STATUS_CODES = {401, 404, 409, 422}


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider kwargs from config for acompletion calls.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.)

    """
    kwargs: dict[str, Any] = {}
    if provider.value in config.providers:
        provider_config = config.providers[provider.value]

        if provider == LLMProvider.VERTEXAI:
            vertex_creds = provider_config.get("credentials")
            vertex_project = provider_config.get("project")
            vertex_location = provider_config.get("location")

            kwargs.update(
                setup_vertex_environment(
                    credentials=vertex_creds,
                    project=vertex_project,
                    location=vertex_location,
                )
            )
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]
        else:
            kwargs = {k: v for k, v in provider_config.items() if k != "client_args"}
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]

    return kwargs


async def log_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    """Log API usage to database and update user spend.

    Args:
        db: Database session
        api_key_id: API key identifier (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        error: Error message (if failed)

    """
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

    pricing = await find_model_pricing(db, provider, model)
    if pricing:
        cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
            usage_data.completion_tokens / 1_000_000
        ) * pricing.output_price_per_million
        usage_log.cost = cost
        record_cost(str(provider or ""), model, cost)
    else:
        model_ref = f"{provider}:{model}" if provider else model
        logger.warning(f"No pricing configured for '{model_ref}'. Usage will be tracked without cost.")

    await log_writer.put(usage_log)


def _extract_platform_user_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return token


def _split_model_selector(model_selector: str) -> tuple[str | None, str]:
    if ":" in model_selector:
        provider, model_name = model_selector.split(":", 1)
        return provider or None, model_name
    if "/" in model_selector:
        provider, model_name = model_selector.split("/", 1)
        return provider or None, model_name
    return None, model_selector


def _platform_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _safe_detail_from_platform(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


async def _post_platform(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)


async def _resolve_platform_credentials(
    config: GatewayConfig,
    user_token: str,
    model_selector: str,
) -> tuple[str, str, str | None, str, bool, str]:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Platform mode is misconfigured",
        )

    provider, model_name = _split_model_selector(model_selector)
    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/provider-keys/resolve")
    resolve_headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    resolve_body: dict[str, Any] = {"model": model_name}
    if provider:
        resolve_body["provider"] = provider

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=resolve_headers,
            body=resolve_body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return (
            str(payload["provider"]),
            str(payload["model"]),
            payload.get("api_base"),
            str(payload["api_key"]),
            bool(payload.get("managed", False)),
            str(payload["correlation_id"]),
        )

    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "Authorization request rejected")
        headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _report_platform_usage(
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
) -> None:
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return

    timeout_ms = int(config.platform.get("usage_timeout_ms", 5000))
    max_retries = int(config.platform.get("usage_max_retries", 3))
    usage_url = _platform_url(platform_base_url, "/gateway/usage")
    headers = {"X-Gateway-Token": config.platform_token or ""}

    payload: dict[str, Any] = {"correlation_id": correlation_id, "status": outcome}
    if outcome == "success":
        token_usage = usage or CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        payload["usage"] = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
        }

    delay_seconds = 0.25
    for attempt in range(1, max_retries + 1):
        should_retry = False
        try:
            response = await _post_platform(
                url=usage_url,
                headers=headers,
                body=payload,
                timeout_seconds=timeout_ms / 1000,
            )
            if response.status_code == 204:
                return
            if response.status_code in _USAGE_NON_RETRYABLE_STATUS_CODES:
                return
            should_retry = response.status_code >= 500
        except (httpx.TimeoutException, httpx.NetworkError):
            should_retry = True

        if not should_retry or attempt == max_retries:
            return

        await asyncio.sleep(delay_seconds)
        delay_seconds *= 2


@router.post("/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: ChatCompletionRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from any-llm providers.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    if not request.model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request: model is required",
        )

    api_key: APIKey | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    platform_mode = config.is_platform_mode
    correlation_id: str | None = None

    if platform_mode:
        user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        provider_name, model, api_base, resolved_api_key, managed, correlation_id = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=request.model,
        )
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        provider = LLMProvider(provider_name)
        provider_kwargs = {"api_key": resolved_api_key}
        if api_base:
            provider_kwargs["api_base"] = api_base
        response.headers["X-Correlation-ID"] = correlation_id
        logger.info(
            "Platform resolve succeeded provider=%s model=%s managed=%s resolve_latency_ms=%.2f",
            provider_name,
            model,
            managed,
            resolve_latency_ms,
        )
    else:
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session unavailable",
            )

        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
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

    # User request fields take precedence over provider config defaults
    request_fields = request.model_dump(exclude_unset=True)
    if platform_mode:
        request_fields["model"] = f"{provider.value}:{model}"

    completion_kwargs = {**provider_kwargs, **request_fields}

    if request.stream and completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}

    try:
        if request.stream:

            def _format_chunk(chunk: ChatCompletionChunk) -> str:
                return f"data: {chunk.model_dump_json()}\n\n"

            def _extract_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
                if not chunk.usage:
                    return None
                return CompletionUsage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

            async def _on_complete(usage_data: CompletionUsage) -> None:
                if platform_mode and correlation_id:
                    asyncio.create_task(
                        _report_platform_usage(
                            config=config,
                            correlation_id=correlation_id,
                            outcome="success",
                            usage=usage_data,
                        )
                    )
                    return

                if db is None:
                    return

                    await log_usage(
                        db=db,
                        log_writer=log_writer,
                        api_key_id=api_key_id,
                        model=model,
                        provider=provider,
                        endpoint="/v1/chat/completions",
                        user_id=user_id,
                        usage_override=usage_data,
                    )

            async def _on_error(error: str) -> None:
                if platform_mode and correlation_id:
                    asyncio.create_task(
                        _report_platform_usage(
                            config=config,
                            correlation_id=correlation_id,
                            outcome="error",
                            usage=None,
                        )
                    )
                    return

                if db is None:
                    return

                    await log_usage(
                        db=db,
                        log_writer=log_writer,
                        api_key_id=api_key_id,
                        model=model,
                        provider=provider,
                        endpoint="/v1/chat/completions",
                        user_id=user_id,
                        error=error,
                    )

            stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**completion_kwargs)  # type: ignore[assignment]
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(
                streaming_generator(
                    stream=stream,
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=OPENAI_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        completion: ChatCompletion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        if platform_mode and correlation_id:
            usage_data = completion.usage
            background_tasks.add_task(
                _report_platform_usage,
                config,
                correlation_id,
                "success",
                usage_data,
            )
        elif db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                response=completion,
            )

    except HTTPException:
        raise
    except Exception as e:
        if platform_mode and correlation_id:
            background_tasks.add_task(
                _report_platform_usage,
                config,
                correlation_id,
                "error",
                None,
            )
        elif db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                error=str(e),
            )

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM provider timeout",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion
