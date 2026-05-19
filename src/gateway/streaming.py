"""Shared SSE streaming utilities for gateway routes."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar

from any_llm.types.completion import CompletionUsage

from gateway.log_config import logger

A = TypeVar("A")  # Attempt-like, opaque to this module
C = TypeVar("C")  # Chunk type emitted by the upstream stream


DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class StreamingAttemptFailure:
    """The reason an attempt was abandoned before any bytes were flushed."""

    error_class: str
    exception: BaseException


@dataclass(frozen=True)
class StreamFormat:
    """SSE formatting configuration for a streaming protocol."""

    done_marker: str
    error_payload: str
    yield_done_on_error: bool


_OPENAI_ERROR = json.dumps({"error": {"message": "An error occurred during streaming", "type": "server_error"}})
_ANTHROPIC_ERROR = json.dumps(
    {"type": "error", "error": {"type": "api_error", "message": "An error occurred during streaming"}}
)

OPENAI_STREAM_FORMAT = StreamFormat(
    done_marker="data: [DONE]\n\n",
    error_payload=f"data: {_OPENAI_ERROR}\n\n",
    yield_done_on_error=True,
)

RESPONSES_STREAM_FORMAT = StreamFormat(
    done_marker="data: [DONE]\n\n",
    error_payload=f"event: error\ndata: {_OPENAI_ERROR}\n\n",
    yield_done_on_error=False,
)

ANTHROPIC_STREAM_FORMAT = StreamFormat(
    done_marker="event: done\ndata: {}\n\n",
    error_payload=f"event: error\ndata: {_ANTHROPIC_ERROR}\n\n",
    yield_done_on_error=False,
)


def _merge_usage(current: CompletionUsage, update: CompletionUsage) -> CompletionUsage:
    """Merge usage data, keeping the last non-zero value for each field."""
    return CompletionUsage(
        prompt_tokens=update.prompt_tokens or current.prompt_tokens,
        completion_tokens=update.completion_tokens or current.completion_tokens,
        total_tokens=update.total_tokens or current.total_tokens,
    )


async def streaming_generator(
    stream: AsyncIterator[Any],
    format_chunk: Callable[[Any], str],
    extract_usage: Callable[[Any], CompletionUsage | None],
    fmt: StreamFormat,
    on_complete: Callable[[CompletionUsage], Awaitable[None]],
    on_error: Callable[[str], Awaitable[None]],
    label: str,
) -> AsyncIterator[str]:
    """Shared SSE streaming generator with usage tracking and error handling.

    Args:
        stream: Async iterator of chunks from the provider
        format_chunk: Formats a chunk into an SSE string
        extract_usage: Extracts usage from a chunk, or returns None if no usage present
        fmt: SSE format configuration (done marker, error payload, etc.)
        on_complete: Called with aggregated usage after successful streaming
        on_error: Called with error message on failure
        label: Identifier for error log messages (e.g., "openai:gpt-4")

    """
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    has_usage = False

    try:
        async for chunk in stream:
            chunk_usage = extract_usage(chunk)
            if chunk_usage:
                usage = _merge_usage(usage, chunk_usage)
                has_usage = True
            yield format_chunk(chunk)
        yield fmt.done_marker

        if has_usage:
            await on_complete(usage)
    except Exception as e:
        yield fmt.error_payload
        if fmt.yield_done_on_error:
            yield fmt.done_marker
        try:
            await on_error(str(e))
        except Exception as log_err:
            logger.error("Failed to log streaming error usage: %s", log_err)
        logger.error("Streaming error for %s: %s", label, e)


async def iterate_streaming_attempts(
    attempts: Sequence[A],
    build_stream: Callable[[A], Awaitable[AsyncIterator[C]]],
    classify_error: Callable[[BaseException], tuple[bool, str]],
    on_attempt_failed: Callable[[A, StreamingAttemptFailure], Awaitable[None]],
    first_chunk_timeout_seconds: float = DEFAULT_FIRST_CHUNK_TIMEOUT_SECONDS,
) -> tuple[A, AsyncIterator[C]]:
    """Iterate over ``attempts`` until one yields its first chunk; return that
    attempt and an iterator that re-emits the chunk followed by the rest of
    the stream.

    For each attempt:

    1. ``build_stream(attempt)`` is awaited — typically wraps ``acompletion``.
       If it raises, the error is classified, ``on_attempt_failed`` is called
       so the caller can record per-attempt failure metadata (regardless of
       whether the error is retryable), then retryable errors continue to the
       next attempt and non-retryable errors propagate.
    2. The first chunk is awaited with a ``first_chunk_timeout_seconds`` cap.
       If the timeout fires or the upstream raises before yielding, the same
       classification + ``on_attempt_failed`` logic applies.
    3. Once a first chunk is in hand, we commit — the function returns and
       the caller flushes the response. Errors after this point reach the
       client; they cannot be hidden without buffering the entire stream.

    This is the streaming-mode analogue of the non-streaming retry loop in
    ``chat.py``. The contract is identical: ``on_attempt_failed`` records every
    failed attempt, retryable failures skip-and-continue, non-retryable
    failures propagate immediately, and the last exception is raised if every
    attempt is exhausted with retryable failures.

    Latency contract: zero added latency in the success case — we hold the
    first chunk only long enough to call this function's caller. In the
    failure case, each abandoned attempt costs at most
    ``first_chunk_timeout_seconds``.
    """
    last_exception: BaseException | None = None

    for attempt in attempts:
        try:
            stream = await build_stream(attempt)
        except BaseException as exc:
            retryable, error_class = classify_error(exc)
            await on_attempt_failed(attempt, StreamingAttemptFailure(error_class, exc))
            last_exception = exc
            if not retryable:
                raise
            continue

        try:
            first_chunk: C = await asyncio.wait_for(
                stream.__anext__(),
                timeout=first_chunk_timeout_seconds,
            )
        except StopAsyncIteration:
            # Stream completed without yielding. Unusual but valid — commit
            # with an empty iterator so the caller still gets a clean SSE
            # close sequence rather than another upstream attempt.
            return attempt, _empty_async_iter()
        except asyncio.TimeoutError as exc:
            await on_attempt_failed(
                attempt,
                StreamingAttemptFailure("timeout", exc),
            )
            await _close_stream_quietly(stream)
            last_exception = exc
            continue
        except BaseException as exc:
            retryable, error_class = classify_error(exc)
            await on_attempt_failed(attempt, StreamingAttemptFailure(error_class, exc))
            await _close_stream_quietly(stream)
            last_exception = exc
            if not retryable:
                raise
            continue

        return attempt, _stitched(first_chunk, stream)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("iterate_streaming_attempts: no attempts provided")


async def _stitched(first: C, remaining: AsyncIterator[C]) -> AsyncIterator[C]:
    yield first
    async for chunk in remaining:
        yield chunk


async def _empty_async_iter() -> AsyncIterator[C]:
    return
    yield  # unreachable; makes this a generator


async def _close_stream_quietly(stream: AsyncIterator[Any]) -> None:
    close = getattr(stream, "aclose", None)
    if callable(close):
        with suppress(BaseException):
            await close()
