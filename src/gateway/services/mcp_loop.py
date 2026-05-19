"""Streaming-aware MCP tool-use loop.

Wraps one or more `acompletion` calls so that when the model emits tool_calls
for tools owned by the MCPClientPool, the loop executes them against the MCP
servers, appends the assistant + tool result messages to the conversation, and
re-calls the provider for the next iteration. Tool calls for user-supplied
(non-MCP) tools end the loop and bubble up to the caller untouched.

Both streaming and non-streaming variants are provided. The streaming variant
yields `ChatCompletionChunk` objects across the entire loop as a single
`AsyncIterator`, which can be fed into the existing `streaming_generator`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from any_llm import acompletion

from gateway.log_config import logger

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

    from gateway.services.mcp_client import MCPClientPool

MAX_TOOL_ITERATIONS_CAP = 25
DEFAULT_MAX_TOOL_ITERATIONS = 10

# Lead-in for the per-source purpose-hint block we prepend to the system message.
# Generic across MCP servers, the sandbox code-execution tool, and any future
# tool source. Surfaced as a constant so phrasing can be tuned for different
# model families (open-weight models in particular benefit from more directive
# language).
PURPOSE_HINT_HEADER = "You have access to the following tools:"


class MaxToolIterationsExceeded(Exception):
    """Raised when the loop fails to reach a non-tool-call finish in N rounds."""


def inject_purpose_hints(
    messages: list[dict[str, Any]],
    hints: list[tuple[str, str]],
    *,
    header: str | None = None,
) -> list[dict[str, Any]]:
    """Prepend or extend the system message with per-tool usage hints.

    Header resolution priority:
      1. ``header`` arg (per-request override, set from the request body)
      2. ``GATEWAY_TOOLS_HEADER`` env (per-deployment override)
      3. :data:`PURPOSE_HINT_HEADER` built-in default
    """
    if not hints:
        return messages

    effective_header = header or os.environ.get("GATEWAY_TOOLS_HEADER") or PURPOSE_HINT_HEADER
    lines = [effective_header]
    for name, hint in hints:
        lines.append(f"- {name}: {hint}")
    block = "\n".join(lines)

    out = list(messages)
    if out and out[0].get("role") == "system":
        existing = out[0].get("content") or ""
        out[0] = {**out[0], "content": f"{existing}\n\n{block}" if existing else block}
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


def _accumulate_tool_call_deltas(slots: dict[int, dict[str, Any]], deltas: list[Any]) -> None:
    """Merge incremental streaming tool_call deltas into per-index slots."""
    for delta in deltas:
        idx = delta.index
        slot = slots.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if getattr(delta, "id", None):
            slot["id"] = delta.id
        if getattr(delta, "type", None):
            slot["type"] = delta.type
        fn = getattr(delta, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["function"]["name"] += fn.name
            if getattr(fn, "arguments", None):
                slot["function"]["arguments"] += fn.arguments


def _finalize_tool_calls(slots: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [slots[i] for i in sorted(slots)]


def _execute_split(tool_calls: list[dict[str, Any]], pool: MCPClientPool) -> tuple[list[dict[str, Any]], bool]:
    """Return (mcp_owned_calls, has_foreign_calls). Foreign = user-supplied, gateway can't execute."""
    mcp_calls: list[dict[str, Any]] = []
    has_foreign = False
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if pool.owns_tool(name):
            mcp_calls.append(tc)
        else:
            has_foreign = True
    return mcp_calls, has_foreign


async def _execute_mcp_calls(pool: MCPClientPool, mcp_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each MCP tool call and return the resulting tool-role messages.

    Tool failures (network errors, server errors, schema mismatches, MCP-specific
    or httpx-level transport errors) are converted to a ``[tool error] …`` message
    so the model can recover. Only cancellation/interrupt-class exceptions
    (``asyncio.CancelledError``, ``KeyboardInterrupt``) escape — they inherit
    from ``BaseException`` and never reach the ``Exception`` clause. That's the
    standard idiom for "treat tool failures as recoverable, let cancellation
    propagate".
    """
    out: list[dict[str, Any]] = []
    for tc in mcp_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        try:
            text = await pool.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.warning("MCP tool %s execution failed: %s", name, exc)
            text = f"[tool error] {exc}"
        out.append({"role": "tool", "tool_call_id": tc["id"] or "", "content": text})
    return out


async def mcp_tool_loop_stream(
    *,
    completion_kwargs: dict[str, Any],
    pool: MCPClientPool,
    max_iterations: int,
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chunks across multiple `acompletion(stream=True)` calls, with MCP execution between rounds.

    Tool-call deltas from intermediate iterations are streamed straight through to the
    caller (so clients that want to render "thinking" still get those bytes), but the
    *terminal* chunk of each intermediate iteration — the one carrying
    ``finish_reason="tool_calls"`` — is buffered and dropped if the loop is going to
    iterate again. Forwarding that chunk would tell an OpenAI-compatible client "this
    is the final answer", and most SDKs stop reading at that point; subsequent
    iterations' content would be silently truncated.

    The terminal chunk is forwarded in three cases:
      * the iteration ended with a non-``tool_calls`` finish_reason (e.g. ``stop``),
      * the model produced foreign tool_calls (caller needs to dispatch them), or
      * the model produced no MCP-owned calls at all (loop exits, terminal goes through).
    """
    messages = list(completion_kwargs.get("messages") or [])
    user_tools = list(completion_kwargs.get("tools") or [])
    merged_tools = user_tools + pool.openai_tools

    base = {k: v for k, v in completion_kwargs.items() if k not in {"messages", "tools"}}
    base["stream"] = True

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, "messages": messages}
        if merged_tools:
            kwargs["tools"] = merged_tools

        stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**kwargs)  # type: ignore[assignment]
        slots: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        pending_terminal: ChatCompletionChunk | None = None

        async for chunk in stream:
            chunk_is_terminal = False
            if chunk.choices:
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "tool_calls", None):
                    _accumulate_tool_call_deltas(slots, delta.tool_calls)
                if choice.finish_reason:
                    # Sticky-tool-calls: a trailing ``stop`` chunk from
                    # Anthropic must not override ``tool_calls`` we've
                    # already seen on this same iteration.
                    if not (finish_reason == "tool_calls" and choice.finish_reason != "tool_calls"):
                        finish_reason = choice.finish_reason
                    chunk_is_terminal = True
            if chunk_is_terminal:
                pending_terminal = chunk
                continue
            yield chunk

        if finish_reason != "tool_calls":
            if pending_terminal is not None:
                yield pending_terminal
            return

        tool_calls = _finalize_tool_calls(slots)
        mcp_calls, has_foreign = _execute_split(tool_calls, pool)
        if has_foreign or not mcp_calls:
            # Mixed (has_foreign with mcp_calls) is handled the same as
            # all-foreign here: the terminal chunk is forwarded so the
            # client sees the full tool_calls set, and any MCP-owned calls
            # in a mixed batch are left for a follow-up turn (the
            # non-streaming variant filters them out; rewriting deltas
            # mid-stream is too invasive to do here).
            if pending_terminal is not None:
                yield pending_terminal
            return

        # All-MCP: silently drop the terminal so the client doesn't think
        # this iteration's response was the final answer.
        messages.append({"role": "assistant", "tool_calls": mcp_calls})
        messages.extend(await _execute_mcp_calls(pool, mcp_calls))

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")


async def mcp_tool_loop(
    *,
    completion_kwargs: dict[str, Any],
    pool: MCPClientPool,
    max_iterations: int,
) -> ChatCompletion:
    """Non-streaming variant. Accumulates usage across iterations into the returned completion."""
    messages = list(completion_kwargs.get("messages") or [])
    user_tools = list(completion_kwargs.get("tools") or [])
    merged_tools = user_tools + pool.openai_tools

    base = {k: v for k, v in completion_kwargs.items() if k not in {"messages", "tools", "stream"}}

    acc_prompt = 0
    acc_completion = 0

    for _ in range(max_iterations):
        kwargs: dict[str, Any] = {**base, "messages": messages, "stream": False}
        if merged_tools:
            kwargs["tools"] = merged_tools

        completion: ChatCompletion = await acompletion(**kwargs)  # type: ignore[assignment]
        if completion.usage:
            acc_prompt += completion.usage.prompt_tokens or 0
            acc_completion += completion.usage.completion_tokens or 0

        if not completion.choices:
            return completion

        choice = completion.choices[0]
        if choice.finish_reason != "tool_calls":
            _fold_usage(completion, acc_prompt, acc_completion)
            return completion

        sdk_calls = choice.message.tool_calls or []
        # Duck-type the SDK tool-call: any object with a `.function` attribute is a
        # function tool-call. We avoid `isinstance` here because `acompletion`'s
        # return type uses the OpenAI SDK's class, while any_llm exposes a
        # same-named but distinct class alias.
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in sdk_calls
            if hasattr(tc, "function")
        ]
        mcp_calls, has_foreign = _execute_split(tool_calls, pool)
        if has_foreign:
            # Mixed batch: execute the MCP-owned subset internally so its
            # work isn't wasted, then filter those calls out of the returned
            # completion so the caller only sees tool_calls it can itself
            # dispatch. The conversation continues client-side; if the
            # caller wants to keep using the gateway's MCP tools they'll
            # send the foreign-tool results back on the next request.
            if mcp_calls:
                await _execute_mcp_calls(pool, mcp_calls)
                foreign_sdk_calls = [
                    tc for tc in sdk_calls if hasattr(tc, "function") and not pool.owns_tool(tc.function.name)
                ]
                try:
                    choice.message.tool_calls = foreign_sdk_calls or None
                except (AttributeError, TypeError):
                    # If the SDK model is frozen, fall back to leaving the
                    # original list. Cleaner UX requires SDK mutability.
                    logger.warning(
                        "MCP-mixed: could not filter tool_calls on response; client will see MCP calls "
                        "the gateway already executed (no-op on the client side).",
                    )
            _fold_usage(completion, acc_prompt, acc_completion)
            return completion
        if not mcp_calls:
            _fold_usage(completion, acc_prompt, acc_completion)
            return completion

        messages.append({"role": "assistant", "tool_calls": mcp_calls})
        messages.extend(await _execute_mcp_calls(pool, mcp_calls))

    raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")


def _fold_usage(completion: ChatCompletion, prompt_total: int, completion_total: int) -> None:
    if completion.usage is None:
        return
    completion.usage.prompt_tokens = prompt_total
    completion.usage.completion_tokens = completion_total
    completion.usage.total_tokens = prompt_total + completion_total
