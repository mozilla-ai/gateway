"""Unit tests for the MCP tool-use loop and its pure helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChunkChoice,
    CompletionUsage,
    Function,
)
from any_llm.types.completion import (
    ChoiceDeltaToolCallFunction as DeltaFn,
)

from gateway.services import mcp_loop as mcp_loop_module
from gateway.services.mcp_loop import (
    MaxToolIterationsExceeded,
    _accumulate_tool_call_deltas,
    _finalize_tool_calls,
    inject_purpose_hints,
    mcp_tool_loop,
    mcp_tool_loop_stream,
)

_FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class _FakePool:
    """Stand-in for MCPClientPool that satisfies the loop's protocol."""

    def __init__(
        self,
        tool_names: list[str],
        purpose_hints: list[tuple[str, str]] | None = None,
        results: dict[str, str] | None = None,
    ):
        self._tool_names = set(tool_names)
        self._hints = purpose_hints or []
        self._results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
            for n in sorted(self._tool_names)
        ]

    def owns_tool(self, name: str) -> bool:
        return name in self._tool_names

    def purpose_hints(self) -> list[tuple[str, str]]:
        return list(self._hints)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name not in self._results:
            return f"ran {name}"
        return self._results[name]


def _completion(
    *,
    finish: _FinishReason,
    content: str | None = None,
    tool_calls: list[tuple[str, str, str]] | None = None,
    prompt: int = 1,
    completion_tokens: int = 1,
) -> ChatCompletion:
    """Build a ChatCompletion. tool_calls items are (id, name, arguments_json)."""
    sdk_calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id=tc[0], type="function", function=Function(name=tc[1], arguments=tc[2])
            )
            for tc in tool_calls
        ]
        if tool_calls
        else None
    )
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=cast(Any, sdk_calls))
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason=finish, index=0, message=message)],
        created=0,
        model="fake",
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt, completion_tokens=completion_tokens, total_tokens=prompt + completion_tokens
        ),
    )


def _chunk(
    *,
    finish: _FinishReason | None = None,
    content: str | None = None,
    tool_calls: list[tuple[int, str | None, str | None, str | None]] | None = None,
) -> ChatCompletionChunk:
    """Build a streaming chunk. tool_calls items are (index, id, name_delta, args_delta)."""
    delta_tool_calls = (
        [
            ChoiceDeltaToolCall(
                index=tc[0],
                id=tc[1],
                type="function" if tc[1] is not None else None,
                function=DeltaFn(name=tc[2], arguments=tc[3]) if (tc[2] or tc[3]) else None,
            )
            for tc in tool_calls
        ]
        if tool_calls
        else None
    )
    delta = ChoiceDelta(role="assistant", content=content, tool_calls=delta_tool_calls)
    return ChatCompletionChunk(
        id="cmpl-1",
        choices=[ChunkChoice(delta=delta, finish_reason=finish, index=0)],
        created=0,
        model="fake",
        object="chat.completion.chunk",
    )


# ---------- pure helpers ----------


def test_inject_purpose_hints_no_hints_returns_unchanged() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert inject_purpose_hints(msgs, []) == msgs


def test_inject_purpose_hints_prepends_when_no_system() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    out = inject_purpose_hints(msgs, [("calendar", "for scheduling")])
    assert out[0]["role"] == "system"
    assert "calendar" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_inject_purpose_hints_extends_existing_system() -> None:
    msgs = [{"role": "system", "content": "be helpful"}, {"role": "user", "content": "hi"}]
    out = inject_purpose_hints(msgs, [("cal", "use it")])
    assert out[0]["role"] == "system"
    assert "be helpful" in out[0]["content"]
    assert "cal" in out[0]["content"]


def test_finalize_tool_calls_orders_by_index() -> None:
    slots: dict[int, dict[str, Any]] = {}
    _accumulate_tool_call_deltas(
        slots,
        [
            ChoiceDeltaToolCall(index=1, id="b", type="function", function=DeltaFn(name="t2", arguments="{}")),
            ChoiceDeltaToolCall(index=0, id="a", type="function", function=DeltaFn(name="t1", arguments="{}")),
        ],
    )
    out = _finalize_tool_calls(slots)
    assert [c["id"] for c in out] == ["a", "b"]


def test_accumulate_concatenates_argument_chunks() -> None:
    slots: dict[int, dict[str, Any]] = {}
    _accumulate_tool_call_deltas(
        slots,
        [ChoiceDeltaToolCall(index=0, id="a", type="function", function=DeltaFn(name="t", arguments='{"x":'))],
    )
    _accumulate_tool_call_deltas(
        slots,
        [ChoiceDeltaToolCall(index=0, id=None, type=None, function=DeltaFn(name=None, arguments=' "y"}'))],
    )
    out = _finalize_tool_calls(slots)
    assert json.loads(out[0]["function"]["arguments"]) == {"x": "y"}


# ---------- non-streaming loop ----------


@pytest.mark.asyncio
async def test_loop_returns_immediately_when_model_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        calls.append(kwargs)
        return _completion(finish="stop", content="hi there")

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pool = _FakePool(tool_names=["fetch_url"])
    out = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert out.choices[0].message.content == "hi there"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_loop_executes_mcp_tool_and_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _completion(finish="tool_calls", tool_calls=[("call_1", "fetch_url", '{"u":"x"}')]),
            _completion(finish="stop", content="fetched: ok"),
        ]
    )
    captured_messages: list[list[dict[str, Any]]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured_messages.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "fetch x"}]},
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    )

    assert out.choices[0].finish_reason == "stop"
    assert pool.calls == [("fetch_url", {"u": "x"})]
    # second call should have assistant tool_calls msg and tool result msg appended
    second_msgs = captured_messages[1]
    assert second_msgs[-2]["role"] == "assistant"
    assert second_msgs[-2]["tool_calls"][0]["function"]["name"] == "fetch_url"
    assert second_msgs[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


@pytest.mark.asyncio
async def test_loop_accumulates_usage_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _completion(finish="tool_calls", tool_calls=[("c1", "fetch_url", "{}")], prompt=10, completion_tokens=2),
            _completion(finish="stop", content="done", prompt=12, completion_tokens=3),
        ]
    )

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return next(responses)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    out = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
        pool=_FakePool(tool_names=["fetch_url"]),  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert out.usage is not None
    assert out.usage.prompt_tokens == 22
    assert out.usage.completion_tokens == 5
    assert out.usage.total_tokens == 27


@pytest.mark.asyncio
async def test_loop_max_iter_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion(finish="tool_calls", tool_calls=[("c", "fetch_url", "{}")])

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    with pytest.raises(MaxToolIterationsExceeded):
        await mcp_tool_loop(
            completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
            pool=_FakePool(tool_names=["fetch_url"]),  # type: ignore[arg-type]
            max_iterations=2,
        )


@pytest.mark.asyncio
async def test_loop_foreign_tool_returns_to_caller_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return _completion(finish="tool_calls", tool_calls=[("c", "user_tool", "{}")])

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pool = _FakePool(tool_names=["fetch_url"])  # doesn't own user_tool
    out = await mcp_tool_loop(
        completion_kwargs={
            "model": "fake",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "user_tool", "parameters": {}}}],
        },
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert out.choices[0].finish_reason == "tool_calls"
    assert pool.calls == []


@pytest.mark.asyncio
async def test_loop_handles_duck_typed_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ensure the loop dispatches tool_calls produced by *any* SDK whose
    function-tool-call class isn't the one any_llm aliases. The OpenAI SDK returns
    its own class; an isinstance check against any_llm's class would silently skip
    every call. The loop must duck-type."""

    from types import SimpleNamespace

    sdk_tool_call = SimpleNamespace(
        id="call_alien",
        type="function",
        function=SimpleNamespace(name="fetch_url", arguments='{"u":"x"}'),
    )
    first = _completion(finish="tool_calls")
    # Replace the test helper's any_llm-typed tool_calls with a foreign-class duck-typed object.
    first.choices[0].message.tool_calls = cast(Any, [sdk_tool_call])

    responses = iter([first, _completion(finish="stop", content="done")])

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        return next(responses)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    out = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "fetch"}]},
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert out.choices[0].finish_reason == "stop"
    assert pool.calls == [("fetch_url", {"u": "x"})], "loop must execute duck-typed function tool_calls"


@pytest.mark.asyncio
async def test_loop_tool_execution_failure_appears_as_tool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _completion(finish="tool_calls", tool_calls=[("c", "fetch_url", "{}")]),
            _completion(finish="stop", content="recovered"),
        ]
    )
    captured: list[list[dict[str, Any]]] = []

    async def fake_acompletion(**kwargs: Any) -> ChatCompletion:
        captured.append(kwargs["messages"])
        return next(responses)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    class FailingPool(_FakePool):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            raise RuntimeError("upstream down")

    pool = FailingPool(tool_names=["fetch_url"])
    out = await mcp_tool_loop(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    )
    assert out.choices[0].message.content == "recovered"
    tool_msg = captured[1][-1]
    assert tool_msg["role"] == "tool"
    assert "tool error" in tool_msg["content"]
    assert "upstream down" in tool_msg["content"]


# ---------- streaming loop ----------


async def _async_iter(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_loop_passes_chunks_through_and_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return _async_iter(_chunk(content="hi"), _chunk(content=" there", finish="stop"))

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pieces: list[str | None] = []
    async for c in mcp_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "hi"}]},
        pool=_FakePool(tool_names=[]),  # type: ignore[arg-type]
        max_iterations=3,
    ):
        pieces.append(c.choices[0].delta.content if c.choices else None)
    assert pieces == ["hi", " there"]


@pytest.mark.asyncio
async def test_stream_loop_runs_mcp_tool_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    iter_streams = iter(
        [
            _async_iter(
                _chunk(tool_calls=[(0, "call_1", "fetch_url", "{}")]),
                _chunk(finish="tool_calls"),
            ),
            _async_iter(_chunk(content="all done", finish="stop")),
        ]
    )

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        return next(iter_streams)

    monkeypatch.setattr(mcp_loop_module, "acompletion", fake_acompletion)

    pool = _FakePool(tool_names=["fetch_url"], results={"fetch_url": "ok"})
    finishes: list[str | None] = []
    async for c in mcp_tool_loop_stream(
        completion_kwargs={"model": "fake", "messages": [{"role": "user", "content": "go"}]},
        pool=pool,  # type: ignore[arg-type]
        max_iterations=5,
    ):
        if c.choices and c.choices[0].finish_reason:
            finishes.append(c.choices[0].finish_reason)
    assert finishes == ["tool_calls", "stop"]
    assert pool.calls == [("fetch_url", {})]
