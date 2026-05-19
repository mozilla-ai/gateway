"""Dispatch `code_execution` tool calls to a sandbox container.

A backend the tool-use loop in :mod:`gateway.services.mcp_loop` dispatches
to whenever the model emits a ``code_execution(code=…)`` call. The sandbox
container is a separate service (built from ``infra/sandbox-image/`` or
pulled from a published image) that runs a Python REPL with a curated set
of data-science libraries pre-installed.

Wire shape against the sandbox container:

* ``POST /sessions``         → creates a session, returns ``session_id``
* ``POST /sessions/{id}/exec``  with ``{tool: "code_execution",
                                        input: {code: "…"},
                                        timeout_seconds: int}``
                              → returns ``{result_block: {…}}``
* ``DELETE /sessions/{id}``  → tears the session down

Session lifecycle is per-request: enter creates a session, exit
destroys it. State does not persist across separate chat-completion
requests in this minimum-viable backend. A future stateful variant
(per-conversation session affinity, warm pool, etc.) is the platform's
problem — see ``docs/sandbox-oss-platform-direction.md`` in the
private platform repo for that picture.

This backend satisfies the same duck-typed protocol the MCP loop uses
for tool dispatch (``openai_tools``, ``owns_tool``, ``purpose_hints``,
``call_tool``), so the loop accepts it as a ``pool`` without any
refactor to :func:`gateway.services.mcp_loop.mcp_tool_loop`.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)

CODE_EXECUTION_TOOL_NAME = "code_execution"
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_PURPOSE_HINT = (
    "Prefer `code_execution` for any computation, data analysis, date "
    "arithmetic, statistics, or anything that benefits from exact output. "
    "Python with numpy/pandas/scipy/sympy/matplotlib pre-installed."
)


class SandboxNotReachableError(RuntimeError):
    """Raised when the sandbox container can't be reached or returns malformed data."""


class SandboxBackend:
    """Async context manager that owns one sandbox session for a request's lifetime.

    Usage::

        async with SandboxBackend(sandbox_url="http://sandbox:8080") as backend:
            # backend duck-types as the MCP loop's `pool` parameter
            result = await mcp_tool_loop(
                completion_kwargs=kwargs, pool=backend, max_iterations=N,
            )
    """

    def __init__(
        self,
        *,
        sandbox_url: str,
        purpose_hint: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._sandbox_url = sandbox_url.rstrip("/")
        self._purpose_hint = purpose_hint or _DEFAULT_PURPOSE_HINT
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._stack: AsyncExitStack = AsyncExitStack()

    async def __aenter__(self) -> SandboxBackend:
        try:
            self._client = await self._stack.enter_async_context(httpx.AsyncClient(timeout=self._timeout_s))
            response = await self._client.post(f"{self._sandbox_url}/sessions", json={})
            response.raise_for_status()
            self._session_id = response.json()["session_id"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            await self._stack.aclose()
            raise SandboxNotReachableError(f"failed to create sandbox session at {self._sandbox_url}: {exc}") from exc
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._client is not None and self._session_id is not None:
            try:
                await self._client.delete(f"{self._sandbox_url}/sessions/{self._session_id}")
            except httpx.HTTPError:
                logger.warning("sandbox session %s cleanup failed", self._session_id, exc_info=True)
        await self._stack.aclose()

    # ----- duck-typed protocol the MCP loop uses on `pool` -----

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": CODE_EXECUTION_TOOL_NAME,
                    "description": (
                        "Execute Python code in a sandboxed REPL. Returns stdout, "
                        "stderr, and any rich result blocks. State persists across "
                        "calls within the same request."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def owns_tool(self, name: str) -> bool:
        return name == CODE_EXECUTION_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return [(CODE_EXECUTION_TOOL_NAME, self._purpose_hint)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name != CODE_EXECUTION_TOOL_NAME:
            raise KeyError(f"SandboxBackend does not own tool {name!r}")
        if self._client is None or self._session_id is None:
            raise RuntimeError("SandboxBackend not entered as an async context manager")

        code = arguments.get("code") or ""
        payload = {
            "tool": CODE_EXECUTION_TOOL_NAME,
            "input": {"code": code},
            "timeout_seconds": int(self._timeout_s),
        }
        try:
            response = await self._client.post(
                f"{self._sandbox_url}/sessions/{self._session_id}/exec",
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SandboxNotReachableError(f"sandbox exec failed: {exc}") from exc

        result_block = body.get("result_block")
        if not isinstance(result_block, dict):
            raise SandboxNotReachableError(f"sandbox returned malformed result: {body!r}")

        return _flatten_result_block(result_block)


def _flatten_result_block(block: dict[str, Any]) -> str:
    """Render the sandbox's structured result as a single string for the model.

    The sandbox returns Anthropic-shaped ``code_execution_tool_result`` content
    blocks with stdout/stderr/return_code fields. We collapse that into a plain
    string the model can read; full structured output is a future enhancement
    that lands alongside the Anthropic-content-block lift.
    """
    is_error = bool(block.get("is_error"))
    content = block.get("content", [])
    if not isinstance(content, list):
        return str(block)

    parts: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            parts.append(str(entry))
            continue
        kind = entry.get("type")
        if kind == "text":
            text = entry.get("text", "")
            if text:
                parts.append(text)
            continue
        # code_execution_output and similar carry stdout/stderr/return_code
        stdout = entry.get("stdout") or ""
        stderr = entry.get("stderr") or ""
        rc = entry.get("return_code")
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if rc not in (None, 0):
            parts.append(f"return_code: {rc}")

    flattened = "\n".join(p for p in parts if p)
    if is_error:
        return f"[tool error] {flattened}"
    return flattened
