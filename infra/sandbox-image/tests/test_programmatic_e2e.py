"""End-to-end test of the programmatic tool calling path.

This is the load-bearing test for Option B: a real runner subprocess
imports a real generated ``_platform_tools`` module and makes a real
(loopback) HTTP call to a stub callback server. Post-Phase-2 we exercise
the path through a real :class:`SessionManager` so the per-session
isolation guarantee is also covered.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, TypeVar

import pytest

from sandbox import platform_tools as pt
from sandbox.sessions import SessionManager

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Stub callback server
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    """HTTP handler that records calls and replies via a configurable hook."""

    server_version = "SandboxStub/0.1"
    handler: Callable[[dict], dict] = lambda payload: {"content": None}  # type: ignore[assignment]
    calls: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        token = self.headers.get("Authorization", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        payload["_token"] = token
        type(self).calls.append(payload)
        try:
            response = type(self).handler(payload)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            response = {"is_error": True, "content": str(exc)}
        body_bytes = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)


@pytest.fixture
def stub_server() -> Iterator[tuple[str, type[_StubHandler]]]:
    class Handler(_StubHandler):
        calls: list[dict] = []
        handler: Callable[[dict], dict] = staticmethod(lambda p: {"content": None})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}/tool-call", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# End-to-end programmatic call (single session)
# ---------------------------------------------------------------------------


async def _setup_session_with_stub(
    sessions_root: Path,
    *,
    callback_url: str,
    token: str,
    tools: tuple[pt.ProgrammaticTool, ...],
):
    mgr = SessionManager(sessions_root=sessions_root)
    session = await mgr.create()
    pt.install_stubs(
        pt.StubConfig(callback_url=callback_url, token=token, tools=tools),
        stub_dir=session.stub_dir,
    )
    await session.runner.start()
    return mgr, session


def test_runner_can_call_a_programmatic_tool(
    sessions_root: Path,
    stub_server: tuple[str, type[_StubHandler]],
) -> None:
    callback_url, handler_cls = stub_server

    def respond(payload: dict) -> dict:
        assert payload["name"] == "echo"
        return {"content": {"received": payload["input"]}}

    handler_cls.handler = staticmethod(respond)

    async def body() -> None:
        mgr, session = await _setup_session_with_stub(
            sessions_root,
            callback_url=callback_url,
            token="phase-2-secret",
            tools=(pt.ProgrammaticTool(name="echo", doc="Echo back the input."),),
        )
        try:
            outcome = await session.runner.execute(
                "from _platform_tools import echo\n"
                "result = echo(query='hello world')\n"
                "import json\n"
                "print(json.dumps(result))\n",
                timeout_seconds=10,
            )
        finally:
            await mgr.shutdown()

        assert outcome.return_code == 0, outcome.stderr
        parsed = json.loads(outcome.stdout)
        assert parsed == {"received": {"query": "hello world"}}

        assert len(handler_cls.calls) == 1
        call = handler_cls.calls[0]
        assert call["_token"] == "Bearer phase-2-secret"
        assert call["name"] == "echo"

    _run(body())


def test_programmatic_tool_error_raises_in_python(
    sessions_root: Path,
    stub_server: tuple[str, type[_StubHandler]],
) -> None:
    callback_url, handler_cls = stub_server

    def respond(_payload: dict) -> dict:
        return {"is_error": True, "content": "permission denied"}

    handler_cls.handler = staticmethod(respond)

    async def body() -> None:
        mgr, session = await _setup_session_with_stub(
            sessions_root,
            callback_url=callback_url,
            token="phase-2",
            tools=(pt.ProgrammaticTool(name="risky"),),
        )
        try:
            outcome = await session.runner.execute(
                "from _platform_tools import risky, ToolError\n"
                "try:\n"
                "    risky()\n"
                "except ToolError as e:\n"
                "    print('caught:', e)\n",
                timeout_seconds=10,
            )
        finally:
            await mgr.shutdown()

        assert outcome.return_code == 0
        assert outcome.stdout == "caught: permission denied"

    _run(body())


def test_two_sessions_have_isolated_stubs(
    sessions_root: Path,
    stub_server: tuple[str, type[_StubHandler]],
) -> None:
    """A stub installed for session A must not be visible in session B."""
    callback_url, handler_cls = stub_server

    def respond(payload: dict) -> dict:
        return {"content": payload["name"]}

    handler_cls.handler = staticmethod(respond)

    async def body() -> None:
        mgr = SessionManager(sessions_root=sessions_root)
        try:
            session_a = await mgr.create()
            session_b = await mgr.create()
            # Install ``tool_a`` for session A only.
            pt.install_stubs(
                pt.StubConfig(
                    callback_url=callback_url,
                    token="t",
                    tools=(pt.ProgrammaticTool(name="tool_a"),),
                ),
                stub_dir=session_a.stub_dir,
            )
            await session_a.runner.start()
            await session_b.runner.start()

            outcome_a = await session_a.runner.execute(
                "from _platform_tools import tool_a\nprint(tool_a())",
                timeout_seconds=10,
            )
            assert outcome_a.return_code == 0
            assert outcome_a.stdout == "tool_a"

            # Session B should not be able to import the module at all.
            outcome_b = await session_b.runner.execute(
                "try:\n    import _platform_tools\n"
                "    print('imported')\n"
                "except ImportError:\n"
                "    print('blocked')\n",
                timeout_seconds=10,
            )
            assert outcome_b.return_code == 0
            assert outcome_b.stdout == "blocked"
        finally:
            await mgr.shutdown()

    _run(body())
