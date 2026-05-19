"""Tests for SessionManager — multi-session isolation, GC, lifetime."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest

from sandbox.sessions import (
    SessionLimitExceededError,
    SessionManager,
    SessionNotFoundError,
)

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    return asyncio.new_event_loop().run_until_complete(coro)


def _new_manager(sessions_root: Path, *, max_sessions: int = 10) -> SessionManager:
    return SessionManager(sessions_root=sessions_root, max_sessions=max_sessions)


def test_create_and_get(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create()
            assert session.session_id.startswith("sbx_")
            assert session.workspace_dir.exists()
            assert session.stub_dir.exists()
            same = mgr.get(session.session_id)
            assert same is session
        finally:
            await mgr.shutdown()

    _run(body())


def test_get_unknown_session_raises(sessions_root: Path) -> None:
    mgr = _new_manager(sessions_root)
    with pytest.raises(SessionNotFoundError):
        mgr.get("sbx_does_not_exist")


def test_destroy_removes_directory(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        session = await mgr.create()
        session_dir = session.workspace_dir.parent
        assert session_dir.exists()
        await mgr.destroy(session.session_id)
        assert not session_dir.exists()
        with pytest.raises(SessionNotFoundError):
            mgr.get(session.session_id)

    _run(body())


def test_destroy_unknown_is_noop(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        await mgr.destroy("never-existed")  # must not raise

    _run(body())


def test_max_sessions_enforced(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root, max_sessions=2)
        try:
            await mgr.create()
            await mgr.create()
            with pytest.raises(SessionLimitExceededError):
                await mgr.create()
        finally:
            await mgr.shutdown()

    _run(body())


def test_two_sessions_have_isolated_workspaces(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            a = await mgr.create()
            b = await mgr.create()
            assert a.workspace_dir != b.workspace_dir
            assert a.stub_dir != b.stub_dir

            (a.workspace_dir / "secret-a.txt").write_text("only-in-a")
            (b.workspace_dir / "secret-b.txt").write_text("only-in-b")

            assert (a.workspace_dir / "secret-a.txt").exists()
            assert not (a.workspace_dir / "secret-b.txt").exists()
            assert (b.workspace_dir / "secret-b.txt").exists()
            assert not (b.workspace_dir / "secret-a.txt").exists()
        finally:
            await mgr.shutdown()

    _run(body())


def test_two_sessions_have_isolated_python_state(sessions_root: Path) -> None:
    """Variables defined in one session must not leak into another."""

    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            a = await mgr.create()
            b = await mgr.create()
            await a.runner.start()
            await b.runner.start()

            await a.runner.execute("session_marker = 'A'", timeout_seconds=5)
            outcome_b = await b.runner.execute(
                "print('marker' in dir())", timeout_seconds=5
            )
            assert outcome_b.return_code == 0
            assert outcome_b.stdout == "False"

            outcome_a = await a.runner.execute(
                "print(session_marker)", timeout_seconds=5
            )
            assert outcome_a.return_code == 0
            assert outcome_a.stdout == "A"
        finally:
            await mgr.shutdown()

    _run(body())


def test_runner_uses_session_workspace_as_cwd(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create()
            await session.runner.start()
            outcome = await session.runner.execute(
                "import os; print(os.getcwd())", timeout_seconds=5
            )
            assert outcome.return_code == 0
            # Resolve symlinks (macOS /private prefix) before comparing.
            assert (
                Path(outcome.stdout).resolve()
                == session.workspace_dir.resolve()
            )
        finally:
            await mgr.shutdown()

    _run(body())


def test_touch_updates_last_activity(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create()
            original = session.last_activity_at
            await asyncio.sleep(0.01)
            session.touch()
            assert session.last_activity_at > original
        finally:
            await mgr.shutdown()

    _run(body())


def test_idle_session_is_reaped(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create(idle_timeout_seconds=1)
            sid = session.session_id
            # Force the session to look idle by backdating its activity.
            session.last_activity_at = time.time() - 5
            destroyed = await mgr.reap_idle()
            assert sid in destroyed
            with pytest.raises(SessionNotFoundError):
                mgr.get(sid)
        finally:
            await mgr.shutdown()

    _run(body())


def test_expired_session_is_reaped(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create(max_lifetime_seconds=1)
            sid = session.session_id
            session.created_at = time.time() - 5
            destroyed = await mgr.reap_idle()
            assert sid in destroyed
        finally:
            await mgr.shutdown()

    _run(body())


def test_active_session_is_not_reaped(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        try:
            session = await mgr.create(idle_timeout_seconds=300)
            destroyed = await mgr.reap_idle()
            assert destroyed == []
            assert mgr.get(session.session_id) is session
        finally:
            await mgr.shutdown()

    _run(body())


def test_shutdown_destroys_everything(sessions_root: Path) -> None:
    async def body() -> None:
        mgr = _new_manager(sessions_root)
        a = await mgr.create()
        b = await mgr.create()
        await mgr.shutdown()
        assert mgr.count() == 0
        assert not a.workspace_dir.exists()
        assert not b.workspace_dir.exists()

    _run(body())
