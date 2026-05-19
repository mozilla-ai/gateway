"""Multi-session sandbox state.

A single sandbox container hosts many isolated *sessions*, one per agent
run. Each session owns:

* its own ``RunnerProcess`` (long-lived Python REPL subprocess),
* its own ``/workspace`` directory under
  ``<SANDBOX_BASE>/sessions/<session_id>/workspace``,
* its own ``_platform_tools`` stub module under
  ``<SANDBOX_BASE>/sessions/<session_id>/lib`` so different sessions
  cannot import each other's stubs even though they share an interpreter
  parent process,
* its own text-editor undo stack,
* timestamps for idle / lifetime garbage collection.

The :class:`SessionManager` is the only entry point — handlers reach
into a session via :meth:`SessionManager.get` and never touch the
``Session`` constructor directly. ``ensure_default`` is provided as a
convenience so the implicit single-session mode of Phase 1 can be
emulated by tests that do not care about isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sandbox.runner_pool import RunnerProcess

logger = logging.getLogger(__name__)


# Default lifetimes — can be overridden via env vars on the container.
DEFAULT_IDLE_TIMEOUT_SECONDS = int(
    os.environ.get("SANDBOX_SESSION_IDLE_TIMEOUT", "900")  # 15 min
)
DEFAULT_MAX_LIFETIME_SECONDS = int(
    os.environ.get("SANDBOX_SESSION_MAX_LIFETIME", "7200")  # 2 hr
)
DEFAULT_GC_INTERVAL_SECONDS = int(
    os.environ.get("SANDBOX_SESSION_GC_INTERVAL", "30")
)


def _sessions_root() -> Path:
    """Base directory under which all per-session state lives.

    Defaults to ``/var/sandbox`` so the runtime image can mount a tmpfs
    or ephemeral volume there. In tests this is overridden via the
    ``SANDBOX_SESSIONS_ROOT`` env var.
    """
    return Path(os.environ.get("SANDBOX_SESSIONS_ROOT", "/var/sandbox/sessions")).resolve()


class SessionNotFoundError(Exception):
    """Raised when a request references a session id that does not exist."""


class SessionLimitExceededError(Exception):
    """Raised when ``SessionManager`` rejects a new session due to a hard cap."""


@dataclass
class Session:
    """One isolated sandbox session.

    The text-editor undo stack lives here so it is per-session, not global.
    Files written via the files API end up under ``workspace_dir`` and
    user Python code spawned by ``runner`` uses the same directory as
    its CWD.
    """

    session_id: str
    workspace_dir: Path
    stub_dir: Path
    runner: RunnerProcess
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_lifetime_seconds: int = DEFAULT_MAX_LIFETIME_SECONDS
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    undo_stack: dict[str, list[str]] = field(default_factory=dict)
    """Per-session text-editor undo history keyed by absolute path."""

    def touch(self) -> None:
        """Mark the session as active so the GC does not reap it."""
        self.last_activity_at = time.time()

    def is_idle(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        return (now - self.last_activity_at) > self.idle_timeout_seconds

    def is_expired(self, now: float | None = None) -> bool:
        if now is None:
            now = time.time()
        return (now - self.created_at) > self.max_lifetime_seconds


class SessionManager:
    """Owns every active session in this sandbox container.

    Thread-/task-safe via a single asyncio lock — concurrent
    ``create``/``destroy`` calls are serialised. Per-session ``/exec``
    serialisation is handled by ``RunnerProcess``'s own lock.
    """

    def __init__(
        self,
        *,
        sessions_root: Path | None = None,
        max_sessions: int = 256,
    ) -> None:
        self._root = sessions_root or _sessions_root()
        self._max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ create / destroy

    async def create(
        self,
        *,
        idle_timeout_seconds: int | None = None,
        max_lifetime_seconds: int | None = None,
    ) -> Session:
        """Allocate a new session and start its REPL."""
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise SessionLimitExceededError(
                    f"sandbox holds {len(self._sessions)} sessions "
                    f"(max={self._max_sessions})"
                )
            session_id = f"sbx_{uuid.uuid4().hex[:24]}"
            session_dir = self._root / session_id
            workspace_dir = session_dir / "workspace"
            stub_dir = session_dir / "lib"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            stub_dir.mkdir(parents=True, exist_ok=True)

            runner = RunnerProcess(
                workspace=str(workspace_dir),
                extra_python_path=str(stub_dir),
            )
            session = Session(
                session_id=session_id,
                workspace_dir=workspace_dir,
                stub_dir=stub_dir,
                runner=runner,
                idle_timeout_seconds=idle_timeout_seconds
                or DEFAULT_IDLE_TIMEOUT_SECONDS,
                max_lifetime_seconds=max_lifetime_seconds
                or DEFAULT_MAX_LIFETIME_SECONDS,
            )
            self._sessions[session_id] = session
            logger.info("session created id=%s", session_id)
            return session

    async def destroy(self, session_id: str) -> None:
        """Tear down a session: stop the runner and remove its directories."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            await session.runner.stop()
        except Exception:  # noqa: BLE001 - best effort cleanup
            logger.warning(
                "failed to stop runner for session %s", session_id, exc_info=True
            )
        # Remove the per-session directory tree.
        session_dir = session.workspace_dir.parent
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.warning(
                "failed to remove session dir %s", session_dir, exc_info=True
            )
        logger.info("session destroyed id=%s", session_id)

    # ------------------------------------------------------------------ access

    def get(self, session_id: str) -> Session:
        """Look up a session, raising :class:`SessionNotFoundError` if missing."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------ gc

    async def reap_idle(self) -> list[str]:
        """Destroy sessions that have exceeded idle or lifetime limits.

        Returns the list of destroyed session IDs.
        """
        now = time.time()
        to_destroy: list[str] = []
        async with self._lock:
            for sid, session in self._sessions.items():
                if session.is_idle(now) or session.is_expired(now):
                    to_destroy.append(sid)
        # Destroy outside the lock so each destroy reacquires it cleanly.
        for sid in to_destroy:
            await self.destroy(sid)
            logger.info("session reaped id=%s", sid)
        return to_destroy

    async def shutdown(self) -> None:
        """Destroy every session — used on container shutdown."""
        ids = self.list_ids()
        for sid in ids:
            await self.destroy(sid)
