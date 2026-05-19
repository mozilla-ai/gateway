"""Pytest configuration shared across all sandbox-image tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sessions_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect SessionManager's per-session storage into a tmp dir.

    Tests that build a real :class:`SessionManager` (rather than passing
    individual workspace paths to lower-level helpers) request this
    fixture so the manager doesn't try to create directories under
    ``/var/sandbox`` on the developer's host.
    """
    root = tmp_path / "sandbox-sessions"
    root.mkdir()
    monkeypatch.setenv("SANDBOX_SESSIONS_ROOT", str(root))
    return root
