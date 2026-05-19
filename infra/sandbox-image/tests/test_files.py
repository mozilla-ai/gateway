"""Unit tests for sandbox.files (post-multi-session refactor).

HTTP-level tests for the session-scoped /files endpoints live in
test_exec_server.py — this file covers the lower-level pure functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.files import (
    MAX_FILE_BYTES,
    list_workspace,
    read_file,
    upload_file,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def test_upload_writes_under_workspace(workspace: Path) -> None:
    result = upload_file(b"hello", "greetings.txt", workspace_root=workspace)
    assert result.path == "greetings.txt"
    assert result.size_bytes == 5
    assert (workspace / "greetings.txt").read_bytes() == b"hello"


def test_upload_creates_parent_dirs(workspace: Path) -> None:
    upload_file(b"x", "nested/dir/file.bin", workspace_root=workspace)
    assert (workspace / "nested" / "dir" / "file.bin").read_bytes() == b"x"


def test_upload_refuses_path_escape(workspace: Path) -> None:
    with pytest.raises(PermissionError, match="escapes workspace"):
        upload_file(b"x", "/etc/passwd", workspace_root=workspace)


def test_upload_refuses_oversize(workspace: Path) -> None:
    payload = b"x" * (MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        upload_file(payload, "big.bin", workspace_root=workspace)


def test_read_returns_mime_type(workspace: Path) -> None:
    (workspace / "doc.html").write_text("<p>hi</p>")
    content, mime = read_file("doc.html", workspace_root=workspace)
    assert content == b"<p>hi</p>"
    assert mime == "text/html"


def test_read_missing_file_raises(workspace: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file("nope.txt", workspace_root=workspace)


def test_read_directory_raises(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    with pytest.raises(IsADirectoryError):
        read_file("sub", workspace_root=workspace)


def test_list_returns_only_files(workspace: Path) -> None:
    (workspace / "a.txt").write_text("a")
    (workspace / "b.txt").write_text("bb")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "c.txt").write_text("ccc")
    entries = list_workspace(".", workspace_root=workspace)
    paths = sorted(e.path for e in entries)
    assert paths == ["a.txt", "b.txt"]


def test_list_subdir(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "x.csv").write_text("x")
    entries = list_workspace("sub", workspace_root=workspace)
    assert len(entries) == 1
    assert entries[0].path == "sub/x.csv"


def test_list_missing_path_raises(workspace: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_workspace("missing", workspace_root=workspace)


def test_two_workspaces_are_isolated(tmp_path: Path) -> None:
    """Files written into one workspace are invisible from another."""
    ws_a = tmp_path / "a"
    ws_a.mkdir()
    ws_b = tmp_path / "b"
    ws_b.mkdir()
    upload_file(b"hello-a", "greeting.txt", workspace_root=ws_a)
    # Workspace B has no greeting.txt.
    with pytest.raises(FileNotFoundError):
        read_file("greeting.txt", workspace_root=ws_b)
    # And listing each shows only its own file.
    assert [e.path for e in list_workspace(".", workspace_root=ws_a)] == ["greeting.txt"]
    assert list_workspace(".", workspace_root=ws_b) == []
