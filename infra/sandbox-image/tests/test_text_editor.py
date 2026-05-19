"""Unit tests for sandbox.text_editor (post-multi-session refactor).

The handlers now take ``workspace_root`` and ``undo_stack`` explicitly so
two concurrent sessions cannot share state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.models import TextEditorInput
from sandbox.text_editor import run_text_editor


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def undo_stack() -> dict[str, list[str]]:
    return {}


async def test_create_writes_file(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    result = await run_text_editor(
        TextEditorInput(command="create", path="notes.md", file_text="# Hello\n"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert (workspace / "notes.md").read_text() == "# Hello\n"


async def test_create_refuses_overwrite(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.txt").write_text("existing")
    result = await run_text_editor(
        TextEditorInput(command="create", path="x.txt", file_text="new"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 1
    assert "already exists" in result.stderr


async def test_view_file(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "data.csv").write_text("a,b,c\n1,2,3\n")
    result = await run_text_editor(
        TextEditorInput(command="view", path="data.csv"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert "a,b,c" in result.stdout


async def test_view_with_range(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "code.py").write_text("line1\nline2\nline3\nline4\nline5\n")
    result = await run_text_editor(
        TextEditorInput(command="view", path="code.py", view_range=[2, 4]),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert result.stdout == "line2\nline3\nline4"


async def test_view_directory_listing(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "a.txt").write_text("a")
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.txt").write_text("b")
    result = await run_text_editor(
        TextEditorInput(command="view", path="."),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert "a.txt" in result.stdout
    assert "sub" in result.stdout


async def test_str_replace_unique(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("def foo():\n    return 1\n")
    result = await run_text_editor(
        TextEditorInput(
            command="str_replace",
            path="x.py",
            old_str="return 1",
            new_str="return 42",
        ),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert (workspace / "x.py").read_text() == "def foo():\n    return 42\n"


async def test_str_replace_rejects_ambiguous(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("a\na\n")
    result = await run_text_editor(
        TextEditorInput(command="str_replace", path="x.py", old_str="a", new_str="b"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 1
    assert "must be unique" in result.stderr


async def test_str_replace_missing(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("hello\n")
    result = await run_text_editor(
        TextEditorInput(
            command="str_replace", path="x.py", old_str="missing", new_str="x"
        ),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 1
    assert "not found" in result.stderr


async def test_insert_at_line(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("line1\nline2\n")
    result = await run_text_editor(
        TextEditorInput(
            command="insert", path="x.py", insert_line=1, new_str="inserted"
        ),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert (workspace / "x.py").read_text() == "line1\ninserted\nline2\n"


async def test_undo_after_str_replace(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("hello\n")
    await run_text_editor(
        TextEditorInput(
            command="str_replace", path="x.py", old_str="hello", new_str="world"
        ),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert (workspace / "x.py").read_text() == "world\n"
    result = await run_text_editor(
        TextEditorInput(command="undo_edit", path="x.py"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 0
    assert (workspace / "x.py").read_text() == "hello\n"


async def test_undo_with_no_history(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    (workspace / "x.py").write_text("hello\n")
    result = await run_text_editor(
        TextEditorInput(command="undo_edit", path="x.py"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 1
    assert "no edits" in result.stderr


async def test_path_escape_blocked(
    workspace: Path, undo_stack: dict[str, list[str]]
) -> None:
    result = await run_text_editor(
        TextEditorInput(command="view", path="/etc/passwd"),
        workspace_root=workspace,
        undo_stack=undo_stack,
    )
    assert result.return_code == 1
    assert "escapes workspace" in result.stderr


async def test_undo_stacks_are_independent(tmp_path: Path) -> None:
    """Two sessions with their own undo stacks cannot interfere."""
    ws_a = tmp_path / "a"
    ws_a.mkdir()
    ws_b = tmp_path / "b"
    ws_b.mkdir()
    (ws_a / "x.txt").write_text("v1")
    (ws_b / "x.txt").write_text("v1")
    stack_a: dict[str, list[str]] = {}
    stack_b: dict[str, list[str]] = {}

    await run_text_editor(
        TextEditorInput(command="str_replace", path="x.txt", old_str="v1", new_str="v2"),
        workspace_root=ws_a,
        undo_stack=stack_a,
    )
    # Session B has not edited anything; its undo stack stays empty.
    result = await run_text_editor(
        TextEditorInput(command="undo_edit", path="x.txt"),
        workspace_root=ws_b,
        undo_stack=stack_b,
    )
    assert result.return_code == 1
    # Session A's undo still works because the stack is independent.
    result = await run_text_editor(
        TextEditorInput(command="undo_edit", path="x.txt"),
        workspace_root=ws_a,
        undo_stack=stack_a,
    )
    assert result.return_code == 0
    assert (ws_a / "x.txt").read_text() == "v1"
