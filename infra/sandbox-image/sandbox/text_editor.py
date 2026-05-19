"""Handlers for the ``text_editor_code_execution`` tool.

Mirrors Anthropic's text-editor command set:

* ``view``       — read a file (or directory listing); supports ``view_range``.
* ``create``     — write a new file with ``file_text``; fails if exists.
* ``str_replace``— replace one occurrence of ``old_str`` with ``new_str``.
* ``insert``     — insert ``new_str`` after ``insert_line``.
* ``undo_edit``  — restore the previous version of a file edited by this tool.

All paths are resolved relative to the *session's* workspace root and a
path-escape check prevents the model from reaching outside the sandbox
filesystem. The undo stack is per-session (passed in by the caller) so
two concurrent sessions cannot interfere with each other's history.
"""

from __future__ import annotations

from pathlib import Path

from sandbox.models import TextEditorInput
from sandbox.runner_pool import ExecOutcome


def _resolve(path: str, workspace_root: Path) -> Path:
    """Resolve *path* under *workspace_root*, refusing escapes."""
    root = workspace_root.resolve()
    candidate = (root / path.lstrip("/")) if not Path(path).is_absolute() else Path(path)
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {path}") from exc
    return candidate


async def run_text_editor(
    req: TextEditorInput,
    *,
    workspace_root: Path,
    undo_stack: dict[str, list[str]],
) -> ExecOutcome:
    """Dispatch *req* to the matching handler and return an ExecOutcome."""
    try:
        if req.command == "view":
            return _handle_view(req, workspace_root)
        if req.command == "create":
            return _handle_create(req, workspace_root)
        if req.command == "str_replace":
            return _handle_str_replace(req, workspace_root, undo_stack)
        if req.command == "insert":
            return _handle_insert(req, workspace_root, undo_stack)
        if req.command == "undo_edit":
            return _handle_undo(req, workspace_root, undo_stack)
    except PermissionError as exc:
        return ExecOutcome(stdout="", stderr=f"text_editor: {exc}\n", return_code=1)
    except FileNotFoundError as exc:
        return ExecOutcome(stdout="", stderr=f"text_editor: {exc}\n", return_code=1)
    except FileExistsError as exc:
        return ExecOutcome(stdout="", stderr=f"text_editor: {exc}\n", return_code=1)
    except ValueError as exc:
        return ExecOutcome(stdout="", stderr=f"text_editor: {exc}\n", return_code=1)
    return ExecOutcome(
        stdout="",
        stderr=f"text_editor: unknown command {req.command}\n",
        return_code=1,
    )


def _handle_view(req: TextEditorInput, workspace_root: Path) -> ExecOutcome:
    target = _resolve(req.path, workspace_root)
    if target.is_dir():
        # Match Anthropic: emit a recursive listing capped to two levels.
        entries: list[str] = []
        for child in sorted(target.rglob("*")):
            try:
                rel = child.relative_to(target)
            except ValueError:
                continue
            if len(rel.parts) > 2:
                continue
            entries.append(str(rel))
        return ExecOutcome(stdout="\n".join(entries) + "\n", stderr="", return_code=0)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {req.path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    if req.view_range and len(req.view_range) == 2:
        start, end = req.view_range
        lines = text.splitlines()
        # Anthropic uses 1-indexed inclusive line numbers; -1 means EOF.
        if end == -1:
            end = len(lines)
        start = max(1, start)
        end = min(len(lines), end)
        text = "\n".join(lines[start - 1 : end])
    return ExecOutcome(stdout=text, stderr="", return_code=0)


def _handle_create(req: TextEditorInput, workspace_root: Path) -> ExecOutcome:
    if req.file_text is None:
        raise ValueError("create requires file_text")
    target = _resolve(req.path, workspace_root)
    if target.exists():
        raise FileExistsError(f"already exists: {req.path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.file_text, encoding="utf-8")
    return ExecOutcome(
        stdout=f"File created: {req.path}\n", stderr="", return_code=0
    )


def _handle_str_replace(
    req: TextEditorInput,
    workspace_root: Path,
    undo_stack: dict[str, list[str]],
) -> ExecOutcome:
    if req.old_str is None or req.new_str is None:
        raise ValueError("str_replace requires old_str and new_str")
    target = _resolve(req.path, workspace_root)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {req.path}")
    text = target.read_text(encoding="utf-8")
    occurrences = text.count(req.old_str)
    if occurrences == 0:
        raise ValueError(f"old_str not found in {req.path}")
    if occurrences > 1:
        raise ValueError(
            f"old_str matched {occurrences} times in {req.path}; must be unique"
        )
    _push_undo(target, text, undo_stack)
    target.write_text(text.replace(req.old_str, req.new_str), encoding="utf-8")
    return ExecOutcome(stdout=f"Edited {req.path}\n", stderr="", return_code=0)


def _handle_insert(
    req: TextEditorInput,
    workspace_root: Path,
    undo_stack: dict[str, list[str]],
) -> ExecOutcome:
    if req.new_str is None or req.insert_line is None:
        raise ValueError("insert requires new_str and insert_line")
    target = _resolve(req.path, workspace_root)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {req.path}")
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if req.insert_line < 0 or req.insert_line > len(lines):
        raise ValueError(
            f"insert_line {req.insert_line} out of range for {req.path}"
        )
    _push_undo(target, text, undo_stack)
    payload = req.new_str if req.new_str.endswith("\n") else req.new_str + "\n"
    lines.insert(req.insert_line, payload)
    target.write_text("".join(lines), encoding="utf-8")
    return ExecOutcome(stdout=f"Inserted into {req.path}\n", stderr="", return_code=0)


def _handle_undo(
    req: TextEditorInput,
    workspace_root: Path,
    undo_stack: dict[str, list[str]],
) -> ExecOutcome:
    target = _resolve(req.path, workspace_root)
    history = undo_stack.get(str(target))
    if not history:
        raise ValueError(f"no edits to undo for {req.path}")
    previous = history.pop()
    target.write_text(previous, encoding="utf-8")
    return ExecOutcome(stdout=f"Reverted {req.path}\n", stderr="", return_code=0)


def _push_undo(
    target: Path, previous_text: str, undo_stack: dict[str, list[str]]
) -> None:
    undo_stack.setdefault(str(target), []).append(previous_text)
