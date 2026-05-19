"""Pydantic shapes for the sandbox HTTP API.

The result blocks returned by ``POST /exec`` deliberately match Anthropic's
``code_execution_20250825`` content-block formats so consumers that already
parse Anthropic shapes (e.g. octonous) work without translation.

Reference shapes:

* ``code_execution_tool_result`` -> Python REPL execution
* ``bash_code_execution_tool_result`` -> bash one-shot
* ``text_editor_code_execution_tool_result`` -> file view/create/edit
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tool kinds
# ---------------------------------------------------------------------------

ToolKind = Literal[
    "code_execution",
    "bash_code_execution",
    "text_editor_code_execution",
]

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CodeExecutionInput(BaseModel):
    """Input for the ``code_execution`` tool (Python REPL)."""

    code: str = Field(min_length=1, max_length=200_000)


class BashExecutionInput(BaseModel):
    """Input for the ``bash_code_execution`` tool."""

    command: str = Field(min_length=1, max_length=200_000)


class TextEditorInput(BaseModel):
    """Input for the ``text_editor_code_execution`` tool.

    Mirrors Anthropic's text-editor command set. ``file_text``, ``old_str``,
    ``new_str`` and ``insert_line`` are command-specific and validated at
    handler time.
    """

    command: Literal["view", "create", "str_replace", "insert", "undo_edit"]
    path: str
    file_text: str | None = None
    old_str: str | None = None
    new_str: str | None = None
    insert_line: int | None = None
    view_range: list[int] | None = None


class ExecRequest(BaseModel):
    """Top-level request body for ``POST /exec``.

    The ``input`` field is parsed by the handler for the chosen ``tool``;
    we accept it as a free-form dict here to keep the wire shape uniform
    across tool kinds and to match Anthropic's tool_use input shape.
    """

    tool: ToolKind
    input: dict[str, Any]
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    tool_use_id: str | None = None


# ---------------------------------------------------------------------------
# Result block shapes (match Anthropic's content blocks)
# ---------------------------------------------------------------------------


class CodeExecutionFileRef(BaseModel):
    """A file produced during code execution.

    Matches the entries Anthropic emits inside ``result.content`` for files
    the sandbox produced (e.g. matplotlib charts, generated CSVs).
    """

    type: Literal["code_execution_output"] = "code_execution_output"
    file_id: str
    filename: str


class CodeExecutionResultContent(BaseModel):
    """The ``content`` payload of a ``code_execution_tool_result`` block.

    Octonous's ``callbacks.py`` reads this exact shape: ``stdout``, ``stderr``,
    ``return_code`` and a list of file refs under ``content``.
    """

    type: Literal["code_execution_result"] = "code_execution_result"
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    content: list[CodeExecutionFileRef] = Field(default_factory=list)


class CodeExecutionResultBlock(BaseModel):
    """``code_execution_tool_result`` content block."""

    type: Literal["code_execution_tool_result"] = "code_execution_tool_result"
    tool_use_id: str
    content: CodeExecutionResultContent


class BashExecutionResultBlock(BaseModel):
    """``bash_code_execution_tool_result`` content block."""

    type: Literal["bash_code_execution_tool_result"] = "bash_code_execution_tool_result"
    tool_use_id: str
    content: CodeExecutionResultContent


class TextEditorResultBlock(BaseModel):
    """``text_editor_code_execution_tool_result`` content block.

    For ``create`` commands, the file content is *not* re-emitted here — it
    already lives inline on the original tool_use block as ``input.file_text``.
    Octonous's ``_extract_inline_file_from_text_editor`` reads it from there.
    """

    type: Literal["text_editor_code_execution_tool_result"] = (
        "text_editor_code_execution_tool_result"
    )
    tool_use_id: str
    content: CodeExecutionResultContent


ResultBlock = (
    CodeExecutionResultBlock | BashExecutionResultBlock | TextEditorResultBlock
)


class ExecResponse(BaseModel):
    """Top-level response from ``POST /exec``."""

    tool_use_id: str
    result_block: ResultBlock
    execution_time_ms: int
