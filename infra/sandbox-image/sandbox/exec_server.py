"""FastAPI app that drives the in-pod sandbox.

Phase 2: multi-session. A single container hosts many isolated sessions
keyed by ``session_id``; each session has its own Python REPL,
``/workspace`` directory, and ``_platform_tools`` stub module. Sessions
are reaped by a background GC task once they exceed their idle or
lifetime limits.

Endpoints:

* ``POST /sessions`` — create a new session and return its id.
* ``GET /sessions/{id}`` — session metadata.
* ``DELETE /sessions/{id}`` — destroy a session and free its resources.
* ``POST /sessions/{id}/exec`` — run a tool call in this session.
* ``POST /sessions/{id}/programmatic-tools/register`` — install the
  ``_platform_tools`` stub for this session.
* ``POST /sessions/{id}/files`` — upload a file into this session's
  workspace.
* ``GET /sessions/{id}/files?path=`` — download a file from this session.
* ``GET /sessions/{id}/files/list?path=`` — list files in this session.
* ``GET /health`` — liveness probe (reports active session count).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from sandbox.bash import run_bash
from sandbox.files import (
    MAX_FILE_BYTES,
    list_workspace,
    read_file,
    upload_file,
)
from sandbox.models import (
    BashExecutionInput,
    BashExecutionResultBlock,
    CodeExecutionInput,
    CodeExecutionResultBlock,
    CodeExecutionResultContent,
    ExecRequest,
    ExecResponse,
    TextEditorInput,
    TextEditorResultBlock,
)
from sandbox.platform_tools import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    ProgrammaticTool,
    StubConfig,
    install_stubs,
)
from sandbox.runner_pool import ExecOutcome
from sandbox.sessions import (
    DEFAULT_GC_INTERVAL_SECONDS,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_LIFETIME_SECONDS,
    Session,
    SessionLimitExceededError,
    SessionManager,
    SessionNotFoundError,
)
from sandbox.text_editor import run_text_editor

logger = logging.getLogger(__name__)

# Module-level singleton — the lifespan context starts/stops it.
_session_manager = SessionManager()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the GC background task and tear down all sessions on shutdown."""
    gc_task = asyncio.create_task(_gc_loop())
    try:
        yield
    finally:
        gc_task.cancel()
        try:
            await gc_task
        except asyncio.CancelledError:
            pass
        await _session_manager.shutdown()


async def _gc_loop() -> None:
    """Background task: reap idle / over-lifetime sessions periodically."""
    while True:
        try:
            destroyed = await _session_manager.reap_idle()
            if destroyed:
                logger.info("GC reaped %d session(s): %s", len(destroyed), destroyed)
        except Exception:  # noqa: BLE001 - background loop must keep going
            logger.exception("session GC failed")
        await asyncio.sleep(DEFAULT_GC_INTERVAL_SECONDS)


app = FastAPI(title="gateway code-execution sandbox", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "session_count": _session_manager.count(),
        "max_sessions": _session_manager._max_sessions,  # noqa: SLF001 - debug
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Optional overrides for new session lifetimes."""

    idle_timeout_seconds: int | None = Field(
        default=None, ge=1, le=DEFAULT_MAX_LIFETIME_SECONDS
    )
    max_lifetime_seconds: int | None = Field(
        default=None, ge=1, le=DEFAULT_MAX_LIFETIME_SECONDS
    )


class SessionPublic(BaseModel):
    """Wire shape for session metadata."""

    session_id: str
    created_at: float
    last_activity_at: float
    idle_timeout_seconds: int
    max_lifetime_seconds: int


def _to_public(session: Session) -> SessionPublic:
    return SessionPublic(
        session_id=session.session_id,
        created_at=session.created_at,
        last_activity_at=session.last_activity_at,
        idle_timeout_seconds=session.idle_timeout_seconds,
        max_lifetime_seconds=session.max_lifetime_seconds,
    )


@app.post("/sessions", response_model=SessionPublic, status_code=201)
async def create_session(
    request: CreateSessionRequest | None = None,
) -> SessionPublic:
    """Allocate a fresh sandbox session and start its REPL."""
    request = request or CreateSessionRequest()
    try:
        session = await _session_manager.create(
            idle_timeout_seconds=request.idle_timeout_seconds,
            max_lifetime_seconds=request.max_lifetime_seconds,
        )
    except SessionLimitExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Eagerly start the runner so the first /exec call is warm.
    await session.runner.start()
    return _to_public(session)


@app.get("/sessions/{session_id}", response_model=SessionPublic)
async def get_session(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> SessionPublic:
    try:
        session = _session_manager.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_public(session)


@app.delete("/sessions/{session_id}", status_code=204)
async def destroy_session(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> Response:
    await _session_manager.destroy(session_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------


def _require_session(session_id: str) -> Session:
    try:
        return _session_manager.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/exec", response_model=ExecResponse)
async def exec_tool(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
    request: ExecRequest,
) -> ExecResponse:
    """Dispatch a tool call against the named session."""
    session = _require_session(session_id)
    session.touch()
    tool_use_id = request.tool_use_id or f"srvtoolu_{uuid.uuid4().hex[:24]}"
    started = time.monotonic()

    if request.tool == "code_execution":
        outcome = await _run_code_execution(session, request)
        block = CodeExecutionResultBlock(
            tool_use_id=tool_use_id,
            content=_outcome_to_content(outcome),
        )
    elif request.tool == "bash_code_execution":
        outcome = await _run_bash_execution(session, request)
        block = BashExecutionResultBlock(
            tool_use_id=tool_use_id,
            content=_outcome_to_content(outcome),
        )
    elif request.tool == "text_editor_code_execution":
        outcome = await _run_text_editor_execution(session, request)
        block = TextEditorResultBlock(
            tool_use_id=tool_use_id,
            content=_outcome_to_content(outcome),
        )
    else:  # pragma: no cover - validated by pydantic
        raise HTTPException(status_code=400, detail=f"unknown tool: {request.tool}")

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return ExecResponse(
        tool_use_id=tool_use_id,
        result_block=block,
        execution_time_ms=elapsed_ms,
    )


async def _run_code_execution(session: Session, request: ExecRequest) -> ExecOutcome:
    try:
        parsed = CodeExecutionInput.model_validate(request.input)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return await session.runner.execute(parsed.code, timeout_seconds=request.timeout_seconds)


async def _run_bash_execution(session: Session, request: ExecRequest) -> ExecOutcome:
    try:
        parsed = BashExecutionInput.model_validate(request.input)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    # Bash inherits the session's workspace as its CWD via SANDBOX_WORKSPACE,
    # which we set per-call so concurrent sessions stay isolated.
    import os as _os

    prev = _os.environ.get("SANDBOX_WORKSPACE")
    _os.environ["SANDBOX_WORKSPACE"] = str(session.workspace_dir)
    try:
        return await run_bash(parsed.command, timeout_seconds=request.timeout_seconds)
    finally:
        if prev is None:
            _os.environ.pop("SANDBOX_WORKSPACE", None)
        else:
            _os.environ["SANDBOX_WORKSPACE"] = prev


async def _run_text_editor_execution(
    session: Session, request: ExecRequest
) -> ExecOutcome:
    try:
        parsed = TextEditorInput.model_validate(request.input)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return await run_text_editor(
        parsed,
        workspace_root=session.workspace_dir,
        undo_stack=session.undo_stack,
    )


def _outcome_to_content(outcome: ExecOutcome) -> CodeExecutionResultContent:
    return CodeExecutionResultContent(
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        return_code=outcome.return_code,
        content=[],
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class FileMetadataResponse(BaseModel):
    path: str
    size_bytes: int
    mime_type: str | None = None
    modified_at: float


class FileListResponse(BaseModel):
    files: list[FileMetadataResponse]


class FileUploadResponse(BaseModel):
    path: str
    size_bytes: int


@app.post(
    "/sessions/{session_id}/files",
    response_model=FileUploadResponse,
    status_code=201,
)
async def files_upload(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
    file: UploadFile = File(...),
    path: str | None = Form(default=None),
) -> FileUploadResponse:
    session = _require_session(session_id)
    session.touch()
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit",
        )
    dest = path or file.filename or "upload.bin"
    try:
        result = upload_file(
            content, dest, workspace_root=session.workspace_dir
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return FileUploadResponse(path=result.path, size_bytes=result.size_bytes)


@app.get("/sessions/{session_id}/files")
async def files_download(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
    path: str = Query(...),
) -> Response:
    session = _require_session(session_id)
    session.touch()
    try:
        content, mime = read_file(path, workspace_root=session.workspace_dir)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IsADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type=mime)


@app.get("/sessions/{session_id}/files/list", response_model=FileListResponse)
async def files_list(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
    path: str = Query(default="."),
) -> FileListResponse:
    session = _require_session(session_id)
    session.touch()
    try:
        entries = list_workspace(path, workspace_root=session.workspace_dir)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileListResponse(
        files=[
            FileMetadataResponse(
                path=e.path,
                size_bytes=e.size_bytes,
                mime_type=e.mime_type,
                modified_at=e.modified_at,
            )
            for e in entries
        ]
    )


# ---------------------------------------------------------------------------
# Programmatic tool registration
# ---------------------------------------------------------------------------


class ProgrammaticToolDescriptor(BaseModel):
    name: str
    doc: str = ""


class RegisterProgrammaticToolsRequest(BaseModel):
    callback_url: str
    token: str
    tools: list[ProgrammaticToolDescriptor] = Field(default_factory=list)
    timeout_seconds: int = Field(
        default=DEFAULT_CALL_TIMEOUT_SECONDS, ge=1, le=3600
    )


class RegisterProgrammaticToolsResponse(BaseModel):
    stub_path: str
    tool_count: int


@app.post(
    "/sessions/{session_id}/programmatic-tools/register",
    response_model=RegisterProgrammaticToolsResponse,
)
async def register_programmatic_tools(
    session_id: Annotated[str, Path(min_length=1, max_length=64)],
    request: RegisterProgrammaticToolsRequest,
) -> RegisterProgrammaticToolsResponse:
    """Generate and install the ``_platform_tools`` stub for this session."""
    session = _require_session(session_id)
    session.touch()
    config = StubConfig(
        callback_url=request.callback_url,
        token=request.token,
        tools=tuple(
            ProgrammaticTool(name=t.name, doc=t.doc) for t in request.tools
        ),
        timeout_seconds=request.timeout_seconds,
    )
    try:
        path = install_stubs(config, stub_dir=session.stub_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RegisterProgrammaticToolsResponse(
        stub_path=str(path), tool_count=len(request.tools)
    )
