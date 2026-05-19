"""File operations on a session's workspace directory.

Used by the per-session ``/files`` endpoints. Path resolution shares the
same workspace-escape guard as the text editor — files outside the
session's workspace are refused.

These endpoints are *not* the platform Files API. They are the in-pod
side that the platform's Files service calls when attaching a file to a
session, or when the user downloads a file the model produced.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

# Hard cap on a single file upload — keeps a malicious or buggy caller from
# filling the pod's ephemeral storage. Matches Anthropic's documented limit.
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


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


@dataclass(frozen=True)
class FileMetadata:
    """Metadata for one file in the workspace."""

    path: str
    size_bytes: int
    mime_type: str | None
    modified_at: float  # POSIX timestamp


@dataclass(frozen=True)
class UploadResult:
    path: str
    size_bytes: int


def upload_file(
    content: bytes, dest_path: str, *, workspace_root: Path
) -> UploadResult:
    """Write *content* to ``dest_path`` under *workspace_root*.

    Parent directories are created on demand. Refuses to write outside the
    workspace and refuses uploads larger than :data:`MAX_FILE_BYTES`.
    """
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(
            f"file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit "
            f"({len(content)} bytes)"
        )
    target = _resolve(dest_path, workspace_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return UploadResult(
        path=str(target.relative_to(workspace_root.resolve())),
        size_bytes=len(content),
    )


def read_file(path: str, *, workspace_root: Path) -> tuple[bytes, str]:
    """Read a file from *workspace_root* and return ``(bytes, mime_type)``."""
    target = _resolve(path, workspace_root)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if target.is_dir():
        raise IsADirectoryError(f"is a directory: {path}")
    mime, _ = mimetypes.guess_type(target.name)
    return target.read_bytes(), mime or "application/octet-stream"


def list_workspace(
    path: str = ".", *, workspace_root: Path
) -> list[FileMetadata]:
    """Return metadata for every file under *path* (non-recursive on dirs)."""
    target = _resolve(path, workspace_root)
    if not target.exists():
        raise FileNotFoundError(f"path not found: {path}")
    if target.is_file():
        return [_metadata_for(target, workspace_root)]
    files: list[FileMetadata] = []
    for child in sorted(target.iterdir()):
        if child.is_file():
            files.append(_metadata_for(child, workspace_root))
    return files


def _metadata_for(target: Path, workspace_root: Path) -> FileMetadata:
    stat = target.stat()
    mime, _ = mimetypes.guess_type(target.name)
    return FileMetadata(
        path=str(target.relative_to(workspace_root.resolve())),
        size_bytes=stat.st_size,
        mime_type=mime,
        modified_at=stat.st_mtime,
    )
