"""Long-lived Python REPL with a length-prefixed sentinel protocol.

The exec server spawns this as a subprocess and feeds it code blocks over
stdin. Each block runs inside the same module ``__main__`` namespace, so
variables, imports and ``/workspace`` filesystem state persist across calls.

Wire protocol (one frame per execution):

1. Client writes the header line: ``<byte_length>\\n``
2. Client writes ``byte_length`` bytes of UTF-8 Python source.
3. Runner executes the source with ``exec(compile(...), globals)``.
4. Runner writes captured stdout, then a stderr-tagged section, then a
   trailing sentinel line: ``---SBX_DONE--- <return_code>\\n``.

Stdout and stderr are interleaved within a single execution (as the user
code emits them) but they are *captured* into separate buffers and emitted
on separate frames so the exec server can return them in distinct fields
of the result block.

Frame separators on the wire:

* ``---SBX_STDOUT---\\n<stdout bytes>\\n---SBX_STDOUT_END---\\n``
* ``---SBX_STDERR---\\n<stderr bytes>\\n---SBX_STDERR_END---\\n``
* ``---SBX_DONE--- <return_code>\\n``

Errors during user code are captured to the stderr buffer and the return
code is set to 1; errors in the runner protocol itself terminate the
process so the exec server can respawn it.

The runner reads/writes the file descriptors directly via ``os.read`` and
``os.write`` to avoid Python text-mode buffering — the parent exec server
needs each frame to land atomically.
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

SENTINEL_DONE = "---SBX_DONE---"
SENTINEL_STDOUT = "---SBX_STDOUT---"
SENTINEL_STDOUT_END = "---SBX_STDOUT_END---"
SENTINEL_STDERR = "---SBX_STDERR---"
SENTINEL_STDERR_END = "---SBX_STDERR_END---"

_STDIN_FD = 0
_STDOUT_FD = 1
_STDERR_FD = 2


def _write_out(text: str) -> None:
    os.write(_STDOUT_FD, text.encode("utf-8"))


def _write_err(text: str) -> None:
    os.write(_STDERR_FD, text.encode("utf-8"))


def _read_line() -> str:
    """Read a single line (terminated by ``\\n``) from stdin via os.read."""
    chars: list[bytes] = []
    while True:
        b = os.read(_STDIN_FD, 1)
        if not b:
            return "".join(c.decode("utf-8", errors="replace") for c in chars)
        if b == b"\n":
            return "".join(c.decode("utf-8", errors="replace") for c in chars)
        chars.append(b)


def _read_exact(n: int) -> bytes:
    """Read exactly *n* bytes from stdin or fewer on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = os.read(_STDIN_FD, n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _write_frame(tag: str, end_tag: str, data: str) -> None:
    """Write a tagged frame to stdout."""
    _write_out(f"{tag}\n")
    _write_out(data)
    if not data.endswith("\n"):
        _write_out("\n")
    _write_out(f"{end_tag}\n")


def _write_done(return_code: int) -> None:
    _write_out(f"{SENTINEL_DONE} {return_code}\n")


def run() -> int:
    """Read code blocks from stdin in a loop and execute them.

    Returns the process exit code (0 on clean EOF).
    """
    # Persistent globals — variables, imports, classes survive across blocks.
    user_globals: dict[str, object] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "__doc__": None,
    }

    # All user code starts in /workspace so relative file paths are stable.
    workspace = os.environ.get("SANDBOX_WORKSPACE", "/workspace")
    os.makedirs(workspace, exist_ok=True)
    try:
        os.chdir(workspace)
    except OSError:
        # Non-fatal: chdir failure just means relative paths resolve oddly.
        pass

    while True:
        header = _read_line()
        if not header:
            return 0  # clean EOF
        header = header.strip()
        if not header:
            continue
        try:
            byte_length = int(header)
        except ValueError:
            # Protocol error — terminate so the exec server respawns us.
            _write_err(f"runner: bad header {header!r}\n")
            return 2

        code_bytes = _read_exact(byte_length)
        if len(code_bytes) != byte_length:
            _write_err(f"runner: short read ({len(code_bytes)}/{byte_length})\n")
            return 2
        code = code_bytes.decode("utf-8", errors="replace")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        return_code = 0

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, user_globals)
        except SystemExit as exc:
            # User code called sys.exit(...) — propagate the exit code but
            # don't actually terminate the runner.
            try:
                return_code = int(exc.code) if exc.code is not None else 0
            except (TypeError, ValueError):
                return_code = 1
        except BaseException:  # noqa: BLE001 - capture absolutely everything
            # Includes user-raised exceptions and KeyboardInterrupt.
            traceback.print_exc(file=stderr_buf)
            return_code = 1

        _write_frame(SENTINEL_STDOUT, SENTINEL_STDOUT_END, stdout_buf.getvalue())
        _write_frame(SENTINEL_STDERR, SENTINEL_STDERR_END, stderr_buf.getvalue())
        _write_done(return_code)


if __name__ == "__main__":
    sys.exit(run())
