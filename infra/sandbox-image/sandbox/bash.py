"""One-shot bash executor for the ``bash_code_execution`` tool.

Bash runs as a fresh subprocess per call (state lives in ``/workspace``,
which Python and bash share). The exec server enforces the per-call
timeout via ``asyncio.wait_for``; on timeout we send SIGKILL and return
whatever stdout was captured.
"""

from __future__ import annotations

import asyncio
import os

from sandbox.runner_pool import ExecOutcome


async def run_bash(command: str, *, timeout_seconds: int) -> ExecOutcome:
    """Run *command* in bash and return its captured output."""
    workspace = os.environ.get("SANDBOX_WORKSPACE", "/workspace")
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-c",
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except Exception:  # noqa: BLE001 - best effort
            stdout_bytes, stderr_bytes = b"", b""
        return ExecOutcome(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=(
                stderr_bytes.decode("utf-8", errors="replace")
                + f"bash: command timed out after {timeout_seconds}s\n"
            ),
            return_code=-1,
            timed_out=True,
        )

    return ExecOutcome(
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        return_code=proc.returncode if proc.returncode is not None else -1,
    )
