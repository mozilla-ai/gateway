"""Subprocess lifecycle and protocol handling for the Python REPL runner.

The exec server holds a single :class:`RunnerProcess` per pod. It manages:

* Spawning the runner subprocess (``python -m sandbox.runner``).
* Sending code blocks using the length-prefixed protocol from
  :mod:`sandbox.runner`.
* Reading framed stdout/stderr/done sentinels back.
* Enforcing per-execution timeouts by killing and respawning the runner
  on deadline (with partial output captured up to that point).
* Restarting the runner if it crashes (segfault, OOM kill, internal
  protocol error) and surfacing the failure as a non-zero ``return_code``.

A single mutex serialises ``execute()`` calls — only one execution at a
time per runner — because the underlying Python REPL is single-threaded
and the protocol is request/response, not multiplexed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from sandbox.limits import (
    LimitKind,
    ResourceLimits,
    apply_runner_rlimits,
    default_limits,
    is_rlimit_supported,
    truncate_output,
)
from sandbox.runner import (
    SENTINEL_DONE,
    SENTINEL_STDERR,
    SENTINEL_STDERR_END,
    SENTINEL_STDOUT,
    SENTINEL_STDOUT_END,
)

# Resolve the runner script path once. We invoke it as a standalone script
# (rather than ``python -m sandbox.runner``) so the subprocess does not
# depend on the ``sandbox`` package being on its sys.path / pip-installed.
_RUNNER_SCRIPT = str(Path(__file__).resolve().parent / "runner.py")

# asyncio's default StreamReader limit is 64 KiB which is too small for the
# stdout frames we send back from the runner — a single ``print`` of a few
# hundred KB would overflow. Bump to 16 MiB; the runner already enforces a
# 200 KB code-input cap and per-call timeouts limit how much output any one
# execution can produce in practice.
_STREAM_BUFFER_LIMIT = 16 * 1024 * 1024

logger = logging.getLogger(__name__)


def _classify_signal_exit(exit_code: int | None) -> LimitKind | None:
    """Map a runner subprocess exit code to a :class:`LimitKind`.

    Subprocesses killed by a signal report ``-N`` where ``N`` is the
    signal number (Python's ``Popen`` convention, mirrored by
    ``asyncio.subprocess.Process.returncode``). The kernel sends
    SIGKILL (-9) for ``RLIMIT_AS`` violations on Linux and SIGXCPU
    (-24 on glibc, signal 24) for ``RLIMIT_CPU``. SIGXFSZ (-25)
    fires on ``RLIMIT_FSIZE``. We map each to a typed limit kind so
    the caller can render a friendly error.

    Returns ``None`` for non-signal exits and for signal numbers we
    don't recognise — the caller falls back to "process exited
    unexpectedly" which is the right thing to say for a real crash.
    """
    if exit_code is None or exit_code >= 0:
        return None
    sig = -exit_code
    if sig == signal.SIGKILL:
        # Most likely RLIMIT_AS or an OOM-killer hit. We can't
        # distinguish without /proc inspection so attribute to memory
        # — that's the cause in 95%+ of cases for our workload.
        return LimitKind.MEMORY
    if sig == signal.SIGXCPU:
        return LimitKind.CPU
    if sig == signal.SIGXFSZ:
        return LimitKind.FILE_SIZE
    return None


@dataclass(frozen=True)
class ExecOutcome:
    """Result of one ``RunnerProcess.execute`` call."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    # Set to a :class:`LimitKind` value when an in-pod resource cap
    # fired during this execution. ``None`` means the run completed
    # without hitting any limit. The gateway runner inspects this and
    # surfaces it as a structured SSE error so callers (Octonous) can
    # render a friendly message instead of guessing from a generic
    # nonzero return code.
    limit_exceeded: LimitKind | None = None
    # Number of bytes elided from stdout by the in-pod truncator (0
    # if no truncation happened). stderr truncation is reported via
    # ``stderr_dropped_bytes``. Both are surfaced for observability;
    # the model already sees the human-readable marker line in the
    # output itself.
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0


class RunnerProcess:
    """Owns a single long-lived ``sandbox.runner`` subprocess."""

    def __init__(
        self,
        *,
        workspace: str = "/workspace",
        extra_python_path: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> None:
        self._workspace = workspace
        self._extra_python_path = extra_python_path
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        # Resolved at construction time so each session sees a stable
        # snapshot — env-var changes mid-run shouldn't retroactively
        # alter a running session's caps.
        self._limits: ResourceLimits = limits or default_limits()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Spawn the runner subprocess if it isn't already running."""
        if self._proc is not None and self._proc.returncode is None:
            return
        env = os.environ.copy()
        env["SANDBOX_WORKSPACE"] = self._workspace
        # Make the per-session stub directory importable from inside this
        # session's REPL only (other sessions get a different dir).
        if self._extra_python_path:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self._extra_python_path}{os.pathsep}{existing}"
                if existing
                else self._extra_python_path
            )
        # Invoke the runner as a standalone script (not ``-m sandbox.runner``)
        # so it works without the ``sandbox`` package being importable in the
        # subprocess. The runner intentionally has no sandbox.* imports.
        #
        # ``preexec_fn`` runs in the child between fork and exec, before any
        # user Python has a chance to run, so the rlimits we set here are
        # already in force when the runner starts reading code blocks. Only
        # applied on Linux — macOS rlimit semantics are weak enough that
        # silently skipping is more useful than half-enforcing them. Real
        # production runs always go through the Linux path.
        preexec = (
            (lambda: apply_runner_rlimits(self._limits))
            if is_rlimit_supported()
            else None
        )
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _RUNNER_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._workspace,
            limit=_STREAM_BUFFER_LIMIT,
            preexec_fn=preexec,
        )
        logger.info(
            "runner started pid=%s memory_mb=%s cpu_seconds=%s rlimits=%s",
            self._proc.pid,
            self._limits.memory_mb,
            self._limits.cpu_seconds,
            "applied" if preexec is not None else "skipped",
        )

    async def stop(self) -> None:
        """Terminate the runner subprocess if running."""
        if self._proc is None:
            return
        if self._proc.returncode is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None

    async def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # ------------------------------------------------------------------ exec

    async def execute(self, code: str, *, timeout_seconds: int) -> ExecOutcome:
        """Send *code* to the runner and return its captured output.

        Serialised — concurrent ``execute`` calls are queued via the lock.
        On timeout, the runner is killed and respawned and partial stdout
        captured before the timeout is included in the result.
        """
        async with self._lock:
            await self.start()
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            payload = code.encode("utf-8")
            header = f"{len(payload)}\n".encode()
            try:
                self._proc.stdin.write(header)
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # Runner died before accepting input — restart and report.
                await self.stop()
                return ExecOutcome(
                    stdout="",
                    stderr="runner: pipe closed before send\n",
                    return_code=-1,
                )

            try:
                outcome = await asyncio.wait_for(
                    self._read_until_done(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Kill the runner so we don't read its output for the
                # next call. The respawn happens lazily on the next start().
                partial_stdout = self._drain_partial_stdout()
                await self.stop()
                truncated_partial, partial_dropped = truncate_output(
                    partial_stdout, max_bytes=self._limits.max_output_bytes
                )
                return ExecOutcome(
                    stdout=truncated_partial,
                    stderr=f"runner: execution timed out after {timeout_seconds}s\n",
                    return_code=-1,
                    timed_out=True,
                    limit_exceeded=LimitKind.WALL_CLOCK,
                    stdout_dropped_bytes=partial_dropped,
                )

            # Truncate per-stream so one bad ``print(huge_thing)``
            # can't blow up the SSE channel or stuff the LLM context
            # with megabytes of irrelevant bytes. We also tag the
            # outcome with ``OUTPUT_BYTES`` so the gateway can
            # surface a structured error if it wants to (currently
            # the model just sees the truncation marker in stdout).
            truncated_stdout, stdout_dropped = truncate_output(
                outcome.stdout, max_bytes=self._limits.max_output_bytes
            )
            truncated_stderr, stderr_dropped = truncate_output(
                outcome.stderr, max_bytes=self._limits.max_output_bytes
            )
            limit_kind = outcome.limit_exceeded
            if (
                limit_kind is None
                and (stdout_dropped > 0 or stderr_dropped > 0)
            ):
                limit_kind = LimitKind.OUTPUT_BYTES

            return ExecOutcome(
                stdout=truncated_stdout,
                stderr=truncated_stderr,
                return_code=outcome.return_code,
                timed_out=outcome.timed_out,
                limit_exceeded=limit_kind,
                stdout_dropped_bytes=stdout_dropped,
                stderr_dropped_bytes=stderr_dropped,
            )

    # ------------------------------------------------------------------ protocol

    async def _read_until_done(self) -> ExecOutcome:
        """Parse one runner response: stdout frame, stderr frame, done line."""
        assert self._proc is not None
        assert self._proc.stdout is not None
        stdout_text = ""
        stderr_text = ""
        in_section: str | None = None
        section_buf: list[str] = []
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                # EOF before sentinel — runner died mid-execution.
                # If we can grab the proc's exit code before stop(),
                # we can map common signals to a typed limit kind so
                # the caller can render a friendly error rather than
                # a generic crash.
                exit_code = self._proc.returncode
                if exit_code is None:
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                    exit_code = self._proc.returncode
                limit_kind = _classify_signal_exit(exit_code)
                await self.stop()
                if limit_kind is not None:
                    stderr_suffix = (
                        f"runner: killed by sandbox limit ({limit_kind.value})\n"
                    )
                else:
                    stderr_suffix = "runner: process exited unexpectedly\n"
                return ExecOutcome(
                    stdout=stdout_text,
                    stderr=stderr_text + stderr_suffix,
                    return_code=-1,
                    limit_exceeded=limit_kind,
                )
            line = raw.decode("utf-8", errors="replace")
            stripped = line.rstrip("\n")

            if stripped == SENTINEL_STDOUT:
                in_section = "stdout"
                section_buf = []
                continue
            if stripped == SENTINEL_STDOUT_END:
                stdout_text = "".join(section_buf)
                # Strip the trailing newline the runner adds for non-newline-
                # ending stdout so we faithfully echo what the user printed.
                if stdout_text.endswith("\n"):
                    stdout_text = stdout_text[:-1]
                in_section = None
                continue
            if stripped == SENTINEL_STDERR:
                in_section = "stderr"
                section_buf = []
                continue
            if stripped == SENTINEL_STDERR_END:
                stderr_text = "".join(section_buf)
                if stderr_text.endswith("\n"):
                    stderr_text = stderr_text[:-1]
                in_section = None
                continue
            if stripped.startswith(SENTINEL_DONE + " "):
                try:
                    rc = int(stripped[len(SENTINEL_DONE) + 1 :])
                except ValueError:
                    rc = -1
                return ExecOutcome(
                    stdout=stdout_text,
                    stderr=stderr_text,
                    return_code=rc,
                )

            if in_section is not None:
                section_buf.append(line)
            # Lines outside any section are silently dropped (shouldn't
            # happen with a well-behaved runner; protocol-noise tolerance).

    def _drain_partial_stdout(self) -> str:
        """Best-effort grab of buffered stdout after a timeout.

        Used to surface partial print output captured before the runner was
        killed. Returns ``""`` if no buffered data is available.
        """
        if (
            self._proc is None
            or self._proc.stdout is None
            or self._proc.returncode is not None
        ):
            return ""
        try:
            buffered = self._proc.stdout._buffer  # type: ignore[attr-defined]
        except AttributeError:
            return ""
        try:
            return bytes(buffered).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best effort
            return ""
