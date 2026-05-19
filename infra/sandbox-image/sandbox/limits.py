"""Per-session resource limits for the in-pod sandbox runner.

These exist *in addition to* the K8s pod-level ``resources.limits``
the sandbox manifest declares: pod limits cap the whole container,
which is too coarse for a multi-session pod where one runaway script
shouldn't OOM its neighbours. The in-pod limits below are tighter,
fire first, and produce a clean structured error instead of a
SIGKILL the user can't tell from a real crash.

Three categories of limits land here:

1. **OS rlimits** applied to the runner subprocess at fork time via
   :func:`apply_runner_rlimits`. ``RLIMIT_AS`` caps virtual memory,
   ``RLIMIT_CPU`` caps CPU seconds, ``RLIMIT_FSIZE`` caps any file
   the runner writes, ``RLIMIT_NOFILE`` caps the file-descriptor
   table. These are *hard* caps — the kernel kills the process when
   it crosses one, and the runner_pool surfaces the SIGKILL as a
   ``LIMIT_EXCEEDED`` outcome.

2. **Output truncation** done by the runner_pool *after* a successful
   execution: stdout/stderr are clipped at
   :data:`MAX_OUTPUT_BYTES` per stream and a clear marker line is
   appended so the model can see what happened. This protects the
   gateway from one bad ``print(huge_thing)`` blowing up the SSE
   stream and the LLM context.

3. **Wall-clock timeout** is unchanged from before — already enforced
   in :class:`sandbox.runner_pool.RunnerProcess.execute` via
   ``asyncio.wait_for``. We re-export the default here for
   discoverability.

All four categories are tunable per-container via env vars on the
sandbox manifest, so the K8s overlay can dial them per-environment
without code changes.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable via env vars on the sandbox container.
# ---------------------------------------------------------------------------


DEFAULT_MEMORY_MB = int(os.environ.get("SANDBOX_RUNNER_MEMORY_MB", "1536"))
"""Per-runner address-space cap. The K8s pod has a 2 GiB memory limit;
this leaves headroom for the exec server itself plus the FastAPI
worker. Tune via ``SANDBOX_RUNNER_MEMORY_MB`` env var if needed."""

DEFAULT_CPU_SECONDS = int(os.environ.get("SANDBOX_RUNNER_CPU_SECONDS", "120"))
"""Hard CPU-second cap. Different from wall-clock — the runner can use
up to this many CPU-seconds across the lifetime of one execution
before the kernel sends SIGXCPU."""

DEFAULT_FSIZE_MB = int(os.environ.get("SANDBOX_RUNNER_FSIZE_MB", "256"))
"""Hard cap on the size of any single file the runner writes."""

DEFAULT_OPEN_FILES = int(os.environ.get("SANDBOX_RUNNER_NOFILE", "256"))
"""Hard cap on the runner's open file descriptors. 256 is plenty for
real Python work and tight enough that a leak shows up fast."""

DEFAULT_MAX_OUTPUT_BYTES = int(
    os.environ.get("SANDBOX_RUNNER_MAX_OUTPUT_BYTES", str(1 * 1024 * 1024))  # 1 MiB
)
"""Per-stream output cap (stdout and stderr each). One bad
``print(open(huge_file).read())`` blows up the SSE stream and pumps
the LLM full of useless context bytes; we'd rather truncate."""

DEFAULT_WALL_CLOCK_SECONDS = int(
    os.environ.get("SANDBOX_RUNNER_WALL_CLOCK_SECONDS", "60")
)
"""Wall-clock cap per execution. Enforced in
:class:`sandbox.runner_pool.RunnerProcess.execute`, re-exported here
for discoverability."""


# Sentinel marker we append after truncated output so the model can
# see the elision happened. Picked to be visible but not parseable as
# Python so the model is unlikely to mistake it for real output.
TRUNCATION_MARKER = "\n... [output truncated by sandbox: {dropped} bytes elided]"


class LimitKind(StrEnum):
    """Which resource limit fired, if any."""

    MEMORY = "memory"
    CPU = "cpu"
    FILE_SIZE = "file_size"
    OUTPUT_BYTES = "output_bytes"
    WALL_CLOCK = "wall_clock"


# ---------------------------------------------------------------------------
# Limit config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceLimits:
    """All limits applied to one runner subprocess.

    Defaults come from env vars on the sandbox container so the K8s
    manifest can override them without code changes.
    """

    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_seconds: int = DEFAULT_CPU_SECONDS
    fsize_mb: int = DEFAULT_FSIZE_MB
    open_files: int = DEFAULT_OPEN_FILES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS


def default_limits() -> ResourceLimits:
    """Return a fresh :class:`ResourceLimits` from current env vars.

    Resolved at call time (not import time) so tests can patch the
    env between cases.
    """
    return ResourceLimits(
        memory_mb=int(os.environ.get("SANDBOX_RUNNER_MEMORY_MB", str(DEFAULT_MEMORY_MB))),
        cpu_seconds=int(
            os.environ.get("SANDBOX_RUNNER_CPU_SECONDS", str(DEFAULT_CPU_SECONDS))
        ),
        fsize_mb=int(
            os.environ.get("SANDBOX_RUNNER_FSIZE_MB", str(DEFAULT_FSIZE_MB))
        ),
        open_files=int(
            os.environ.get("SANDBOX_RUNNER_NOFILE", str(DEFAULT_OPEN_FILES))
        ),
        max_output_bytes=int(
            os.environ.get(
                "SANDBOX_RUNNER_MAX_OUTPUT_BYTES", str(DEFAULT_MAX_OUTPUT_BYTES)
            )
        ),
        wall_clock_seconds=int(
            os.environ.get(
                "SANDBOX_RUNNER_WALL_CLOCK_SECONDS",
                str(DEFAULT_WALL_CLOCK_SECONDS),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Subprocess rlimit application
# ---------------------------------------------------------------------------


def apply_runner_rlimits(limits: ResourceLimits) -> None:
    """Apply *limits* to the current process via :mod:`resource`.

    Designed to be passed as the ``preexec_fn`` of an
    :func:`asyncio.create_subprocess_exec` call so the limits are
    inherited by the runner subprocess at fork time, before any user
    code has a chance to run. The kernel enforces them, not us, so a
    runaway script can't override them via ``import resource;
    resource.setrlimit(...)`` — that call would also be subject to
    the (lower) hard cap.

    No-op on platforms where :mod:`resource` is not available
    (Windows) or where ``setrlimit`` semantics are too different to
    be reliable (macOS in some setups). The runner pool also has a
    Linux check before passing this as a preexec_fn so the local
    dev experience on macOS is unaffected.

    The defensive try/except wraps each rlimit individually because
    some hardened sandboxes ship with one or two of these already
    capped lower than our request — we want the others to still
    apply rather than aborting early on the first failure.
    """
    if sys.platform == "win32":
        return
    try:
        import resource
    except ImportError:
        return

    def _set(name: int, soft: int, hard: int | None = None) -> None:
        try:
            resource.setrlimit(name, (soft, hard if hard is not None else soft))
        except (ValueError, OSError) as exc:  # pragma: no cover - platform-specific
            logger.warning("apply_runner_rlimits: %s setrlimit failed: %s", name, exc)

    # RLIMIT_AS — virtual address space cap. Bytes.
    _set(resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024)
    # RLIMIT_CPU — CPU seconds. Soft cap delivers SIGXCPU; hard cap is SIGKILL.
    # We set them to the same value so the runner gets exactly one signal.
    _set(resource.RLIMIT_CPU, limits.cpu_seconds)
    # RLIMIT_FSIZE — max bytes for any single file write.
    _set(resource.RLIMIT_FSIZE, limits.fsize_mb * 1024 * 1024)
    # RLIMIT_NOFILE — max open file descriptors.
    _set(resource.RLIMIT_NOFILE, limits.open_files)


def is_rlimit_supported() -> bool:
    """Return ``True`` if we should apply rlimits on this host.

    The sandbox container always runs Linux, but local dev /
    integration tests may run on macOS where setrlimit semantics are
    weak (e.g. ``RLIMIT_AS`` is silently ignored). On macOS we
    deliberately skip applying limits so the test suite is
    deterministic — production runs always go through the Linux path.
    """
    return sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def truncate_output(
    text: str, *, max_bytes: int
) -> tuple[str, int]:
    """Clip *text* to ``max_bytes`` bytes, appending a marker if dropped.

    Returns ``(clipped_text, dropped_bytes)``. ``dropped_bytes`` is 0
    if no truncation happened, otherwise the number of bytes that
    were elided. Operates on UTF-8 byte length so the limit is the
    real wire length, not the str length (which counts code points
    and would be off for any non-ASCII content).

    The marker is appended *after* the clipped portion, in human-
    readable English, so the model can see explicitly that bytes were
    dropped and how many.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, 0

    dropped = len(encoded) - max_bytes
    # Slice on bytes, then decode with errors='replace' so we don't
    # split a multi-byte sequence in half and produce garbled output.
    clipped_bytes = encoded[:max_bytes]
    clipped_text = clipped_bytes.decode("utf-8", errors="replace")
    return clipped_text + TRUNCATION_MARKER.format(dropped=dropped), dropped
