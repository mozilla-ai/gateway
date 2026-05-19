"""Tests for ``sandbox.limits`` and runner_pool integration with it.

Two layers:

* Pure-function tests for :func:`truncate_output` and the limit
  config dataclass — fast, deterministic, run on every platform.

* Subprocess-level tests for :class:`RunnerProcess` that exercise the
  truncation flow end-to-end (the runner actually emits a huge
  print, the pool clips it, the outcome carries the right flags).
  These run wherever the rest of the runner_pool tests run.

Notably absent: a real OOM test that crosses RLIMIT_AS. Triggering
the OOM-killer reliably from a unit test is fragile and very platform-
dependent (macOS rlimit semantics in particular are weak), so we
verify the *classification* of a signal exit deterministically via
:func:`_classify_signal_exit` instead, and rely on the kind/integration
test in 5A.1's smoke pass to confirm the rlimit actually fires on
real Linux.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from sandbox.limits import (
    LimitKind,
    ResourceLimits,
    TRUNCATION_MARKER,
    apply_runner_rlimits,
    default_limits,
    is_rlimit_supported,
    truncate_output,
)
from sandbox.runner_pool import (
    RunnerProcess,
    _classify_signal_exit,
)


# ---------------------------------------------------------------------------
# truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_short_output_unchanged(self) -> None:
        text, dropped = truncate_output("hello", max_bytes=100)
        assert text == "hello"
        assert dropped == 0

    def test_exact_boundary_unchanged(self) -> None:
        # 5 bytes under a 5-byte cap → no truncation.
        text, dropped = truncate_output("12345", max_bytes=5)
        assert text == "12345"
        assert dropped == 0

    def test_one_byte_over_truncates(self) -> None:
        text, dropped = truncate_output("123456", max_bytes=5)
        assert dropped == 1
        assert text.startswith("12345")
        assert "elided" in text
        # The marker says how many bytes were dropped.
        assert "1 bytes" in text

    def test_unicode_byte_counted_not_codepoint(self) -> None:
        # "é" is 2 bytes in UTF-8 but 1 code point.
        # 5 "é" = 10 bytes; cap at 6 bytes leaves 3 chars.
        text, dropped = truncate_output("ééééé", max_bytes=6)
        assert dropped == 4  # 10 - 6
        # The clipped portion may have a replacement char if a multi-
        # byte sequence got split, but 6 bytes = exactly 3 "é" chars,
        # so we should see 3 of them.
        assert text.startswith("ééé")

    def test_unicode_split_handled_with_replacement(self) -> None:
        # Cap at 5 bytes splits a 2-byte char in half — UTF-8 decode
        # with errors='replace' should produce a clean string.
        text, dropped = truncate_output("ééééé", max_bytes=5)
        assert dropped == 5  # 10 - 5
        # The truncation marker is appended after the (possibly
        # garbled) clipped portion.
        assert "elided" in text

    def test_marker_format(self) -> None:
        text, dropped = truncate_output("a" * 100, max_bytes=10)
        assert dropped == 90
        assert TRUNCATION_MARKER.format(dropped=90) in text


# ---------------------------------------------------------------------------
# default_limits / ResourceLimits
# ---------------------------------------------------------------------------


class TestResourceLimits:
    def test_dataclass_defaults_match_module_constants(self) -> None:
        limits = ResourceLimits()
        # Don't pin to specific values; just assert all the fields
        # exist with sensible non-zero defaults.
        assert limits.memory_mb > 0
        assert limits.cpu_seconds > 0
        assert limits.fsize_mb > 0
        assert limits.open_files > 0
        assert limits.max_output_bytes > 0
        assert limits.wall_clock_seconds > 0

    def test_default_limits_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SANDBOX_RUNNER_MEMORY_MB", "777")
        monkeypatch.setenv("SANDBOX_RUNNER_CPU_SECONDS", "33")
        monkeypatch.setenv("SANDBOX_RUNNER_MAX_OUTPUT_BYTES", "4096")
        limits = default_limits()
        assert limits.memory_mb == 777
        assert limits.cpu_seconds == 33
        assert limits.max_output_bytes == 4096


# ---------------------------------------------------------------------------
# apply_runner_rlimits
# ---------------------------------------------------------------------------


class TestApplyRunnerRlimits:
    def test_does_not_raise_on_call(self) -> None:
        # Always succeeds — even on macOS, the function swallows
        # individual setrlimit failures so a partial set is fine.
        apply_runner_rlimits(default_limits())

    def test_is_rlimit_supported_matches_platform(self) -> None:
        import sys
        assert is_rlimit_supported() == sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# _classify_signal_exit
# ---------------------------------------------------------------------------


class TestClassifySignalExit:
    def test_none_for_clean_exit(self) -> None:
        assert _classify_signal_exit(0) is None
        assert _classify_signal_exit(1) is None
        assert _classify_signal_exit(None) is None

    def test_sigkill_classified_as_memory(self) -> None:
        assert _classify_signal_exit(-signal.SIGKILL) == LimitKind.MEMORY

    def test_sigxcpu_classified_as_cpu(self) -> None:
        assert _classify_signal_exit(-signal.SIGXCPU) == LimitKind.CPU

    def test_sigxfsz_classified_as_file_size(self) -> None:
        assert _classify_signal_exit(-signal.SIGXFSZ) == LimitKind.FILE_SIZE

    def test_unknown_signal_returns_none(self) -> None:
        # SIGALRM is not a limit signal we map.
        assert _classify_signal_exit(-signal.SIGALRM) is None


# ---------------------------------------------------------------------------
# RunnerProcess truncation end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


async def test_runner_truncates_huge_stdout(workspace: Path) -> None:
    # Cap stdout at 256 bytes for this test so we don't have to print
    # a megabyte. Pass the limit explicitly so we don't depend on
    # env-var leakage between tests.
    limits = ResourceLimits(max_output_bytes=256)
    runner = RunnerProcess(workspace=str(workspace), limits=limits)
    try:
        await runner.start()
        outcome = await runner.execute(
            "print('A' * 5000)",
            timeout_seconds=10,
        )
    finally:
        await runner.stop()
    assert outcome.return_code == 0
    assert outcome.limit_exceeded == LimitKind.OUTPUT_BYTES
    assert outcome.stdout_dropped_bytes > 0
    assert "elided" in outcome.stdout
    # Output is capped near the limit (allow some overshoot for the
    # appended marker, which lives outside the clipped byte budget).
    assert len(outcome.stdout.encode("utf-8")) <= 256 + len(
        TRUNCATION_MARKER.format(dropped=outcome.stdout_dropped_bytes)
    )


async def test_runner_does_not_flag_truncation_under_limit(
    workspace: Path,
) -> None:
    runner = RunnerProcess(
        workspace=str(workspace),
        limits=ResourceLimits(max_output_bytes=4096),
    )
    try:
        await runner.start()
        outcome = await runner.execute("print('hello')", timeout_seconds=5)
    finally:
        await runner.stop()
    assert outcome.return_code == 0
    assert outcome.stdout == "hello"
    assert outcome.limit_exceeded is None
    assert outcome.stdout_dropped_bytes == 0
    assert outcome.stderr_dropped_bytes == 0


async def test_runner_timeout_tags_wall_clock_limit(workspace: Path) -> None:
    runner = RunnerProcess(workspace=str(workspace))
    try:
        await runner.start()
        outcome = await runner.execute(
            "import time; time.sleep(5)",
            timeout_seconds=1,
        )
    finally:
        await runner.stop()
    assert outcome.timed_out is True
    assert outcome.limit_exceeded == LimitKind.WALL_CLOCK
    assert outcome.return_code == -1
