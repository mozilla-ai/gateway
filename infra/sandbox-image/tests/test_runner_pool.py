"""Unit tests for sandbox.runner_pool — the subprocess + protocol layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.runner_pool import RunnerProcess


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
async def runner(workspace: Path):
    proc = RunnerProcess(workspace=str(workspace))
    await proc.start()
    yield proc
    await proc.stop()


async def test_simple_print(runner: RunnerProcess) -> None:
    outcome = await runner.execute("print('hello')", timeout_seconds=5)
    assert outcome.return_code == 0
    assert outcome.stdout == "hello"
    assert outcome.stderr == ""
    assert outcome.timed_out is False


async def test_state_persists_across_calls(runner: RunnerProcess) -> None:
    first = await runner.execute("x = 41", timeout_seconds=5)
    assert first.return_code == 0
    second = await runner.execute("print(x + 1)", timeout_seconds=5)
    assert second.return_code == 0
    assert second.stdout == "42"


async def test_imports_persist(runner: RunnerProcess) -> None:
    first = await runner.execute("import math", timeout_seconds=5)
    assert first.return_code == 0
    second = await runner.execute("print(math.pi)", timeout_seconds=5)
    assert second.return_code == 0
    assert "3.14" in second.stdout


async def test_stderr_captured_separately(runner: RunnerProcess) -> None:
    outcome = await runner.execute(
        "import sys; print('out'); print('err', file=sys.stderr)",
        timeout_seconds=5,
    )
    assert outcome.return_code == 0
    assert outcome.stdout == "out"
    assert outcome.stderr == "err"


async def test_user_exception_returns_nonzero(runner: RunnerProcess) -> None:
    outcome = await runner.execute("raise ValueError('boom')", timeout_seconds=5)
    assert outcome.return_code == 1
    assert "ValueError" in outcome.stderr
    assert "boom" in outcome.stderr
    assert outcome.stdout == ""


async def test_state_survives_after_exception(runner: RunnerProcess) -> None:
    await runner.execute("x = 1", timeout_seconds=5)
    await runner.execute("raise RuntimeError('oops')", timeout_seconds=5)
    after = await runner.execute("print(x)", timeout_seconds=5)
    assert after.return_code == 0
    assert after.stdout == "1"


async def test_systemexit_zero(runner: RunnerProcess) -> None:
    outcome = await runner.execute("import sys; sys.exit(0)", timeout_seconds=5)
    assert outcome.return_code == 0


async def test_systemexit_nonzero(runner: RunnerProcess) -> None:
    outcome = await runner.execute("import sys; sys.exit(7)", timeout_seconds=5)
    assert outcome.return_code == 7


async def test_workspace_is_cwd(runner: RunnerProcess, workspace: Path) -> None:
    outcome = await runner.execute(
        "import os; print(os.getcwd())", timeout_seconds=5
    )
    assert outcome.return_code == 0
    # Resolve both sides to handle macOS /private symlink prefix.
    assert Path(outcome.stdout).resolve() == workspace.resolve()


async def test_files_persist_in_workspace(
    runner: RunnerProcess, workspace: Path
) -> None:
    write = await runner.execute(
        "open('hello.txt', 'w').write('world')", timeout_seconds=5
    )
    assert write.return_code == 0
    assert (workspace / "hello.txt").read_text() == "world"
    read = await runner.execute(
        "print(open('hello.txt').read())", timeout_seconds=5
    )
    assert read.stdout == "world"


async def test_timeout_returns_partial_stdout(runner: RunnerProcess) -> None:
    code = (
        "import time, sys\n"
        "for i in range(5):\n"
        "    print(i, flush=True)\n"
        "    time.sleep(0.5)\n"
    )
    outcome = await runner.execute(code, timeout_seconds=1)
    assert outcome.timed_out is True
    assert outcome.return_code == -1
    assert "timed out" in outcome.stderr


async def test_runner_recovers_after_timeout(workspace: Path) -> None:
    proc = RunnerProcess(workspace=str(workspace))
    await proc.start()
    try:
        bad = await proc.execute(
            "import time; time.sleep(10)", timeout_seconds=1
        )
        assert bad.timed_out is True
        good = await proc.execute("print('still alive')", timeout_seconds=5)
        assert good.return_code == 0
        assert good.stdout == "still alive"
    finally:
        await proc.stop()


async def test_large_stdout(runner: RunnerProcess) -> None:
    outcome = await runner.execute(
        "print('x' * 100_000)", timeout_seconds=10
    )
    assert outcome.return_code == 0
    assert len(outcome.stdout) == 100_000


async def test_multiline_output_preserved(runner: RunnerProcess) -> None:
    outcome = await runner.execute(
        "print('a'); print('b'); print('c')", timeout_seconds=5
    )
    assert outcome.return_code == 0
    assert outcome.stdout == "a\nb\nc"


async def test_extra_python_path_is_importable(tmp_path: Path) -> None:
    """Modules under extra_python_path are importable from inside the REPL."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "_my_test_module.py").write_text("VALUE = 1234\n")

    proc = RunnerProcess(workspace=str(workspace), extra_python_path=str(stub_dir))
    await proc.start()
    try:
        outcome = await proc.execute(
            "import _my_test_module; print(_my_test_module.VALUE)",
            timeout_seconds=5,
        )
        assert outcome.return_code == 0
        assert outcome.stdout == "1234"
    finally:
        await proc.stop()
