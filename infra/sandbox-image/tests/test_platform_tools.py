"""Unit tests for sandbox.platform_tools (post-multi-session refactor).

``install_stubs`` now takes an explicit ``stub_dir`` parameter (each
session writes its stub into a different directory) so the tests no
longer monkey-patch a module-level constant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.platform_tools import (
    STUB_MODULE_NAME,
    ProgrammaticTool,
    StubConfig,
    generate_module_source,
    install_stubs,
)


def test_generates_callable_stub_for_each_tool() -> None:
    config = StubConfig(
        callback_url="http://router.local/tool-call",
        token="secret",
        tools=(
            ProgrammaticTool(name="gmail_list_emails", doc="List Gmail messages."),
            ProgrammaticTool(name="slack_post"),
        ),
    )
    source = generate_module_source(config)
    assert "def gmail_list_emails(**kwargs):" in source
    assert "def slack_post(**kwargs):" in source
    assert "List Gmail messages." in source
    assert "http://router.local/tool-call" in source
    assert "secret" in source


def test_generated_module_compiles() -> None:
    config = StubConfig(
        callback_url="http://router.local/tool-call",
        token="secret",
        tools=(ProgrammaticTool(name="echo"),),
    )
    source = generate_module_source(config)
    compile(source, "<test>", "exec")


def test_install_stubs_writes_into_specified_dir(tmp_path: Path) -> None:
    stub_dir = tmp_path / "lib"
    config = StubConfig(
        callback_url="http://x", token="t", tools=(ProgrammaticTool(name="foo"),)
    )
    path = install_stubs(config, stub_dir=stub_dir)
    assert path.exists()
    assert path.parent == stub_dir
    assert path.name == f"{STUB_MODULE_NAME}.py"
    assert "def foo(**kwargs):" in path.read_text()


def test_install_stubs_creates_parent_dir(tmp_path: Path) -> None:
    stub_dir = tmp_path / "deep" / "nested" / "lib"
    config = StubConfig(
        callback_url="http://x", token="t", tools=(ProgrammaticTool(name="foo"),)
    )
    path = install_stubs(config, stub_dir=stub_dir)
    assert path.exists()


def test_two_stub_dirs_are_independent(tmp_path: Path) -> None:
    """Each session's stub module is isolated to its own directory."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    install_stubs(
        StubConfig(
            callback_url="http://x",
            token="t-a",
            tools=(ProgrammaticTool(name="tool_a"),),
        ),
        stub_dir=dir_a,
    )
    install_stubs(
        StubConfig(
            callback_url="http://x",
            token="t-b",
            tools=(ProgrammaticTool(name="tool_b"),),
        ),
        stub_dir=dir_b,
    )
    assert "tool_a" in (dir_a / f"{STUB_MODULE_NAME}.py").read_text()
    assert "tool_b" in (dir_b / f"{STUB_MODULE_NAME}.py").read_text()
    assert "tool_a" not in (dir_b / f"{STUB_MODULE_NAME}.py").read_text()
    assert "tool_b" not in (dir_a / f"{STUB_MODULE_NAME}.py").read_text()


def test_rejects_invalid_tool_name() -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        generate_module_source(
            StubConfig(
                callback_url="http://x",
                token="t",
                tools=(ProgrammaticTool(name="123bad"),),
            )
        )


def test_rejects_keyword_tool_name() -> None:
    with pytest.raises(ValueError, match="shadows Python keyword"):
        generate_module_source(
            StubConfig(
                callback_url="http://x",
                token="t",
                tools=(ProgrammaticTool(name="class"),),
            )
        )


def test_rejects_duplicate_tool_name() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        generate_module_source(
            StubConfig(
                callback_url="http://x",
                token="t",
                tools=(
                    ProgrammaticTool(name="foo"),
                    ProgrammaticTool(name="foo"),
                ),
            )
        )


def test_rejects_empty_callback_url() -> None:
    with pytest.raises(ValueError, match="callback_url"):
        generate_module_source(
            StubConfig(callback_url="", token="t", tools=())
        )


def test_rejects_empty_token() -> None:
    with pytest.raises(ValueError, match="token"):
        generate_module_source(
            StubConfig(callback_url="http://x", token="", tools=())
        )
