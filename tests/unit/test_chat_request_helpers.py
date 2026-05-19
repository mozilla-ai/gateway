"""Unit tests for the tool-extraction helper in `gateway.api.routes.chat`.

The helper has to recognise both the gateway-native short form and the
provider-native shapes (OpenAI `code_interpreter`, Anthropic versioned
`code_execution_*`) so that pointing an OpenAI/Anthropic SDK at the gateway's
`base_url` keeps working unchanged.
"""

from __future__ import annotations

from gateway.api.routes.chat import _extract_code_execution_tool


def test_extracts_gateway_native_short_form() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_execution"}])
    assert entry == {"type": "code_execution"}
    assert remaining is None


def test_extracts_openai_code_interpreter_alias() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_interpreter"}])
    assert entry == {"type": "code_interpreter"}
    assert remaining is None


def test_extracts_anthropic_versioned_alias() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "code_execution_20250825"}])
    assert entry == {"type": "code_execution_20250825"}
    assert remaining is None


def test_extracts_future_anthropic_version_by_prefix() -> None:
    entry, _ = _extract_code_execution_tool([{"type": "code_execution_20991231"}])
    assert entry is not None


def test_passes_through_unrelated_tools() -> None:
    user_tool = {"type": "function", "function": {"name": "get_weather"}}
    entry, remaining = _extract_code_execution_tool([user_tool, {"type": "code_execution"}])
    assert entry == {"type": "code_execution"}
    assert remaining == [user_tool]


def test_takes_only_the_first_code_execution_entry() -> None:
    entry, remaining = _extract_code_execution_tool(
        [
            {"type": "code_execution", "purpose_hint": "first"},
            {"type": "code_interpreter"},
        ]
    )
    assert entry == {"type": "code_execution", "purpose_hint": "first"}
    assert remaining == [{"type": "code_interpreter"}]


def test_returns_no_entry_when_absent() -> None:
    entry, remaining = _extract_code_execution_tool([{"type": "function", "function": {"name": "f"}}])
    assert entry is None
    assert remaining == [{"type": "function", "function": {"name": "f"}}]


def test_empty_tools_returns_no_entry() -> None:
    entry, remaining = _extract_code_execution_tool(None)
    assert entry is None
    assert remaining is None


def test_does_not_match_unrelated_types_starting_with_code() -> None:
    entry, _ = _extract_code_execution_tool([{"type": "code_review"}])
    assert entry is None


def test_non_string_type_does_not_match() -> None:
    entry, _ = _extract_code_execution_tool([{"type": None}, {"type": 42}])
    assert entry is None
