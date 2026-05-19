"""Generator for the in-pod ``_platform_tools`` stub module.

When the platform creates a sandbox, it POSTs the list of "programmatic
tools" the agent has declared (e.g. ``gmail_list_emails``, ``slack_post``)
along with a callback URL and a per-sandbox shared secret. The exec server
calls :func:`install_stubs` which writes a Python module to a directory
on ``sys.path`` (``/opt/sandbox/lib``) so user code can simply do::

    from _platform_tools import gmail_list_emails
    emails = gmail_list_emails(query="from:boss")

Each generated stub blocks on a synchronous HTTP POST back to the platform's
in-cluster programmatic tool router. The router pushes the tool call over
the active WebSocket to the caller, awaits the reply, and responds. The
``urllib`` stdlib client is used (no third-party dependency) so the user-
facing module stays minimal.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path

STUB_MODULE_NAME = "_platform_tools"

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Default per-call timeout — generous enough for human-in-the-loop tools.
DEFAULT_CALL_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class ProgrammaticTool:
    """Description of one tool the agent has declared callable from Python."""

    name: str
    doc: str = ""


@dataclass(frozen=True)
class StubConfig:
    """Configuration the platform supplies when registering tools."""

    callback_url: str
    token: str
    tools: tuple[ProgrammaticTool, ...]
    timeout_seconds: int = DEFAULT_CALL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_module_source(config: StubConfig) -> str:
    """Return the Python source for the ``_platform_tools`` stub module.

    The generated source is deliberately small and stdlib-only so it
    imports cheaply inside user code. Each tool becomes a function that
    accepts ``**kwargs`` and forwards them as the input payload.
    """
    if not config.callback_url:
        raise ValueError("callback_url must not be empty")
    if not config.token:
        raise ValueError("token must not be empty")
    _validate_tool_names(config.tools)

    callback = json.dumps(config.callback_url)
    token = json.dumps(config.token)
    timeout = int(config.timeout_seconds)
    tool_defs = "\n\n".join(_render_tool(t) for t in config.tools)

    return _MODULE_TEMPLATE.format(
        callback=callback,
        token=token,
        timeout=timeout,
        tool_defs=tool_defs or "# No tools registered.",
    )


def install_stubs(config: StubConfig, *, stub_dir: Path) -> Path:
    """Write the generated module under *stub_dir* and return its path.

    The caller is responsible for ensuring *stub_dir* is on the runner
    subprocess's ``PYTHONPATH``. In the multi-session world this is the
    per-session ``lib`` directory created by :class:`SessionManager`.
    """
    stub_dir.mkdir(parents=True, exist_ok=True)
    source = generate_module_source(config)
    target = stub_dir / f"{STUB_MODULE_NAME}.py"
    target.write_text(source, encoding="utf-8")
    return target


def clear_stubs(stub_dir: Path) -> None:
    """Remove the stub module under *stub_dir* if it exists (test helper)."""
    target = stub_dir / f"{STUB_MODULE_NAME}.py"
    try:
        target.unlink()
    except FileNotFoundError:
        pass


def _validate_tool_names(tools: tuple[ProgrammaticTool, ...]) -> None:
    seen: set[str] = set()
    for tool in tools:
        if not _VALID_NAME.match(tool.name):
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if keyword.iskeyword(tool.name):
            raise ValueError(f"tool name shadows Python keyword: {tool.name!r}")
        if tool.name in seen:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        seen.add(tool.name)


def _render_tool(tool: ProgrammaticTool) -> str:
    doc_literal = json.dumps(tool.doc or f"Programmatic tool {tool.name}.")
    return _TOOL_TEMPLATE.format(name=tool.name, doc_literal=doc_literal)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_MODULE_TEMPLATE = '''"""Auto-generated programmatic tool stubs. Do not edit by hand.

Each function here is a thin RPC stub that POSTs back to the platform's
programmatic tool router. The router pushes the call to the agent over
the active WebSocket and replies with the tool result, which becomes the
return value of the stub. Errors raise :class:`ToolError`.
"""

import json
import urllib.error
import urllib.request
import uuid

_CALLBACK_URL = {callback}
_TOKEN = {token}
_TIMEOUT = {timeout}


class ToolError(Exception):
    """Raised when a programmatic tool call fails."""


def _call(name, payload):
    body = json.dumps({{
        "call_id": uuid.uuid4().hex,
        "name": name,
        "input": payload,
    }}).encode("utf-8")
    request = urllib.request.Request(
        _CALLBACK_URL,
        data=body,
        headers={{
            "Content-Type": "application/json",
            "Authorization": "Bearer " + _TOKEN,
        }},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ToolError("tool {{!r}} failed: HTTP {{}}".format(name, exc.code)) from exc
    except urllib.error.URLError as exc:
        raise ToolError("tool {{!r}} unreachable: {{}}".format(name, exc.reason)) from exc
    except OSError as exc:
        raise ToolError("tool {{!r}} transport error: {{}}".format(name, exc)) from exc
    if data.get("is_error"):
        raise ToolError(data.get("content", "unknown tool error"))
    return data.get("content")


{tool_defs}
'''

_TOOL_TEMPLATE = '''def {name}(**kwargs):
    {doc_literal}
    return _call("{name}", kwargs)'''
