"""HTTP-level tests for sandbox.exec_server.

The server is multi-session post-Phase-2: every endpoint that touches
session state lives under ``/sessions/{id}/...`` and tests open a session
explicitly via POST /sessions before making any tool calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(sessions_root: Path) -> Iterator[TestClient]:
    """Yield a TestClient with a fresh SessionManager pointed at the tmp root.

    We replace ``exec_server._session_manager`` with a freshly-constructed
    instance whose sessions dir is the test's tmp directory. We do *not*
    reload the ``sandbox.sessions`` module: doing so would create a second
    ``SessionNotFoundError`` class object and break ``pytest.raises``
    matching in tests that imported the original.
    """
    import sandbox.exec_server as es
    from sandbox.sessions import SessionManager

    es._session_manager = SessionManager(sessions_root=sessions_root)  # noqa: SLF001
    with TestClient(es.app) as c:
        yield c


def _create_session(client: TestClient) -> str:
    response = client.post("/sessions", json={})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


# ---------------------------------------------------------------------------
# Health & sessions
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["session_count"] == 0


def test_create_session_returns_metadata(client: TestClient) -> None:
    response = client.post("/sessions", json={})
    assert response.status_code == 201
    body = response.json()
    assert body["session_id"].startswith("sbx_")
    assert body["created_at"] > 0
    assert body["idle_timeout_seconds"] > 0


def test_get_session_metadata(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.get(f"/sessions/{sid}")
    assert response.status_code == 200
    assert response.json()["session_id"] == sid


def test_get_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/sessions/sbx_nope")
    assert response.status_code == 404


def test_destroy_session_then_404(client: TestClient) -> None:
    sid = _create_session(client)
    deleted = client.delete(f"/sessions/{sid}")
    assert deleted.status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_health_reports_active_session_count(client: TestClient) -> None:
    _create_session(client)
    _create_session(client)
    body = client.get("/health").json()
    assert body["session_count"] == 2


# ---------------------------------------------------------------------------
# Exec
# ---------------------------------------------------------------------------


def test_exec_code_execution_returns_anthropic_shape(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "code_execution",
            "input": {"code": "print(2 + 2)"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tool_use_id"].startswith("srvtoolu_")
    block = body["result_block"]
    assert block["type"] == "code_execution_tool_result"
    assert block["tool_use_id"] == body["tool_use_id"]
    content = block["content"]
    assert content["type"] == "code_execution_result"
    assert content["stdout"] == "4"
    assert content["return_code"] == 0
    assert content["content"] == []
    assert body["execution_time_ms"] >= 0


def test_exec_state_persistence_across_calls_same_session(
    client: TestClient,
) -> None:
    sid = _create_session(client)
    first = client.post(
        f"/sessions/{sid}/exec",
        json={"tool": "code_execution", "input": {"code": "x = 5"}},
    )
    assert first.status_code == 200
    second = client.post(
        f"/sessions/{sid}/exec",
        json={"tool": "code_execution", "input": {"code": "print(x * 2)"}},
    )
    assert second.status_code == 200
    assert second.json()["result_block"]["content"]["stdout"] == "10"


def test_exec_state_isolation_across_sessions(client: TestClient) -> None:
    """Variables from one session must not be visible in another."""
    sid_a = _create_session(client)
    sid_b = _create_session(client)

    client.post(
        f"/sessions/{sid_a}/exec",
        json={"tool": "code_execution", "input": {"code": "marker = 'A'"}},
    )
    response = client.post(
        f"/sessions/{sid_b}/exec",
        json={
            "tool": "code_execution",
            "input": {"code": "print('marker' in dir())"},
        },
    )
    assert response.status_code == 200
    assert response.json()["result_block"]["content"]["stdout"] == "False"


def test_exec_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/sessions/sbx_does_not_exist/exec",
        json={"tool": "code_execution", "input": {"code": "print(1)"}},
    )
    assert response.status_code == 404


def test_exec_bash(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/exec",
        json={"tool": "bash_code_execution", "input": {"command": "echo hello"}},
    )
    assert response.status_code == 200
    block = response.json()["result_block"]
    assert block["type"] == "bash_code_execution_tool_result"
    assert block["content"]["stdout"] == "hello\n"
    assert block["content"]["return_code"] == 0


def test_exec_text_editor_create(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "text_editor_code_execution",
            "input": {
                "command": "create",
                "path": "draft.md",
                "file_text": "# Draft\n",
            },
        },
    )
    assert response.status_code == 200
    block = response.json()["result_block"]
    assert block["type"] == "text_editor_code_execution_tool_result"
    assert block["content"]["return_code"] == 0


def test_exec_text_editor_view_after_create(client: TestClient) -> None:
    sid = _create_session(client)
    client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "text_editor_code_execution",
            "input": {
                "command": "create",
                "path": "x.py",
                "file_text": "hello\nworld\n",
            },
        },
    )
    response = client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "text_editor_code_execution",
            "input": {"command": "view", "path": "x.py"},
        },
    )
    assert response.status_code == 200
    assert "hello" in response.json()["result_block"]["content"]["stdout"]


def test_exec_invalid_input_returns_422(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/exec",
        json={"tool": "code_execution", "input": {}},
    )
    assert response.status_code == 422


def test_exec_uses_provided_tool_use_id(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "code_execution",
            "input": {"code": "print('ok')"},
            "tool_use_id": "srvtoolu_test_123",
        },
    )
    assert response.status_code == 200
    assert response.json()["tool_use_id"] == "srvtoolu_test_123"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_http_upload_and_download_roundtrip(client: TestClient) -> None:
    sid = _create_session(client)
    upload = client.post(
        f"/sessions/{sid}/files",
        files={"file": ("data.csv", BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert upload.status_code == 201
    body = upload.json()
    assert body["path"] == "data.csv"
    assert body["size_bytes"] == 8

    download = client.get(f"/sessions/{sid}/files", params={"path": "data.csv"})
    assert download.status_code == 200
    assert download.content == b"a,b\n1,2\n"
    assert download.headers["content-type"].startswith("text/csv")


def test_http_upload_explicit_path_override(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/files",
        files={"file": ("orig.bin", BytesIO(b"hi"), "application/octet-stream")},
        data={"path": "renamed/inside.bin"},
    )
    assert response.status_code == 201
    assert response.json()["path"] == "renamed/inside.bin"


def test_http_upload_path_escape_blocked(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/files",
        files={"file": ("x.txt", BytesIO(b"x"), "text/plain")},
        data={"path": "/etc/passwd"},
    )
    assert response.status_code == 403


def test_http_download_missing_returns_404(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.get(f"/sessions/{sid}/files", params={"path": "nope.txt"})
    assert response.status_code == 404


def test_http_list(client: TestClient) -> None:
    sid = _create_session(client)
    client.post(
        f"/sessions/{sid}/files",
        files={"file": ("one.txt", BytesIO(b"1"), "text/plain")},
    )
    client.post(
        f"/sessions/{sid}/files",
        files={"file": ("two.txt", BytesIO(b"22"), "text/plain")},
    )
    response = client.get(f"/sessions/{sid}/files/list", params={"path": "."})
    assert response.status_code == 200
    files = response.json()["files"]
    paths = sorted(f["path"] for f in files)
    assert paths == ["one.txt", "two.txt"]


def test_files_visible_to_runner_after_upload(client: TestClient) -> None:
    """Files uploaded via HTTP must be readable from inside the REPL."""
    sid = _create_session(client)
    client.post(
        f"/sessions/{sid}/files",
        files={"file": ("greeting.txt", BytesIO(b"hello"), "text/plain")},
    )
    response = client.post(
        f"/sessions/{sid}/exec",
        json={
            "tool": "code_execution",
            "input": {"code": "print(open('greeting.txt').read())"},
        },
    )
    assert response.status_code == 200
    assert response.json()["result_block"]["content"]["stdout"] == "hello"


def test_files_isolated_across_sessions(client: TestClient) -> None:
    """A file uploaded into session A is invisible from session B."""
    sid_a = _create_session(client)
    sid_b = _create_session(client)
    client.post(
        f"/sessions/{sid_a}/files",
        files={"file": ("only-a.txt", BytesIO(b"a"), "text/plain")},
    )
    response = client.get(
        f"/sessions/{sid_b}/files", params={"path": "only-a.txt"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Programmatic tools registration
# ---------------------------------------------------------------------------


def test_programmatic_tools_registered_per_session(client: TestClient) -> None:
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/programmatic-tools/register",
        json={
            "callback_url": "http://router.local/tool-call",
            "token": "secret",
            "tools": [{"name": "echo", "doc": "Echo back."}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tool_count"] == 1
    # The stub path lives under the session's lib dir, not a global one.
    assert "lib" in body["stub_path"]
