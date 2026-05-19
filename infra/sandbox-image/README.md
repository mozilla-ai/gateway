# sandbox-image

Code-execution sandbox image used by the gateway. Built locally by
`docker compose` for now; a published image to a registry is a follow-up.

The container runs a multi-session HTTP API for Python/bash/text-editor
operations. Each session has its own `/var/sandbox/sessions/<id>/` tree with
its own workspace and REPL subprocess.

The wire shapes returned by `POST /exec` match Anthropic's
`code_execution_20250825` content blocks (`code_execution_tool_result`,
`bash_code_execution_tool_result`, `text_editor_code_execution_tool_result`)
so consumers that already parse Anthropic shapes work without translation.

## Layout

```
sandbox/
  models.py           # Pydantic shapes (request, result blocks)
  runner.py           # Long-lived Python REPL with sentinel protocol
  exec_server.py      # FastAPI app: /sessions, /exec, /files, /health
  text_editor.py      # view/create/str_replace/insert/undo_edit handlers
tests/
Dockerfile            # python:3.12-slim base, pinned package set
Makefile              # build, test, run
```

## Local development

```sh
make install   # uv sync
make test      # run pytest
make build     # build the Docker image
make run       # docker run -p 8080:8080
```

## API

```
POST   /sessions                       -> create session
POST   /sessions/{id}/exec             -> run code in session
POST   /sessions/{id}/exec/stream      -> SSE: stdout/stderr deltas
DELETE /sessions/{id}                  -> destroy session
GET    /health                         -> 200 if server alive
```

The full request/response schemas live in `sandbox/models.py`.
