#!/usr/bin/env bash
# Walkthrough for code execution against the OSS gateway.
#
# The gateway accepts three equivalent tool-array shapes for code execution:
#   * {"type": "code_execution"}              — gateway short form
#   * {"type": "code_interpreter"}            — OpenAI Responses/Assistants
#   * {"type": "code_execution_20250825"}     — Anthropic, versioned
#
# All three map to the same sandbox backend, so swapping the OpenAI/Anthropic
# SDK's `base_url` to the gateway keeps existing client code working.
#
# Usage:
#   ./demo_flow.sh                                  # runs every provider that has credentials
#   ./demo_flow.sh --anthropic                      # subset of providers, in the given order
#   ./demo_flow.sh --openai --llamafile             # multiple flags compose
#
# Provider preconditions (the script checks each before running):
#   --anthropic   needs ANTHROPIC_API_KEY in .env
#   --openai      needs OPENAI_API_KEY in .env
#   --llamafile   needs a llamafile server reachable from the gateway container
#                 (LLAMAFILE_API_BASE; default http://host.docker.internal:8080/v1)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ASK="$HERE/ask.sh"

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

BOLD=$'\e[1m'; DIM=$'\e[2m'; YEL=$'\e[33m'; CYN=$'\e[36m'; GRN=$'\e[32m'; RED=$'\e[31m'; RST=$'\e[0m'

# Provider selection ────────────────────────────────────────────────────────
PROVIDERS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anthropic) PROVIDERS+=("anthropic"); shift ;;
    --openai)    PROVIDERS+=("openai");    shift ;;
    --llamafile) PROVIDERS+=("llamafile"); shift ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
# Default: all providers, in a stable order.
if [[ ${#PROVIDERS[@]} -eq 0 ]]; then
  PROVIDERS=(anthropic openai llamafile)
fi

# Preflight: drop providers without satisfied prereqs and warn the operator.
ENABLED=()
for p in "${PROVIDERS[@]}"; do
  case "$p" in
    anthropic)
      if [[ -n "${ANTHROPIC_API_KEY:-}" && "$ANTHROPIC_API_KEY" != *REPLACE_ME* ]]; then
        ENABLED+=("anthropic")
      else
        echo "${YEL}skipping --anthropic: ANTHROPIC_API_KEY not set in .env${RST}"
      fi
      ;;
    openai)
      if [[ -n "${OPENAI_API_KEY:-}" && "$OPENAI_API_KEY" != *REPLACE_ME* ]]; then
        ENABLED+=("openai")
      else
        echo "${YEL}skipping --openai: OPENAI_API_KEY not set in .env${RST}"
      fi
      ;;
    llamafile)
      base=${LLAMAFILE_API_BASE:-http://host.docker.internal:8080/v1}
      # The compose-mounted base points at host.docker.internal from inside
      # the gateway container; on the host, the same llamafile is reachable
      # at 127.0.0.1 — translate for the local reachability check.
      probe=${base/host.docker.internal/127.0.0.1}
      if curl -sf "${probe}/models" >/dev/null 2>&1; then
        ENABLED+=("llamafile")
      else
        echo "${YEL}skipping --llamafile: no llamafile reachable at $probe${RST}"
      fi
      ;;
  esac
done

if [[ ${#ENABLED[@]} -eq 0 ]]; then
  echo "${RED}no providers available. Set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env"
  echo "or start a llamafile on http://localhost:8080.${RST}"
  exit 1
fi

# Provider → (model, tool-type, ask.sh extra flags) ─────────────────────────
provider_model() {
  case "$1" in
    anthropic) echo "anthropic:claude-sonnet-4-6" ;;
    openai)    echo "openai:gpt-4o-mini" ;;
    llamafile)
      local base probe
      base=${LLAMAFILE_API_BASE:-http://host.docker.internal:8080/v1}
      probe=${base/host.docker.internal/127.0.0.1}
      echo "llamafile:$(curl -sS "${probe}/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
      ;;
  esac
}
provider_tool_type() {
  case "$1" in
    anthropic) echo "code_execution_20250825" ;;  # Anthropic's native versioned shape
    openai)    echo "code_interpreter" ;;          # OpenAI's native shape
    llamafile) echo "code_execution" ;;            # gateway-native short form
  esac
}
provider_extra() {
  # Llamafile doesn't stream tool calls; force non-streaming for that one.
  [[ "$1" == "llamafile" ]] && echo "--no-stream" || echo ""
}

pause() { echo; read -r -p "${DIM}  press Enter…${RST}" _; echo; }

present() {
  local title="$1"; shift
  echo
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  echo "${BOLD}${YEL}  $title${RST}"
  echo "${BOLD}${YEL}═══════════════════════════════════════════════════════════════════════${RST}"
  for line in "$@"; do
    echo "${DIM}  $line${RST}"
  done
  echo
}

architecture_diagram() {
  cat <<EOF
${BOLD}what's running${RST} (all via \`docker compose up\` in the gateway repo)

                ┌──────────────────────────┐
                │  Client (./ask.sh, curl, │
                │  your app, an SDK)       │
                └─────────────┬────────────┘
                              │
                  POST /v1/chat/completions
                              │
                              ▼
            ┌──────────────────────────────────┐         ┌─────────────┐
            │   OSS Gateway                    │ ──────▶ │  Postgres   │
            │                                  │  state  └─────────────┘
            │   tool-use loop:                 │
            │    • detects code_execution /    │
            │      code_interpreter /          │
            │      code_execution_<ver>        │
            │    • adds it to model's tools[]  │
            │    • on tool_call: POST sandbox  │
            │    • feed stdout/stderr back     │
            │    • stream SSE to client        │
            └────┬───────────────────────┬─────┘
                 │                       │
       Provider chat API        sandbox HTTP API
                 │                       │
                 ▼                       ▼
       ┌─────────────────┐    ┌──────────────────────────┐
       │  Any model      │    │  Sandbox container       │
       │  any-llm        │    │  Python REPL,            │
       │  routes to      │    │  pandas/numpy/scipy …    │
       └─────────────────┘    └──────────────────────────┘
EOF
}

show_request_shapes() {
  local query="$1"
  cat <<EOF
${BOLD}three shapes the gateway accepts${RST} — same backend, different SDKs:

  ${DIM}# Gateway-native (terse):${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "code_execution" }]${RST}
  ${CYN}}${RST}

  ${DIM}# OpenAI SDK code — switch base_url to the gateway, no code change:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "code_interpreter" }]${RST}
  ${CYN}}${RST}

  ${DIM}# Anthropic SDK code — same idea, versioned tool type:${RST}
  ${CYN}{${RST}
  ${CYN}  "messages": [{ "role": "user", "content": "$query" }],${RST}
  ${GRN}  "tools": [{ "type": "code_execution_20250825" }]${RST}
  ${CYN}}${RST}
EOF
}

show_what_llm_receives() {
  local query="$1"

  echo "${BOLD}A) bare request — gateway uses defaults${RST}"
  echo
  printf "${CYN}{${RST}\n"
  printf "${CYN}  \"messages\": [{ \"role\": \"user\", \"content\": \"%s\" }],${RST}\n" "$query"
  printf "${GRN}  \"tools\": [{ \"type\": \"code_execution\" }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec gateway-gateway-1 python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.sandbox_backend import SandboxBackend

backend = SandboxBackend(sandbox_url='http://sandbox:8080')
print(inject_purpose_hints(
    [{'role': 'user', 'content': sys.argv[1]}],
    backend.purpose_hints(),
)[0]['content'])
" "$query" 2>&1 | sed "s/^/    ${GRN}/; s/$/${RST}/"

  echo
  echo "${BOLD}B) per-request override — \"purpose_hint\" on the tool entry${RST}"
  echo
  printf "${CYN}{${RST}\n"
  printf "${CYN}  \"messages\": [{ \"role\": \"user\", \"content\": \"%s\" }],${RST}\n" "$query"
  printf "${GRN}  \"tools\": [{${RST}\n"
  printf "${GRN}    \"type\": \"code_execution\",${RST}\n"
  printf "${YEL}    \"purpose_hint\": \"Only use code_execution for math involving large numbers.\"${RST}\n"
  printf "${GRN}  }]${RST}\n"
  printf "${CYN}}${RST}\n"
  echo
  echo "${DIM}  system message the gateway prepends:${RST}"
  docker exec gateway-gateway-1 python -c "
import sys
sys.path.insert(0, '/app/src')
from gateway.services.mcp_loop import inject_purpose_hints
from gateway.services.sandbox_backend import SandboxBackend

backend = SandboxBackend(
    sandbox_url='http://sandbox:8080',
    purpose_hint='Only use code_execution for math involving large numbers.',
)
print(inject_purpose_hints(
    [{'role': 'user', 'content': sys.argv[1]}],
    backend.purpose_hints(),
)[0]['content'])
" "$query" 2>&1 | sed "s/^/    ${YEL}/; s/$/${RST}/"

  echo
  echo "${DIM}Knobs that shape the system message — priority order:${RST}"
  echo
  echo "${DIM}  Per-tool hint (e.g. 'Use this for math'):${RST}"
  echo "${DIM}    1. tools[i].purpose_hint              (per-request)${RST}"
  echo "${DIM}    2. GATEWAY_SANDBOX_PURPOSE_HINT       (env, per-deployment)${RST}"
  echo "${DIM}    3. built-in default${RST}"
  echo
  echo "${DIM}  List header (e.g. 'Prefer MCP tools over code_execution'):${RST}"
  echo "${DIM}    1. tools_header (top-level request field)  (per-request)${RST}"
  echo "${DIM}    2. GATEWAY_TOOLS_HEADER                     (env, per-deployment)${RST}"
  echo "${DIM}    3. 'You have access to the following tools:'  (built-in)${RST}"
}

# ───────────────────────────────────────────────────────────────────────
architecture_diagram
pause


# ───────────────────────────────────────────────────────────────────────
QUERY="Compute 23! and 50! using code_execution. Show both."

present "Under the hood: what the LLM actually receives" \
        "Two things the gateway adds before forwarding to the model:" \
        " (a) a system message naming each tool source and its purpose hint" \
        " (b) a tools[] entry for code_execution" \
        "The client never mentions code_execution — the gateway injects it."
show_what_llm_receives "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
present "Same gateway, three request shapes" \
        "The gateway accepts the keyword each SDK already emits, so swapping" \
        "the SDK's base_url to the gateway keeps existing code working."
show_request_shapes "$QUERY"
pause


# ───────────────────────────────────────────────────────────────────────
N=${#ENABLED[@]}
present "Code-execution end to end — ${N} provider$( [[ $N -gt 1 ]] && echo s )" \
        "Same question, same sandbox. Each call uses the SDK keyword native" \
        "to its provider — proving the gateway recognises all three."
for p in "${ENABLED[@]}"; do
  model=$(provider_model "$p")
  tool_type=$(provider_tool_type "$p")
  extra=$(provider_extra "$p")

  echo
  echo "${BOLD}${GRN}── $p ── model=$model tool-type=$tool_type${RST}"
  # shellcheck disable=SC2086
  "$ASK" --model "$model" --tool-type "$tool_type" $extra "$QUERY"
done
pause


# ───────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo "${BOLD}${GRN}  fin — questions?${RST}"
echo "${BOLD}${GRN}═══════════════════════════════════════════════════════════════════════${RST}"
echo
echo "${DIM}  Same flow works for any model the gateway can route to —${RST}"
echo "${DIM}  open-weight or frontier. The gateway just runs the loop.${RST}"
