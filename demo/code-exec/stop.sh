#!/usr/bin/env bash
# Tear the demo stack down.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$GATEWAY_ROOT"
exec docker compose --env-file "$HERE/.env" down "$@"
