#!/usr/bin/env bash
# Bring up the OSS gateway demo (gateway + sandbox + postgres) with the keys
# and ports configured in this folder's .env. Loads .env via docker-compose's
# --env-file so the API keys never need to live in shell history.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_FILE="$HERE/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy .env.example to .env and fill in your keys." >&2
  exit 1
fi

if grep -qE '^[A-Z_]+=.*REPLACE_ME' "$ENV_FILE"; then
  echo "$ENV_FILE has an uncommented REPLACE_ME — fill in your real key before starting." >&2
  exit 1
fi

cd "$GATEWAY_ROOT"
exec docker compose --env-file "$ENV_FILE" up "$@"
