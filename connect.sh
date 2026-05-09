#!/usr/bin/env bash
# Open a localhost tunnel to the Slurm ds4 endpoint recorded by this checkout.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FILE="${DS4_ENDPOINT_FILE:-$ROOT/logs/current-ds4.env}"

if [ ! -f "$FILE" ]; then
    printf 'No endpoint metadata found: %s\n' "$FILE" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$FILE"

HOST="${DS4_HOST:?missing DS4_HOST in $FILE}"
PORT="${DS4_PORT:-8000}"

printf 'Opening SSH tunnel: localhost:%s -> %s:%s\n' "$PORT" "$HOST" "$PORT"
printf 'Use OPENAI_BASE_URL=http://127.0.0.1:%s/v1 while this tunnel is open.\n' "$PORT"
exec ssh -NL "${PORT}:${HOST}:${PORT}" "$HOST"
