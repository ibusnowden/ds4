#!/usr/bin/env bash
# Print the current ds4 OpenAI-compatible endpoint from this checkout.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FILE="${DS4_ENDPOINT_FILE:-$ROOT/logs/current-ds4.env}"
MODE="direct"

case "${1:-}" in
    --local) MODE="local"; shift ;;
    --export) MODE="export"; shift ;;
    -h|--help)
        printf 'Usage: %s [--local|--export]\n' "$0"
        exit 0
        ;;
esac

if [ ! -f "$FILE" ]; then
    printf 'No endpoint metadata found: %s\n' "$FILE" >&2
    printf 'Submit ./serve-slurm-rtx.sh first, or set DS4_ENDPOINT_FILE.\n' >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$FILE"

case "$MODE" in
    local)
        printf '%s\n' "${LOCAL_OPENAI_BASE_URL:-http://127.0.0.1:${DS4_PORT:-8000}/v1}"
        ;;
    export)
        printf 'export OPENAI_API_KEY=%q\n' "${OPENAI_API_KEY:-dsv4-local}"
        printf 'export OPENAI_BASE_URL=%q\n' "${OPENAI_BASE_URL:?missing OPENAI_BASE_URL in $FILE}"
        ;;
    *)
        printf '%s\n' "${OPENAI_BASE_URL:?missing OPENAI_BASE_URL in $FILE}"
        ;;
esac
