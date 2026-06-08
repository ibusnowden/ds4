#!/usr/bin/env bash
# End-to-end bench driver. Runs from any client (Pi or laptop) against a
# ds4-server. For each prompt size (short, medium, long) it sends a
# /v1/chat/completions request via streaming SSE so we can measure:
#
#   - ttft_s   : wall-clock time to first generated token (~= prefill time)
#   - decode_s : wall-clock from first to last generated token
#   - e2e_s    : total wall-clock for the request
#   - gen_tok  : count of streamed delta tokens
#   - decode_tps, e2e_tps
#
# Detailed prefill/decode/e2e tps from the server's perspective are also in
# the server log line "ds4-server: bench id=... prefill_tps=... decode_tps=...
# e2e_tps=..." — capture that on the server side for the most accurate numbers.
#
# Usage:
#   bench/pi-bench.sh [-u BASE_URL] [-k API_KEY] [-m MODEL] [-n MAX_TOKENS]
#                     [-o OUT_CSV] [PROMPT_NAME...]

set -euo pipefail

usage() {
    cat <<USAGE
pi-bench: drive ds4-server with short / medium / long prompts and record CSV.

  -u URL    OpenAI-compatible base URL (e.g. http://127.0.0.1:8000/v1).
            Falls back to OPENAI_BASE_URL or logs/current-ds4.env.
  -k KEY    API key (falls back to OPENAI_API_KEY).
  -m MODEL  Model name (default: deepseek-v4-flash).
  -n N      max_tokens per request (default: 256).
  -o FILE   CSV output path (default: bench/results/pi-<UTC>.csv).
  PROMPT_NAME...  optional names from bench/prompts/*.txt to run.
                  Default: short medium long.
USAGE
}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE_URL="${OPENAI_BASE_URL:-}"
API_KEY="${OPENAI_API_KEY:-dsv4-local}"
MODEL="${DS4_MODEL_NAME:-deepseek-v4-flash}"
MAX_TOKENS=256
OUT_CSV=""

while getopts ":u:k:m:n:o:h" opt; do
    case "$opt" in
        u) BASE_URL="$OPTARG" ;;
        k) API_KEY="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        n) MAX_TOKENS="$OPTARG" ;;
        o) OUT_CSV="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

PROMPTS=("$@")
if [ "${#PROMPTS[@]}" -eq 0 ]; then
    PROMPTS=(short medium long)
fi

if [ -z "$BASE_URL" ] && [ -f "$ROOT/logs/current-ds4.env" ]; then
    # shellcheck disable=SC1090
    source "$ROOT/logs/current-ds4.env"
    BASE_URL="${OPENAI_BASE_URL:-${LOCAL_OPENAI_BASE_URL:-}}"
fi
if [ -z "$BASE_URL" ]; then
    echo "pi-bench: no base URL. Set -u, OPENAI_BASE_URL, or run after serve-slurm-rtx.sh." >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "pi-bench: python3 is required" >&2; exit 2
fi
if [ -z "$OUT_CSV" ]; then
    mkdir -p "$ROOT/bench/results"
    OUT_CSV="$ROOT/bench/results/pi-$(date -u +%Y%m%dT%H%M%SZ).csv"
fi

echo "pi-bench: base_url=$BASE_URL model=$MODEL max_tokens=$MAX_TOKENS"
echo "pi-bench: writing $OUT_CSV"
echo "prompt,prompt_chars,gen_tok,ttft_s,decode_s,e2e_s,decode_tps,e2e_tps,server_id" > "$OUT_CSV"

for name in "${PROMPTS[@]}"; do
    file="$ROOT/bench/prompts/${name}.txt"
    if [ ! -r "$file" ]; then
        echo "pi-bench: skipping $name: $file not readable" >&2
        continue
    fi
    DS4_PI_BENCH_NAME="$name" \
    DS4_PI_BENCH_FILE="$file" \
    DS4_PI_BENCH_URL="${BASE_URL%/}/chat/completions" \
    DS4_PI_BENCH_KEY="$API_KEY" \
    DS4_PI_BENCH_MODEL="$MODEL" \
    DS4_PI_BENCH_MAX="$MAX_TOKENS" \
    DS4_PI_BENCH_CSV="$OUT_CSV" \
    python3 - <<'PY'
import json, os, time, urllib.request, urllib.error, sys

name = os.environ["DS4_PI_BENCH_NAME"]
url = os.environ["DS4_PI_BENCH_URL"]
key = os.environ["DS4_PI_BENCH_KEY"]
model = os.environ["DS4_PI_BENCH_MODEL"]
max_tok = int(os.environ["DS4_PI_BENCH_MAX"])
csv_path = os.environ["DS4_PI_BENCH_CSV"]
with open(os.environ["DS4_PI_BENCH_FILE"], "r") as f:
    prompt = f.read()

body = json.dumps({
    "model": model,
    "stream": True,
    "max_tokens": max_tok,
    "temperature": 0,
    "messages": [{"role": "user", "content": prompt}],
}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=body,
    method="POST",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    },
)
t0 = time.monotonic()
first = -1.0
last = -1.0
gen = 0
sid = ""
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not sid:
                sid = obj.get("id", "")
            choices = obj.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.monotonic()
                    if first < 0:
                        first = now
                    last = now
                    gen += 1
except urllib.error.HTTPError as e:
    print(f"pi-bench: {name} HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}", file=sys.stderr)
    sys.exit(1)
t_end = time.monotonic()
ttft = (first - t0) if first >= 0 else (t_end - t0)
dec = (last - first) if last > first else 0.0
e2e = t_end - t0
dec_tps = (gen / dec) if dec > 0 else 0.0
e2e_tps = (gen / e2e) if e2e > 0 else 0.0
nchars = len(prompt)
with open(csv_path, "a") as f:
    f.write(f"{name},{nchars},{gen},{ttft:.3f},{dec:.3f},{e2e:.3f},{dec_tps:.2f},{e2e_tps:.2f},{sid}\n")
print(f"{name:<7s} prompt_chars={nchars} gen={gen} ttft={ttft:.2f}s decode={dec:.2f}s e2e={e2e:.2f}s decode_tps={dec_tps:.2f} e2e_tps={e2e_tps:.2f}")
PY
done

echo "pi-bench: done. CSV: $OUT_CSV"
echo "pi-bench: server-side detail: grep 'ds4-server: bench' on the cluster log."
