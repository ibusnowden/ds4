#!/usr/bin/env bash
# One-command repo-local bootstrap for daily Pi coding with ds4.
#
# Default:
#   ./start-ds4-pi.sh
#
# It ensures the q2 GGUF is present, submits the Slurm RTX server job, waits
# for /v1/models, verifies Pi metadata, runs a small benchmark, and prints the
# environment to launch Pi against the endpoint.

set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOG_DIR="$ROOT/logs"
BENCH_DIR="$LOG_DIR/benchmarks"
ENDPOINT_FILE="${DS4_ENDPOINT_FILE:-$LOG_DIR/current-ds4.env}"
MODEL_LINK="$ROOT/ds4flash.gguf"
MODEL_ID="deepseek-v4-flash"
API_KEY="${DS4_API_KEY:-dsv4-local}"

QUANT="${DS4_MODEL_QUANT:-q2}"
CTX="${DS4_CTX:-1000000}"
PORT="${DS4_PORT:-8000}"
TOKENS="${DS4_MAX_TOKENS:-384000}"
TIMEOUT="${DS4_STARTUP_TIMEOUT:-1800}"
DOWNLOAD=1
RUN_BENCH=1
MODE=start

usage() {
    cat <<EOF
Usage: ./start-ds4-pi.sh [options]

Modes:
  --status          Show current endpoint/job status.
  --stop            Cancel the recorded Slurm job and archive endpoint metadata.
  --cuda-smoke      Submit a short CUDA allocation probe and exit.

Options:
  --quant q2|q4     Model quant to download if ds4flash.gguf is missing. Default: q2
  --ctx N           Server context. Default: 1000000
  --port N          Server port. Default: 8000
  --timeout SEC     Startup wait timeout. Default: 1800
  --no-download     Fail if model weights are missing instead of downloading.
  --no-benchmark    Skip the startup benchmark.
  -h, --help        Show this help.

Runtime files stay under:
  $ROOT
EOF
}

log() {
    printf '[ds4-pi %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf '[ds4-pi %s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --status) MODE=status; shift ;;
        --stop) MODE=stop; shift ;;
        --cuda-smoke) MODE=cuda-smoke; shift ;;
        --quant) QUANT="${2:-}"; shift 2 ;;
        --quant=*) QUANT="${1#--quant=}"; shift ;;
        --ctx) CTX="${2:-}"; shift 2 ;;
        --ctx=*) CTX="${1#--ctx=}"; shift ;;
        --port) PORT="${2:-}"; shift 2 ;;
        --port=*) PORT="${1#--port=}"; shift ;;
        --timeout) TIMEOUT="${2:-}"; shift 2 ;;
        --timeout=*) TIMEOUT="${1#--timeout=}"; shift ;;
        --no-download) DOWNLOAD=0; shift ;;
        --no-benchmark) RUN_BENCH=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$QUANT" in
    q2|q4) ;;
    *) die "invalid --quant $QUANT; expected q2 or q4" ;;
esac

mkdir -p "$LOG_DIR" "$BENCH_DIR"

source_endpoint() {
    [ -f "$ENDPOINT_FILE" ] || return 1
    # shellcheck source=/dev/null
    source "$ENDPOINT_FILE"
}

job_state() {
    local jobid="$1"
    command -v squeue >/dev/null 2>&1 || return 0
    squeue -h -j "$jobid" -o "%T" 2>/dev/null | head -n1
}

job_node() {
    local jobid="$1"
    local node
    node="$(squeue -h -j "$jobid" -o "%N" 2>/dev/null | awk 'NF { print; exit }')"
    [ -n "$node" ] && [ "$node" != "(null)" ] || return 1
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$node" 2>/dev/null | awk 'NF { print; exit }'
    else
        printf '%s\n' "$node"
    fi
}

http_ready() {
    local base_url="$1"
    command -v curl >/dev/null 2>&1 || return 1
    curl -sf -o /dev/null --max-time 5 "$base_url/models"
}

write_endpoint() {
    local jobid="$1" node="$2" status="$3"
    local direct="http://${node}:${PORT}/v1"
    local local_url="http://127.0.0.1:${PORT}/v1"
    {
        printf 'DS4_JOB_ID=%q\n' "$jobid"
        printf 'DS4_JOB_NAME=%q\n' "ds4-v4flash-rtx"
        printf 'DS4_BOOTSTRAP_STATUS=%q\n' "$status"
        printf 'DS4_BACKEND=%q\n' "cuda"
        printf 'DS4_HOST=%q\n' "$node"
        printf 'DS4_PORT=%q\n' "$PORT"
        printf 'DS4_CTX=%q\n' "$CTX"
        printf 'DS4_MODEL=%q\n' "$MODEL_ID"
        printf 'DS4_MODEL_PATH=%q\n' "$MODEL_LINK"
        printf 'OPENAI_BASE_URL=%q\n' "$direct"
        printf 'LOCAL_OPENAI_BASE_URL=%q\n' "$local_url"
        printf 'OPENAI_API_KEY=%q\n' "$API_KEY"
    } > "$ENDPOINT_FILE"
}

print_launch_env() {
    source_endpoint || die "endpoint metadata missing: $ENDPOINT_FILE"
    cat <<EOF

Endpoint is ready.

Run Pi with:
  export DS4_ENDPOINT_FILE=$ENDPOINT_FILE
  export DS4_BASE_URL=${OPENAI_BASE_URL}
  export DS4_MANAGED_SERVER=0
  pi

OpenAI-compatible endpoint:
  ${OPENAI_BASE_URL}

Endpoint metadata:
  $ENDPOINT_FILE
EOF
}

verify_pi_metadata() {
    local provider="$ROOT/pi-sd4-provider.ts"
    grep -q 'xhigh.*"xhigh"' "$provider" || die "Pi provider is missing xhigh effort mapping"
    grep -Eq 'contextWindow:[[:space:]]*1_?000_?000' "$provider" || die "Pi provider is missing 1M contextWindow"
    log "Pi metadata verified: xhigh effort and 1M context"
}

ensure_model() {
    if [ -e "$MODEL_LINK" ]; then
        log "model link present: $MODEL_LINK"
        return 0
    fi
    if [ "$DOWNLOAD" -eq 0 ]; then
        die "model weights are missing: $MODEL_LINK; run ./download_model.sh $QUANT or omit --no-download"
    fi
    log "model weights missing; downloading $QUANT into this checkout"
    "$ROOT/download_model.sh" "$QUANT"
    [ -e "$MODEL_LINK" ] || die "download completed but $MODEL_LINK is still missing"
}

show_status() {
    if ! source_endpoint; then
        die "no endpoint metadata found: $ENDPOINT_FILE"
    fi
    printf 'Endpoint file: %s\n' "$ENDPOINT_FILE"
    printf 'Model: %s\n' "${DS4_MODEL:-$MODEL_ID}"
    printf 'Context: %s\n' "${DS4_CTX:-unknown}"
    printf 'Backend: %s\n' "${DS4_BACKEND:-unknown}"
    printf 'Base URL: %s\n' "${OPENAI_BASE_URL:-unknown}"
    if [ -n "${DS4_JOB_ID:-}" ]; then
        printf 'Slurm job: %s %s\n' "$DS4_JOB_ID" "$(job_state "$DS4_JOB_ID")"
    fi
    if [ -n "${OPENAI_BASE_URL:-}" ] && http_ready "$OPENAI_BASE_URL"; then
        printf 'HTTP: ready\n'
    else
        printf 'HTTP: not ready\n'
    fi
    verify_pi_metadata
}

stop_job() {
    if ! source_endpoint; then
        die "no endpoint metadata found: $ENDPOINT_FILE"
    fi
    if [ -n "${DS4_JOB_ID:-}" ] && command -v scancel >/dev/null 2>&1; then
        scancel "$DS4_JOB_ID" 2>/dev/null || true
        log "cancelled Slurm job $DS4_JOB_ID"
    fi
    mv "$ENDPOINT_FILE" "$LOG_DIR/current-ds4.env.stopped-$(date +%s)"
    log "archived endpoint metadata"
}

submit_job() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found"
    (
        cd "$ROOT"
        DS4_CTX="$CTX" DS4_PORT="$PORT" DS4_MAX_TOKENS="$TOKENS" \
            DS4_SLURM_MODE="${DS4_SLURM_MODE:-server}" \
            sbatch --parsable ./serve-slurm-rtx.sh
    )
}

submit_smoke_job() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found"
    (
        cd "$ROOT"
        DS4_CTX="$CTX" DS4_PORT="$PORT" DS4_MAX_TOKENS="$TOKENS" \
            DS4_SLURM_MODE=smoke \
            sbatch --parsable ./serve-slurm-rtx.sh
    )
}

wait_running() {
    local jobid="$1"
    local deadline=$(( $(date +%s) + TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local state
        state="$(job_state "$jobid")"
        case "$state" in
            RUNNING) return 0 ;;
            "") die "Slurm job $jobid left the queue before running" ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                die "Slurm job $jobid failed before startup: $state"
                ;;
        esac
        log "waiting for job $jobid (${state:-unknown})"
        sleep 5
    done
    die "timed out waiting for job $jobid to run"
}

wait_endpoint() {
    local jobid="$1" node="$2"
    local base_url="http://${node}:${PORT}/v1"
    local deadline=$(( $(date +%s) + TIMEOUT ))
    write_endpoint "$jobid" "$node" "starting"
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if http_ready "$base_url"; then
            write_endpoint "$jobid" "$node" "ready"
            return 0
        fi
        local state
        state="$(job_state "$jobid")"
        case "$state" in
            RUNNING) ;;
            "") die "Slurm job $jobid exited before endpoint became ready; inspect logs/slurm-ds4-rtx-${jobid}.*" ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                die "Slurm job $jobid failed during startup: $state; inspect logs/slurm-ds4-rtx-${jobid}.*"
                ;;
        esac
        log "waiting for endpoint $base_url/models"
        sleep 10
    done
    write_endpoint "$jobid" "$node" "timeout"
    die "timed out waiting for endpoint $base_url"
}

wait_smoke_done() {
    local jobid="$1"
    local deadline=$(( $(date +%s) + TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local state
        state="$(job_state "$jobid")"
        if [ -z "$state" ]; then
            if command -v sacct >/dev/null 2>&1; then
                state="$(sacct -j "$jobid" -n -X -o State 2>/dev/null | awk 'NF { print $1; exit }')"
                case "$state" in
                    COMPLETED) return 0 ;;
                    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY) die "CUDA smoke job $jobid ended with $state; inspect logs/cuda-smoke-${jobid}.log" ;;
                    *) ;;
                esac
            fi
            [ -f "$LOG_DIR/cuda-smoke-${jobid}.log" ] && return 0
        fi
        log "waiting for CUDA smoke job $jobid (${state:-finishing})"
        sleep 5
    done
    die "timed out waiting for CUDA smoke job $jobid"
}

benchmark_endpoint() {
    source_endpoint || die "endpoint metadata missing before benchmark"
    [ "$RUN_BENCH" -eq 1 ] || return 0
    command -v curl >/dev/null 2>&1 || die "curl not found"
    command -v node >/dev/null 2>&1 || die "node not found"

    local ts req raw out start_ns end_ns elapsed_ns
    ts="$(date -u '+%Y%m%dT%H%M%SZ')"
    req="$BENCH_DIR/request-${ts}.json"
    raw="$BENCH_DIR/raw-${ts}.json"
    out="$BENCH_DIR/ds4-rtx-${ts}.json"

    cat > "$req" <<JSON
{
  "model": "$MODEL_ID",
  "messages": [
    {
      "role": "user",
      "content": "Write a small C function that counts ASCII whitespace in a buffer, then explain the edge cases in one paragraph."
    }
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": false
}
JSON

    log "running startup benchmark"
    start_ns="$(date +%s%N)"
    curl -sS --max-time 600 \
        -H "Authorization: Bearer ${OPENAI_API_KEY:-$API_KEY}" \
        -H "Content-Type: application/json" \
        -d @"$req" \
        "${OPENAI_BASE_URL}/chat/completions" > "$raw"
    end_ns="$(date +%s%N)"
    elapsed_ns=$((end_ns - start_ns))

    ELAPSED_NS="$elapsed_ns" RAW_FILE="$raw" OUT_FILE="$out" BASE_URL="$OPENAI_BASE_URL" DS4_CTX="$CTX" node <<'JS'
const fs = require("fs");
const raw = fs.readFileSync(process.env.RAW_FILE, "utf8");
let parsed;
try {
  parsed = JSON.parse(raw);
} catch (error) {
  parsed = { parse_error: String(error), raw_preview: raw.slice(0, 2000) };
}
const elapsed = Number(process.env.ELAPSED_NS) / 1e9;
const content = parsed?.choices?.[0]?.message?.content ?? "";
const ok = !parsed?.error && Array.isArray(parsed?.choices) && parsed.choices.length > 0;
const completionTokens = ok
  ? (parsed?.usage?.completion_tokens ?? Math.max(1, Math.round(content.length / 4)))
  : 0;
const result = {
  timestamp: new Date().toISOString(),
  base_url: process.env.BASE_URL,
  model: "deepseek-v4-flash",
  context: Number(process.env.DS4_CTX || "1000000"),
  ok,
  error: parsed?.error ?? parsed?.parse_error ?? null,
  elapsed_seconds: elapsed,
  completion_tokens: completionTokens,
  completion_tok_per_sec: ok && elapsed > 0 ? completionTokens / elapsed : null,
  usage: parsed?.usage ?? null,
  response_chars: content.length,
  raw_file: process.env.RAW_FILE
};
fs.writeFileSync(process.env.OUT_FILE, JSON.stringify(result, null, 2) + "\n");
if (ok) {
  console.log(`benchmark: ${result.completion_tok_per_sec.toFixed(2)} tok/s (${completionTokens} tokens, ${elapsed.toFixed(2)}s)`);
} else {
  console.log(`benchmark blocked: endpoint returned no completion (${elapsed.toFixed(2)}s)`);
}
console.log(`benchmark file: ${process.env.OUT_FILE}`);
JS
}

if [ "$MODE" = status ]; then
    show_status
    exit 0
fi

if [ "$MODE" = stop ]; then
    stop_job
    exit 0
fi

verify_pi_metadata
ensure_model

if [ "$MODE" = cuda-smoke ]; then
    log "submitting ds4 CUDA smoke job"
    jobid="$(submit_smoke_job)"
    log "submitted CUDA smoke job $jobid"
    wait_smoke_done "$jobid"
    log "CUDA smoke log: $LOG_DIR/cuda-smoke-${jobid}.log"
    exit 0
fi

if source_endpoint && [ -n "${OPENAI_BASE_URL:-}" ] && http_ready "$OPENAI_BASE_URL"; then
    log "existing endpoint is ready: $OPENAI_BASE_URL"
    benchmark_endpoint
    print_launch_env
    exit 0
fi

log "submitting ds4 Slurm RTX job"
jobid="$(submit_job)"
log "submitted job $jobid"
wait_running "$jobid"
node="$(job_node "$jobid")"
[ -n "$node" ] || die "could not resolve node for job $jobid"
log "job $jobid running on $node"
wait_endpoint "$jobid" "$node"
benchmark_endpoint
print_launch_env
