#!/usr/bin/env bash
# Submit with:
#   sbatch ./serve-slurm-rtx.sh
#
# All runtime state stays inside this checkout: logs, endpoint metadata, GGUF
# files, KV cache, and the ds4-server binary.

#SBATCH --job-name=ds4-v4flash-rtx
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/slurm-ds4-rtx-%j.out
#SBATCH --error=logs/slurm-ds4-rtx-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx.sh" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fi
LOG_DIR="$ROOT/logs"
KV_DIR="${DS4_KV_DIR:-$ROOT/kv}"
MODEL_PATH="${DS4_MODEL_PATH:-$ROOT/ds4flash.gguf}"
MTP_PATH="${DS4_MTP_PATH:-}"
HOST="${DS4_HOST:-0.0.0.0}"
PORT="${DS4_PORT:-8000}"
CTX="${DS4_CTX:-1000000}"
TOKENS="${DS4_MAX_TOKENS:-384000}"
GPUS="${DS4_SLURM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
ENDPOINT_FILE="$LOG_DIR/current-ds4.env"
SLURM_MODE="${DS4_SLURM_MODE:-server}"

mkdir -p "$LOG_DIR" "$KV_DIR"

if [ -n "${DS4_NVRTC_LIB_DIR:-}" ] && [ -d "$DS4_NVRTC_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$DS4_NVRTC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif command -v python >/dev/null 2>&1; then
    NVRTC_LIB_DIR="$(python -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("nvidia.cuda_nvrtc"); print(pathlib.Path(spec.origin).parent / "lib" if spec and spec.origin else "")' 2>/dev/null || true)"
    if [ -n "$NVRTC_LIB_DIR" ] && [ -d "$NVRTC_LIB_DIR" ]; then
        export LD_LIBRARY_PATH="$NVRTC_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ds4: refusing to start: nvidia-smi is not available in this Slurm allocation" >&2
    exit 2
fi

GPU_NAMES=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
JOB_ID="${SLURM_JOB_ID:-manual}"
DIRECT_BASE_URL="http://${NODE_NAME}:${PORT}/v1"
LOCAL_BASE_URL="http://127.0.0.1:${PORT}/v1"

if [ ! -e "$MODEL_PATH" ]; then
    echo "ds4: model not found at $MODEL_PATH" >&2
    echo "ds4: run ./download_model.sh q2 or set DS4_MODEL_PATH before submitting" >&2
    exit 2
fi

case "$SLURM_MODE" in
    server|smoke) ;;
    *)
        echo "ds4: invalid DS4_SLURM_MODE=$SLURM_MODE; expected server or smoke" >&2
        exit 2
        ;;
esac

if [ "$SLURM_MODE" = smoke ]; then
    make -C "$ROOT" ds4
else
    make -C "$ROOT" ds4-server
fi

{
    printf 'DS4_JOB_ID=%q\n' "$JOB_ID"
    printf 'DS4_JOB_NAME=%q\n' "ds4-v4flash-rtx"
    printf 'DS4_BACKEND=%q\n' "cuda"
    printf 'DS4_GPUS_ON_NODE=%q\n' "$GPUS"
    printf 'DS4_GPU_NAMES=%q\n' "$GPU_NAMES"
    printf 'DS4_HOST=%q\n' "$NODE_NAME"
    printf 'DS4_PORT=%q\n' "$PORT"
    printf 'DS4_CTX=%q\n' "$CTX"
    printf 'DS4_MODEL=%q\n' "deepseek-v4-flash"
    printf 'DS4_MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'DS4_SLURM_MODE=%q\n' "$SLURM_MODE"
    printf 'OPENAI_BASE_URL=%q\n' "$DIRECT_BASE_URL"
    printf 'LOCAL_OPENAI_BASE_URL=%q\n' "$LOCAL_BASE_URL"
    printf 'OPENAI_API_KEY=%q\n' "dsv4-local"
} > "$ENDPOINT_FILE"

echo "[$(date -Iseconds)] ds4 Slurm RTX endpoint metadata: $ENDPOINT_FILE"
echo "[$(date -Iseconds)] node: $NODE_NAME"
echo "[$(date -Iseconds)] GPUs: $GPU_NAMES"
echo "[$(date -Iseconds)] direct OpenAI base URL: $DIRECT_BASE_URL"
echo "[$(date -Iseconds)] context: $CTX"

if [ "$SLURM_MODE" = smoke ]; then
    SMOKE_LOG="$LOG_DIR/cuda-smoke-${JOB_ID}.log"
    echo "[$(date -Iseconds)] running CUDA smoke probe: $SMOKE_LOG"
    set +e
    "$ROOT/ds4" --backend cuda --model "$MODEL_PATH" --ctx "$CTX" --inspect --session-smoke > "$SMOKE_LOG" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        status="cuda-smoke-pass"
    else
        status="cuda-smoke-fail"
    fi
    {
        printf 'DS4_JOB_ID=%q\n' "$JOB_ID"
        printf 'DS4_JOB_NAME=%q\n' "ds4-v4flash-rtx"
        printf 'DS4_BOOTSTRAP_STATUS=%q\n' "$status"
        printf 'DS4_BACKEND=%q\n' "cuda"
        printf 'DS4_GPUS_ON_NODE=%q\n' "$GPUS"
        printf 'DS4_GPU_NAMES=%q\n' "$GPU_NAMES"
        printf 'DS4_HOST=%q\n' "$NODE_NAME"
        printf 'DS4_PORT=%q\n' "$PORT"
        printf 'DS4_CTX=%q\n' "$CTX"
        printf 'DS4_MODEL=%q\n' "deepseek-v4-flash"
        printf 'DS4_MODEL_PATH=%q\n' "$MODEL_PATH"
        printf 'DS4_SMOKE_LOG=%q\n' "$SMOKE_LOG"
        printf 'DS4_SLURM_MODE=%q\n' "$SLURM_MODE"
    } > "$ENDPOINT_FILE"
    cat "$SMOKE_LOG"
    exit "$rc"
fi

ARGS=(
    --backend cuda
    --model "$MODEL_PATH"
    --ctx "$CTX"
    --tokens "$TOKENS"
    --host "$HOST"
    --port "$PORT"
    --kv-disk-dir "$KV_DIR"
    --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB:-32768}"
)

if [ -n "$MTP_PATH" ]; then
    ARGS+=(--mtp "$MTP_PATH" --mtp-draft "${DS4_MTP_DRAFT:-2}")
fi

exec "$ROOT/ds4-server" "${ARGS[@]}"
