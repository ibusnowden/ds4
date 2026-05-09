#!/usr/bin/env bash
# Parity smoke: run the same prompt through the new CUDA executor and the
# DS4_CUDA_BRIDGE fallback. Assert the first decoded token matches.

#SBATCH --job-name=ds4-cuda-parity-smoke
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cuda-parity-smoke-%j.out
#SBATCH --error=logs/cuda-parity-smoke-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx-parity.sh" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fi
cd "$ROOT"

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

make -C "$ROOT" ds4

CTX=4096
TOKENS=32
PROMPT="The quick brown fox jumps over the lazy dog."

GPU_OUT_FILE=$(mktemp)
BRIDGE_OUT_FILE=$(mktemp)
trap 'rm -f "$GPU_OUT_FILE" "$BRIDGE_OUT_FILE"' EXIT

echo "=== GPU executor ==="
"$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
            -p "$PROMPT" -n "$TOKENS" --temp 0 \
        > "$GPU_OUT_FILE" 2>/dev/null
cat "$GPU_OUT_FILE"
echo
echo "=== CPU bridge (DS4_CUDA_BRIDGE=1) ==="
DS4_CUDA_BRIDGE=1 "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
            -p "$PROMPT" -n "$TOKENS" --temp 0 \
        > "$BRIDGE_OUT_FILE" 2>/dev/null
cat "$BRIDGE_OUT_FILE"
echo

if diff -q "$GPU_OUT_FILE" "$BRIDGE_OUT_FILE" >/dev/null; then
    echo "=== PARITY: PASS (outputs identical) ==="
    exit 0
else
    echo "=== PARITY: FAIL (outputs differ) ==="
    diff -u "$BRIDGE_OUT_FILE" "$GPU_OUT_FILE" || true
    exit 1
fi
