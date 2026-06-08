#!/usr/bin/env bash
# Parity + bench for CUDA Graph capture in chunked prefill.
# Runs --cuda-chunked-prefill-test twice (DS4_CUDA_GRAPHS=0 / =1) on the same
# prompts and compares the test's pass/fail and logged max_abs/RMS deltas.
# Times each invocation as a coarse bench against the chunked baseline.

#SBATCH --job-name=ds4-cuda-graphs-parity
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cuda-graphs-parity-%j.out
#SBATCH --error=logs/cuda-graphs-parity-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/cuda-graphs-parity.sh" ]; then
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
PROMPT="The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs."

run_one() {
    local mode="$1"
    echo "=== DS4_CUDA_GRAPHS=$mode ==="
    local t0=$(date +%s.%N)
    set +e
    env DS4_CUDA_GRAPHS="$mode" \
        "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
                    --cuda-chunked-prefill-test -p "$PROMPT" 2>&1
    local rc=$?
    set -e
    local t1=$(date +%s.%N)
    awk -v t0="$t0" -v t1="$t1" -v rc="$rc" 'BEGIN { printf "[wallclock] DS4_CUDA_GRAPHS=%s rc=%d elapsed=%.3fs\n", ENVIRON["MODE"], rc, t1 - t0 }' MODE="$mode"
    echo
}

run_one 0
run_one 1
