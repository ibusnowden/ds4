#!/usr/bin/env bash
# Logits-level parity probe: runs the same prompt through the CUDA executor
# and the CPU reference per-token, then prints the post-prefill logits delta
# (max-abs, RMS, top-1 / top-5 agreement). Diagnostic; does not assert.

#SBATCH --job-name=ds4-cuda-parity-probe
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/cuda-parity-probe-%j.out
#SBATCH --error=logs/cuda-parity-probe-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx-parity-probe.sh" ]; then
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
PROMPT="The quick brown fox jumps over the lazy dog."

echo "=== CUDA parity probe: $PROMPT (ctx=$CTX) ==="
"$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
            --cuda-parity-probe -p "$PROMPT" 2>&1
rc=$?
echo "=== exit=$rc (0 = top-1 match and max-abs < 0.5; nonzero = divergence) ==="
exit "$rc"
