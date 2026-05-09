#!/usr/bin/env bash
# Match the README "Speed" table conditions: --ctx 32768 --nothink -n 256, greedy,
# short Italian story prompt. Records the prefill/generation tok/s lines.

#SBATCH --job-name=ds4-cuda-bench
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/cuda-bench-%j.out
#SBATCH --error=logs/cuda-bench-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx-bench.sh" ]; then
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

PROMPT="Scrivi una storia su una papera scansafatiche."
echo "=== short prompt: ctx=32768 -n 256 --nothink --temp 0 ==="
time "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" \
    --ctx 32768 --nothink -n 256 --temp 0 \
    -p "$PROMPT" \
    > /dev/null 2>>logs/cuda-bench-stderr.log
tail -3 logs/cuda-bench-stderr.log
