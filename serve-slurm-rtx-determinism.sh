#!/usr/bin/env bash
#SBATCH --job-name=ds4-cuda-determinism
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cuda-det-%j.out
#SBATCH --error=logs/cuda-det-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx-determinism.sh" ]; then
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

PROMPT="The quick brown fox jumps over the lazy dog."
TOKENS=24
CTX=4096

OUT1=$(mktemp); OUT2=$(mktemp)
trap 'rm -f "$OUT1" "$OUT2"' EXIT

echo "=== run 1 ==="
"$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" -p "$PROMPT" -n "$TOKENS" --temp 0 > "$OUT1" 2>/dev/null
cat "$OUT1"
echo
echo "=== run 2 ==="
"$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" -p "$PROMPT" -n "$TOKENS" --temp 0 > "$OUT2" 2>/dev/null
cat "$OUT2"
echo
if diff -q "$OUT1" "$OUT2" >/dev/null; then
    echo "=== DETERMINISTIC: PASS ==="
    exit 0
else
    echo "=== DETERMINISTIC: FAIL (runs differ) ==="
    diff -u "$OUT1" "$OUT2" || true
    exit 1
fi
