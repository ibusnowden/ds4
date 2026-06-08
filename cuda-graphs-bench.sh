#!/usr/bin/env bash
# Bench: sweep prompt lengths for chunked prefill with DS4_CUDA_GRAPHS=0/1.
# Each run does prefill + 1-token decode (--temp 0 -n 1).  Wallclock is the
# whole process (load + prefill + decode + teardown); the delta between
# graphs=0 and graphs=1 at the same prompt length isolates the graph win.

#SBATCH --job-name=ds4-cuda-graphs-bench
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cuda-graphs-bench-%j.out
#SBATCH --error=logs/cuda-graphs-bench-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/cuda-graphs-bench.sh" ]; then
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

CTX=8192
FILLER="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. The five boxing wizards jump quickly. How vexingly quick daft zebras jump. Sphinx of black quartz judge my vow. "
# Each filler is ~250 chars ~= ~55 tokens.  Build prompts at ~256 / ~1024 / ~3072 tokens.
P_SHORT=$(printf "%s" "${FILLER}")$(printf "%s" "${FILLER}")$(printf "%s" "${FILLER}")$(printf "%s" "${FILLER}")$(printf "%s" "${FILLER}")
P_MED=""; for _ in $(seq 1 20); do P_MED+="$FILLER"; done
P_LONG=""; for _ in $(seq 1 60); do P_LONG+="$FILLER"; done

declare -A PROMPTS=(
    [short]="$P_SHORT"
    [medium]="$P_MED"
    [long]="$P_LONG"
)
LENGTHS=(short medium long)

run_one() {
    local mode="$1"
    local label="$2"
    local prompt="$3"
    local t0 t1
    t0=$(date +%s.%N)
    set +e
    DS4_CUDA_GRAPHS="$mode" \
        "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
                    -p "$prompt" -n 1 --temp 0 >/dev/null 2>&1
    local rc=$?
    set -e
    t1=$(date +%s.%N)
    awk -v t0="$t0" -v t1="$t1" -v mode="$mode" -v label="$label" -v rc="$rc" \
        -v plen="${#prompt}" \
        'BEGIN { printf "graphs=%s len=%-6s chars=%-6d rc=%d elapsed=%.3fs\n", mode, label, plen, rc, t1 - t0 }'
}

# 3 reps per (mode, length) for noise reduction.  Interleave modes per length
# so any drift in cluster load affects both modes equally.
for label in "${LENGTHS[@]}"; do
    p="${PROMPTS[$label]}"
    echo "=== prompt: $label (${#p} chars) ==="
    for rep in 1 2 3; do
        echo "-- rep $rep --"
        run_one 0 "$label" "$p"
        run_one 1 "$label" "$p"
    done
    echo
done
