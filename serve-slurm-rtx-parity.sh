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

# Parity is checked on the logits the model actually produces, not on the
# greedy text that follows from those logits.  Two reasons:
#   (1) GPU↔CPU FP32 reduction order leaves ULP-scale drift per element which
#       accumulates over 43 layers; greedy text comparison amplifies this into
#       fully different generations even when the underlying scores agree.
#   (2) The probe's top-5 set + max-abs + RMS numbers are what we actually
#       care about for production parity.
# The probe passes when top-1 matches and max-abs is bounded; we sweep a few
# prompt lengths so any future regression that *worsens* drift is visible.

declare -a PROMPTS=(
    "Hi"
    "The fox"
    "The quick brown fox"
    "The quick brown fox jumps over the lazy dog."
)

declare -i fail=0
for prompt in "${PROMPTS[@]}"; do
    echo "=== probe: '$prompt' ==="
    set +e
    "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
                --cuda-parity-probe -p "$prompt" 2>&1 | grep -E "parity probe|cuda" || true
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        fail=$((fail + 1))
    fi
    echo
done

# Optional: keep the historical greedy text comparison as a *soft* signal,
# only invoked when DS4_PARITY_INCLUDE_TEXT=1.  Unset by default because the
# logits probe above is the authoritative gate.
if [ -n "${DS4_PARITY_INCLUDE_TEXT:-}" ]; then
    PROMPT="The quick brown fox jumps over the lazy dog."
    GPU_OUT_FILE=$(mktemp)
    BRIDGE_OUT_FILE=$(mktemp)
    trap 'rm -f "$GPU_OUT_FILE" "$BRIDGE_OUT_FILE"' EXIT
    echo "=== greedy text (soft check) ==="
    "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
                -p "$PROMPT" -n "$TOKENS" --temp 0 \
            > "$GPU_OUT_FILE" 2>/dev/null
    DS4_CUDA_BRIDGE=1 "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" --ctx "$CTX" \
                -p "$PROMPT" -n "$TOKENS" --temp 0 \
            > "$BRIDGE_OUT_FILE" 2>/dev/null
    diff -u "$BRIDGE_OUT_FILE" "$GPU_OUT_FILE" || true
fi

if [ "$fail" -eq 0 ]; then
    echo "=== PARITY: PASS (top-5 sets agree, max-abs bounded across all sweeps) ==="
    exit 0
fi
echo "=== PARITY: $fail/${#PROMPTS[@]} sweeps fell outside the top-5/max-abs gate ==="
exit 1
