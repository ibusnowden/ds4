#!/usr/bin/env bash
# Minimal SLURM smoke test that actually exercises cuda_graph_eval_token by
# running a short prefill+decode through the CLI on the RTX node.
#
# Submit with: sbatch ./serve-slurm-rtx-decode.sh

#SBATCH --job-name=ds4-cuda-decode-smoke
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/cuda-decode-smoke-%j.out
#SBATCH --error=logs/cuda-decode-smoke-%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -n "${DS4_ROOT:-}" ]; then
    ROOT="$DS4_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/serve-slurm-rtx-decode.sh" ]; then
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

CTX="${DS4_CTX:-4096}"
PROMPT="${DS4_PROMPT:-Hello}"
TOKENS="${DS4_PREDICT:-8}"

echo "[$(date -Iseconds)] running CUDA decode smoke: ctx=$CTX tokens=$TOKENS prompt=\"$PROMPT\""
set +e
DS4_DECODE_PROFILE_DETAIL=1 \
    "$ROOT/ds4" --backend cuda --model "$ROOT/ds4flash.gguf" \
                --ctx "$CTX" -p "$PROMPT" -n "$TOKENS" --temp 0
rc=$?
set -e
echo "[$(date -Iseconds)] decode smoke exit code: $rc"
exit "$rc"
