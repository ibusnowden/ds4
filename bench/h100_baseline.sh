#!/usr/bin/env bash
# Baseline decode/prefill benchmark on whatever GPU(s) the job holds.
# Runs from the NFS gguf (no /dev/shm staging — node RAM is tight).
set -uo pipefail
ROOT=/project/inniang/v4flash
NVRTC_DIR=/project/inniang/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
M=${DS4_MODEL:-$ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}
P='Scrivi una storia lunga e dettagliata su una papera con tante avventure nel bosco.'
N=${N:-64}

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo "model=$M"
echo "--- build ---"
make ds4 2>&1 | tail -3
echo "--- decode bench (n=$N, temp 0, nothink) ---"
t0=$(date +%s.%N)
./ds4 --backend cuda --model "$M" --ctx 4096 --nothink -n "$N" --temp 0 -p "$P" \
    >/tmp/ds4_base.out 2>/tmp/ds4_base.err
rc=$?; t1=$(date +%s.%N)
echo "exit=$rc  wall=$(echo "$t1 - $t0" | bc)s"
grep -iE 'prefill|generation|t/s|tokens|expert|hot|promot|resident|host-mapped|VRAM|GiB|GB' /tmp/ds4_base.err | tail -30
echo "--- tail of generation ---"
tail -c 200 /tmp/ds4_base.out
