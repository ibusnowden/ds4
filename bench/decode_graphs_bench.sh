#!/usr/bin/env bash
# Compare decode latency with vs without CUDA-graph capture of the per-token DAG.
set -uo pipefail
ROOT=/project/inniang/v4flash
NVRTC_DIR=/project/inniang/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
M=/dev/shm/ds4/model.gguf
P='Scrivi una storia lunga e dettagliata su una papera con tante avventure nel bosco.'

echo "=== baseline (no graphs) ==="
./ds4 --backend cuda --model "$M" --ctx 4096 --nothink -n 96 --temp 0 -p "$P" \
    >/tmp/base.out 2>/tmp/base.err
grep -iE 'generation:|prefill:' /tmp/base.err | tail -1

echo "=== DS4_CUDA_GRAPHS=1 (decode DAG captured + replayed) ==="
DS4_CUDA_GRAPHS=1 ./ds4 --backend cuda --model "$M" --ctx 4096 --nothink -n 96 --temp 0 -p "$P" \
    >/tmp/graph.out 2>/tmp/graph.err
grep -iE 'generation:|prefill:|capture|graph|invalidat|error' /tmp/graph.err | tail -3

echo "=== output parity (greedy, should be identical) ==="
if diff -q /tmp/base.out /tmp/graph.out >/dev/null; then echo "OUTPUT IDENTICAL"; else
  echo "OUTPUT DIFFERS:"; echo "  base : $(tail -c 120 /tmp/base.out | tr '\n' ' ')"; echo "  graph: $(tail -c 120 /tmp/graph.out | tr '\n' ' ')"; fi
