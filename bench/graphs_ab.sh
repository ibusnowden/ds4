#!/usr/bin/env bash
# Settle the fusion question on H100: does eliminating per-launch overhead help
# decode throughput?  CUDA graphs capture the whole per-token DAG (zero launch
# latency) — if graphs=1 ties graphs=0, the per-token chain is compute/bandwidth
# bound and further kernel fusion cannot raise throughput.
exec > /project/inniang/v4flash/bench/_graphs_ab.txt 2>&1
set -u
ROOT=/project/inniang/v4flash
export LD_LIBRARY_PATH=/project/inniang/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$ROOT"
M=${DS4_MODEL:-/tmp/inniang/ds4/model.gguf}
P='Scrivi una storia lunga e dettagliata su una papera con tante avventure nel bosco.'
N=${N:-96}
CTX=${CTX:-4096}
# Realistic default residency (single-GPU expert-hot). Graphs capture the whole
# per-token DAG, so graphs=1 vs graphs=0 isolates per-launch overhead.
BASEENV=(DS4_CUDA_NO_PEER=1)

run() {  # label  env...
  local label="$1"; shift
  local out="$ROOT/bench/_gout_$label.txt" err="$ROOT/bench/_gerr_$label.txt"
  echo "### RUN $label :: $*"
  env "$@" ./ds4 --backend cuda --model "$M" --ctx "$CTX" --nothink -n "$N" --temp 0 -p "$P" >"$out" 2>"$err"
  echo "  rc=$?"
  grep -iE 'promoted|generation|prefill|capture|graph|invalidat' "$err" | sed 's/^/  | /'
}

echo "### host=$(hostname) N=$N"
run graphs0 "${BASEENV[@]}" DS4_CUDA_GRAPHS=0
run graphs1 "${BASEENV[@]}" DS4_CUDA_GRAPHS=1
echo "### parity graphs0 vs graphs1"
if diff -q bench/_gout_graphs0.txt bench/_gout_graphs1.txt >/dev/null; then echo "  IDENTICAL ✓"; else echo "  DIFFER ✗"; fi
echo "### GRAPHS_AB_DONE"
