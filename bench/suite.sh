#!/usr/bin/env bash
# Build + parity + benchmark suite for the multi-GPU expert-residency work.
# Logs everything to NFS files under bench/ so the driver can Read them live.
exec > /project/inniang/v4flash/bench/_suite.txt 2>&1
set -u
ROOT=/project/inniang/v4flash
export LD_LIBRARY_PATH=/project/inniang/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$ROOT"
M=${DS4_MODEL:-/tmp/inniang/ds4/model.gguf}
P='Scrivi una storia lunga e dettagliata su una papera con tante avventure nel bosco.'
N=${N:-64}
CTX=${CTX:-4096}

echo "### host=$(hostname) date=$(date +%T) gpus=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "### model=$M size=$(stat -c %s "$M" 2>/dev/null)"
echo "### build"
make ds4 2>&1 | tail -4
[ -x ./ds4 ] || { echo "BUILD FAILED"; echo "### SUITE_DONE"; exit 1; }

run() {  # label  env...
  local label="$1"; shift
  local out="$ROOT/bench/_out_$label.txt"
  local err="$ROOT/bench/_err_$label.txt"
  echo "### RUN $label  :: $*"
  local t0 t1
  t0=$(date +%s.%N)
  env "$@" ./ds4 --backend cuda --model "$M" --ctx "$CTX" --nothink -n "$N" --temp 0 -p "$P" >"$out" 2>"$err"
  local rc=$?
  t1=$(date +%s.%N)
  echo "  rc=$rc wall=$(echo "$t1-$t0" | bc)s"
  grep -iE 'promoted|peer|generation|prefill|t/s|host-mapped|error|fail' "$err" | sed 's/^/  | /'
}

# Pick which configs to run via CONFIGS env (default: all). Each value below.
CONFIGS="${CONFIGS:-dev0_full dev0_cap20 peer_cap20}"
for c in $CONFIGS; do
  case "$c" in
    dev0_full)  run dev0_full  DS4_CUDA_NO_PEER=1 ;;
    dev0_cap20) run dev0_cap20 DS4_CUDA_NO_PEER=1 DS4_CUDA_EXPERT_BUDGET_GIB=20 ;;
    peer_cap20) run peer_cap20 DS4_CUDA_EXPERT_BUDGET_GIB=20 ;;
    peer_full)  run peer_full ;;
  esac
done

echo "### parity (greedy temp0 outputs must be byte-identical across configs)"
ref=""
for c in $CONFIGS; do
  f="$ROOT/bench/_out_$c.txt"
  [ -s "$f" ] || { echo "  $c: (no output)"; continue; }
  if [ -z "$ref" ]; then ref="$f"; echo "  ref=$c"; continue; fi
  if diff -q "$ref" "$f" >/dev/null; then echo "  $c == ref  ✓"; else echo "  $c != ref  ✗"; fi
done
echo "### ref tail:"; tail -c 180 "$ref" 2>/dev/null
echo
echo "### SUITE_DONE"
