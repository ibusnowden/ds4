#!/usr/bin/env bash
# End-to-end: start the optimized ds4-server, then drive the real pi CLI
# (headless) against it on a tool-requiring coding task.  Run via srun on the
# allocation node (server + pi share 127.0.0.1).
set -uo pipefail
ROOT=/project/inniang/v4flash
NVRTC_DIR=/project/inniang/.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.bun/bin:$HOME/.local/bin:$PATH"
cd "$ROOT"

echo "=== start ds4-server (optimized, EXPERT_HOT default) ==="
pkill -9 -x ds4-server 2>/dev/null; sleep 1
./ds4-server --backend cuda --model /dev/shm/ds4/model.gguf --ctx 32768 \
    --host 127.0.0.1 --port 8000 > /tmp/ds4_server.log 2>&1 &
SRV=$!

echo "=== wait for /v1/models ==="
ready=0
for i in $(seq 1 90); do
    if curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; echo "READY after ~${i}0s"; break; fi
    sleep 10
done
if [ "$ready" != 1 ]; then echo "SERVER NOT READY"; tail -20 /tmp/ds4_server.log; kill $SRV 2>/dev/null; exit 1; fi

echo "=== run pi (headless, ds4 provider, read tool) ==="
# Compute nodes lack /usr/bin/node; pi is a Node script, so run it via bun
# (on shared /home), which is Node-compatible.
PI_CLI="$HOME/.bun/install/global/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
cd "$ROOT/bench/pi-test/ws"
PI_CODING_AGENT_DIR="$ROOT/bench/pi-test/cfg" \
timeout 600 bun "$PI_CLI" --provider ds4 --model deepseek-v4-flash \
    --system-prompt "You are a coding assistant. Use the available tools to inspect files before answering." \
    --tools read,ls -np -nc \
    -p "Read config.py in the current directory and tell me the value of PORT. Answer with just the number." 2>&1
PI_RC=$?
echo "=== pi exit=$PI_RC ==="

echo "=== server-side tool/bench log lines ==="
grep -iE "ds4-server: bench|TOOLS|tool_calls|finish=" /tmp/ds4_server.log | tail -10

kill -9 $SRV 2>/dev/null
echo "=== done ==="
