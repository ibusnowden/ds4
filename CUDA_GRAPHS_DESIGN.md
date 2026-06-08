# CUDA Graphs path — design

## Why

Chunked prefill currently sits at 1.11× over per-token (3.91 vs 3.53
t/s).  EXPERT_HOT=1 (32 GiB of routed-expert weights moved into VRAM)
gave 0× — confirming PCIe is **not** the bottleneck.  Per `BASELINE.md`,
the remaining ceiling is **per-launch driver latency**: each chunk
issues ~1300 kernels × 43 layers ≈ 56K driver calls; even at ~1 µs of
launch overhead per call, that's ~60 ms/chunk of pure latency before
compute.

CUDA Graphs collapses an entire captured DAG into one `cuGraphLaunch`.
Replay submits the whole graph in a single driver call — eliminating
per-launch latency for static-shape kernels.

Target: push the chunk's per-launch overhead from ~60 ms to ~1 ms,
which (back-of-envelope) takes 3.91 t/s → 6-8 t/s on long prefill.

## What's involved

### 1. Stream-capture readiness

Existing infrastructure (good news):

- All `cuLaunchKernel` calls already go through a single non-default
  stream (`g_cuda_stream`, created in `ds4_cuda_init`). ✓
- Worker thread CUDA context binding via
  `ds4_cuda_attach_thread` is already wired. ✓

What's NOT ready (blocker):

- `ds4_cuda_tensor_write` calls **synchronous** `cuMemcpyHtoD_v2`.
  Sync H→D from the same thread during stream capture is disallowed.
- `ds4_cuda_tensor_copy` calls **synchronous** `cuMemcpyDtoD_v2`.
  Same issue.
- Both must be replaced with `cuMemcpyHtoDAsync` and
  `cuMemcpyDtoDAsync` on `g_cuda_stream`, with a final sync only at
  chunk end.

### 2. Captured-vs-host control flow

The current `cuda_graph_eval_layer_batch` has host-side branches:

```c
if (L.ratio != 0) { compressor_proj_state ... }
if (should_compress) { pool/RoPE/FP8/push_comp/state_shift ... }
if (L.ratio == 4u && lc->n_index_comp > 0) { indexer mask ... }
```

Stream capture **records what executes** — these branches resolve at
capture time.  Two strategies:

**A. Capture-per-chunk-fresh**: capture once, launch once, throw away.
- Pro: simplest.  No conditional handling — each chunk's actual code
  path is captured.
- Pro: per-launch latency saved (the win).
- Con: capture overhead per chunk (~1-2 ms for ~1300 nodes).
- Con: no replay reuse.
- Net: still a big win (60 ms → ~3 ms per chunk).

**B. Capture-once / replay-many** with parameter updates between
replays via `cuGraphExecKernelNodeSetParams`.
- Requires: chunk's emit pattern to be identical across replays.
- Setup: pin `M = lcm(ratios) = 128` (or smaller for memory). Align
  chunks to 128-token boundaries so emit pattern is fixed: ratio=4
  emits at relative positions 3,7,...,127; ratio=128 emits at 127.
- Per-replay updates: just `pos0`.  Move `pos0` into device memory
  (`g->batch_param_pos0_d`); kernels that need it (RoPE, compressor)
  read from there.  Capture sees a fixed pointer; updates via
  `cuMemcpyHtoDAsync`.
- Pro: capture once at session start, replay forever.
- Con: every kernel that takes `pos`, `n_raw`, `n_comp` as kernel arg
  must be rewritten to take a device pointer.  Substantial.
- Con: handling of variable-size shared memory in attention (depends
  on n_total = n_raw + n_comp, which grows). Either fix shared memory
  size to a max or capture per-token-in-chunk.

### 3. Recommended first cut: Strategy A

Strategy A wins enough (60 ms → ~3 ms) without touching kernel
signatures.  It's the right first step.

**Steps**:

1. **Replace sync memcpy with async on g_cuda_stream**
   - Add `cuMemcpyHtoDAsync`, `cuMemcpyDtoDAsync` symbols to
     `ds4_cuda.c`.
   - Either:
     - (a) replace sync calls outright, with a final sync at the
       end of any operation that previously relied on the implicit
       sync; OR
     - (b) keep sync versions but add `_async` siblings the chunked
       path uses.  Less risky.
   - Verify the existing single-token decode still works.

2. **Wrap layer batch in stream capture**
   - In `cuda_graph_prefill_chunked_range`, around the per-chunk
     work:
     ```c
     CUgraph graph = NULL; CUgraphExec exec = NULL;
     cuStreamBeginCapture(g->stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
     // run all the layer_batch + output_head_at logic
     cuStreamEndCapture(g->stream, &graph);
     cuGraphInstantiate(&exec, graph, NULL, NULL, 0);
     cuGraphLaunch(exec, g->stream);
     cuStreamSynchronize(g->stream);
     cuGraphExecDestroy(exec);
     cuGraphDestroy(graph);
     ```
   - Gate behind `DS4_CUDA_GRAPHS=1` initially.

3. **Validate parity** — the existing
   `--cuda-chunked-prefill-test` should pass identically (same
   kernels, same order; just batched into one launch).

4. **Bench** — re-run `chunked-vs-pertoken-bench` with
   `DS4_CUDA_GRAPHS=1`; expect 3.9 t/s → 6-10 t/s on long prefill.

5. **Measure capture overhead** — if dominant, move toward Strategy B.

### 4. New CUDA driver symbols needed

```
cuStreamBeginCapture       → stream capture start
cuStreamEndCapture         → stream capture end → CUgraph
cuStreamIsCapturing        → debug: confirm in-capture state
cuGraphInstantiate         → CUgraph → CUgraphExec
cuGraphLaunch              → submit CUgraphExec
cuGraphExecDestroy         → free CUgraphExec
cuGraphDestroy             → free CUgraph
cuMemcpyHtoDAsync          → async H→D on stream
cuMemcpyDtoDAsync          → async D→D on stream
```

All available since CUDA 10.0.  We dlopen libcuda.so already; just add
symbol loads.

### 5. Edge cases

- **Capture failure on certain operations**: stream capture has a list
  of disallowed operations (host-blocking memcpy, certain context
  ops).  If we hit one mid-capture, capture gets invalidated and the
  end call returns `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`.  Solution:
  carefully audit which operations happen during the captured region.
  Pre-chunk setup (token upload, embed_token launch) is fine if all
  use async stream ops.

- **First-chunk capture overhead**: with capture-per-chunk, capture is
  ~1-2 ms.  For 2-chunk prefill at 950 tokens, that's 4 ms total.
  Negligible.

- **Memory**: each `CUgraphExec` allocates internal buffers.  For
  per-chunk graphs, destroy after launch.  No leak risk.

- **Stream capture mode**:
  - `THREAD_LOCAL` — only this thread's stream ops captured.  Safest.
  - `GLOBAL` — all streams captured.  Risky if other threads use CUDA.
  - We use `THREAD_LOCAL`.

### 6. Test plan

1. Unit-level: `--cuda-batch-kernel-test` — kernel-level parity, no
   chunked path.  Should still pass (no change to kernels).

2. Function-level: `--cuda-chunked-prefill-test --prompt "Hi"` (M=1
   bit-exact).  Then 4-token, then 19-token.  All with
   `DS4_CUDA_GRAPHS=1`.  Expected: same parity as without
   `DS4_CUDA_GRAPHS=1` since the captured kernel sequence is identical.

3. End-to-end: `chunked-vs-pertoken-bench.sh` with
   `DS4_CUDA_GRAPHS=1` for the chunked run, comparing against the
   existing chunked baseline.  Expected: t/s improvement.

### 7. Rollout

Behind env flag `DS4_CUDA_GRAPHS=1` initially.  Keep the
non-graph path as default until parity + bench confirm safety.

After parity is proven over a wider battery (all tests in
`tests/`), promote to default with `DS4_CUDA_NO_GRAPHS=1` for opt-out.

## Open questions

- Will `cuMemHostRegister`-mapped weights work with stream capture?
  These reads happen inside kernels, not as standalone memcpy —
  should be fine.  Verify on first run.

- Does NVRTC kernel debug logging during capture cause issues?  If
  so, gate `DS4_CUDA_DEBUG_SYNC=1` off during capture.

- Is `cuStreamSynchronize` allowed inside a captured region?  No —
  capturing a sync invalidates capture.  Need to remove all sync
  points from `cuda_graph_eval_layer_batch` and `output_head_at`
  during capture (the end-of-token sync happens after capture ends).
