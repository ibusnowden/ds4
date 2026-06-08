# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

## Optimization roadmap (CUDA backend)

UPDATE 2026-06-06: decode 8.5 t/s, prefill 9.6 t/s on 1× RTX 6000 Ada after
fixing two real bottlenecks (see BASELINE.md "2026-06-06 optimization pass"):
the routed-MoE kernels ran one thread per output (now block-cooperative), and
EXPERT_HOT had a pass-2 bug that left experts host-mapped (now fixed and
on by default).  The MoE is bandwidth-bound on expert reads, NOT launch
overhead — the old "EXPERT_HOT 0× gain" was the bug, not evidence.  The next
decisive lever is multi-GPU full expert residency.

Historical (pre-fix) baseline (RTX 6000 Ada, 1× GPU): ~3.5 t/s decode, ~3.9 t/s
chunked prefill.  The note below that blamed kernel-launch overhead and cited
"EXPERT_HOT=1 producing 0× gain" was WRONG — that 0× was the pass-2 promotion
bug, not a real measurement.

### What's already done

- **CUDA Graph capture-per-chunk** (Strategy A in CUDA_GRAPHS_DESIGN.md) is
  implemented and gated behind `DS4_CUDA_GRAPHS=1`.  `ds4_cuda_capture_begin`
  / `ds4_cuda_capture_end_launch` wrap the chunk body; `cuGraphExecUpdate`
  caches the exec across chunks with matching topology.  Async memcpy
  (`cuMemcpyHtoDAsync` / `cuMemcpyDtoDAsync`) is already wired in
  `ds4_cuda_tensor_write` / `ds4_cuda_tensor_copy` during stream capture.
- **Chunked prefill** (`cuda_graph_prefill_chunked_range`) is the default path
  for ctx ≥ 4096.  Batched kernels cover all stateless matvec/norm/RoPE/MoE
  operations.  The per-token sequential chain (KV push, compressor, indexer,
  attention) remains the cap at ~1.1× over per-token.
- **Graph bench log** shows graphs=0 vs graphs=1 within noise for all prompt
  lengths.  The async memcpy conversion is done; the remaining launch overhead
  is inside the per-token sequential chain, which is not captured (it runs
  inside the `for (t = 0; t < M; t++)` loop that calls per-token kernels).

### Recommended order (revised)

1. **Profile with Nsight Systems/Compute** and produce an actual per-token
   kernel count/time table.  The current bottleneck model is inferred from
   aggregate t/s; a profile will confirm or refute it and reveal which kernels
   dominate.

2. **Fuse compressor emit path** — pool + RMS + RoPE + optional FP8 + push.
   The compressor chain (ds4.c:16357-16370) launches 5 small kernels per emit.
   A fused kernel would cut this to 1 launch.  Same for the indexer compressor
   chain (ds4.c:16379-16391).

3. **Fuse indexer mask path** — scores + top-k + mask, ideally avoiding the
   separate score buffer.  The indexer chain (ds4.c:16420-16451) launches
   4-5 small kernels per token.  A fused kernel would cut this to 1-2.

4. **Replace memcpy-based KV state/state_shift with kernels or ring-buffer
   addressing**.  `push_raw_kv`, `push_comp_kv`, and `compressor_state_shift`
   are DtoD memcpy calls that move cache rows.  A ring-buffer approach or a
   small kernel that writes directly to the right location would eliminate
   these copies.

5. **Improve chunked prefill using real batched GEMM / Tensor Core paths**.
   The current batched kernels are grid-extended per-token matvecs, not true
   GEMMs.  For chunked prefill with M > 1, converting matvecs to GEMMs would
   let Tensor Cores participate.

6. **Revisit decode CUDA Graphs as graph variants** after control flow is
   reduced.  The current per-token sequential chain has token-dependent
   control flow (compressor emit every `ratio` tokens, `n_comp` growing,
   raw KV shifting after SWA cap, dynamic shared memory).  Once fusion
   reduces the number of kernels and ring-buffers eliminate copies, the
   remaining graph is simpler to capture.

7. **Multi-GPU expert sharding** — IMPLEMENTED 2026-06-08 as a peer-device
   overflow pool (`ds4_cuda_peer_init` + `cuda_hot_walk_add_peer`; overflow
   experts live in device-1 VRAM, read by device-0 kernels via peer access /
   unified addressing — no kernel changes).  Correctness verified (full 43/43
   residency, byte-identical greedy output).  **Payoff is gated by interconnect
   bandwidth**: on 2× RTX 6000 Ada (PCIe P2P 26.7 GB/s ≈ host 24.2 GB/s) it is a
   wash (9.26 vs 9.69 t/s); the win needs NVLink (H100, ~30× host PCIe).  See
   BASELINE.md "2026-06-08".  Opt out with `DS4_CUDA_NO_PEER=1`.

### Key insight (settled 2026-06-08)

Decode is **bandwidth-bound on routed-expert reads, not launch-bound.**  A
direct CUDA-graphs A/B settles it: capturing the whole per-token DAG
(`DS4_CUDA_GRAPHS=1`, zero per-launch latency) gives **no speedup** over
`=0` (9.27 vs 9.61 t/s on RTX 6000 Ada, identical output).  Eliminating all
launch overhead changes nothing, so the remaining per-token chain (compressor +
indexer + attention) is compute/bandwidth-limited — **further kernel fusion
cannot raise throughput**, and the emit/top-k fusions already in place are kept
only for code clarity, not speed.  The lever that *does* matter is expert read
bandwidth: VRAM residency (EXPERT_HOT) and, where the interconnect is fast
enough, multi-GPU peer residency (roadmap item 7).  See BASELINE.md "2026-06-08".
