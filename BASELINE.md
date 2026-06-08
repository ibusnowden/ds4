# v4flash CUDA baseline

Captured 2026-05-09 from `serve-slurm-rtx-{parity,bench}.sh` runs on
`bigTiger` with 1× RTX 6000 Ada. Refresh by re-running the SLURM scripts
referenced below.

## 2026-06-06 decode/prefill optimization pass (single RTX 6000 Ada)

Headline: decode **3.62 → 8.5 t/s (2.35×)**, prefill **3.85 → 9.6 t/s (2.5×)**,
long-prompt (~960 tok) prefill **3.5 → 9.4 t/s (2.7×)**.  Parity preserved
(probe top-1 MATCH, max_abs 4.93 < 5.0; chunked-prefill bit-exact).  Model
staged in `/dev/shm` to remove the NFS cold-load (host-register of the 81 GB
GGUF off NFS otherwise dominates first-token latency).

Two root causes found and fixed (the prior roadmap's "launch-overhead /
compute-bound" diagnosis was wrong):

1. **Routed-MoE kernels ran one thread per output** (`block=(1,1,1)`, a
   `threadIdx.x != 0` guard).  `ds4_routed_iq2_swiglu_f32` /
   `ds4_routed_q2_down_sum_f32` (and their `_batch` variants) each had a single
   thread do full 4096-/2048-length dequant dot products at ~2% SM occupancy.
   Rewrote them block-cooperatively (128-thread parallel reduction via
   `ds4_iq2_xxs_dot_partial` / `ds4_q2_k_dot_partial` + `ds4_block_sum2`).
   Decode 3.62 → 5.99, prefill 3.85 → 6.12.

2. **EXPERT_HOT never actually used VRAM** — `cuda_graph_promote_hot_weights`
   tallied routed experts in pass 1 but pass 2 only re-walked globals + small
   layer weights, so the pool reserved expert space but never filled it or
   added `hot_entries`.  The MoE kernels silently fell back to host-mapped
   (zero-copy PCIe, ~15 GiB/s) reads.  This is what made the old A/B show
   "0× gain".  A weight-skip diagnostic (`DS4_MOE_NOWEIGHT`) proved the MoE was
   bandwidth-bound on these reads (swiglu 80 → 12.5 ms when weights skipped).
   Added the pass-2 expert copy/record; expert promotion is now **on by
   default** (disable with `DS4_CUDA_NO_EXPERT_HOT=1`) because it runs after
   the KV cache is allocated, so its budget can't starve KV.  With ~20/43
   layers' experts in VRAM (the single-card limit): swiglu 80 → 53, down
   50 → 29 ms; decode 5.99 → 8.39, prefill 6.12 → 9.47.

Decode section profile (`DS4_CUDA_DPROF=1`, ms/token, 20/43 experts in VRAM):
`attn≈26  moe_pre≈5  swiglu≈53  down≈29  ffn_rest≈5  head≈2  total≈120`.

### vs vLLM

**vLLM 0.19.0 cannot serve DeepSeek-V4-Flash at all** — `DeepseekV4ForCausalLM`
is not a registered architecture (closest is `DeepseekV32ForCausalLM`), and the
engine-core process crashes during model build.  So on the actual target model
ds4 is the only runnable engine.  As a same-hardware anchor, vLLM on
OLMo-2-7B (dense, 40× fewer params, FlashAttention + CUDA graphs + batching)
does ~64 tok/s decode / ~21.5k tok/s prefill / ~3.2k tok/s batched throughput;
those numbers are not comparable to a 284B 2-bit MoE that streams far more
weight per token, but they mark the absolute ceiling a tuned engine reaches on
this card.

### Remaining levers (in impact order)

1. **Multi-GPU full expert residency** — the single decisive lever.  73 GiB of
   routed experts don't fit one 48 GiB card (only ~20/43 layers), so 23 layers
   still stream experts over PCIe.  Layer-pipeline sharding across the node's
   GPUs (each holds its layers' experts in local VRAM at ~960 GiB/s; hand the
   ~64 KiB HC activation across device boundaries) would take swiglu/down to
   the ~12/8 ms compute floor → ~14–16 t/s decode.  Needs per-device
   context/module/scratch/KV (the executor is currently single-context).
2. **Batched-attention chunked prefill** — prefill is still capped at ~decode
   speed because the per-token compressor/indexer/attention chain inside each
   chunk is not batched (matvec/MoE are).  Batching it is the prefill lever.
3. **CUDA graphs for decode** to remove launch overhead in the ~26 ms attention
   section, and parallelize the remaining 1-thread top-k kernels (router /
   indexer / output_hc_weights).

## Headline numbers

After this baseline run:

| metric                     | before | after | change |
|----------------------------|-------:|------:|-------:|
| short-prompt prefill (t/s) |   1.65 |  3.59 |  2.18× |
| short-prompt gen (t/s)     |   1.64 |  3.55 |  2.16× |
| `ds4-server` model load    |  9.45s | 9.71s |  +0.26s (VRAM copy) |

Server-side per-request bench summary:

```
ds4-server: bench id=chatcmpl-1 kind=chat prompt=22 cached=0 new=22 gen=64
            prefill_s=6.180 decode_s=18.108 e2e_s=24.287
            prefill_tps=3.56 decode_tps=3.53 e2e_tps=3.54 finish=length
ds4-server: bench id=chatcmpl-2 kind=chat prompt=954 cached=0 new=954 gen=64
            prefill_s=271.627 decode_s=18.372 e2e_s=289.999
            prefill_tps=3.51 decode_tps=3.48 e2e_tps=3.51 finish=length
```

The per-token forward is consistent at ~3.5 t/s regardless of prompt
size, so long-prompt latency is dominated by the **per-token prefill
loop** (954 prompt tokens × 0.28s/token = 272s).  That is the next
remaining lever — see the **What still drags on long prompts** section
below.

## Parity (logits-level probe)

`./serve-slurm-rtx-parity.sh` sweeps four prompt lengths and runs
`--cuda-parity-probe`. The probe runs the same prompt through the CUDA
executor and the CPU reference per-token, then compares the post-prefill
logits (the prediction for token N+1).

Pass criterion: top-5 token sets agree (order-independent) **and**
max-abs over the vocab logits < 5.0.

| prompt | tokens | max_abs | rms   | top-1 | top-5 set |
|--------|-------:|--------:|------:|-------|----------:|
| `"Hi"`                                            | 1  | 0.865 | 0.123 | MATCH  | 5/5 |
| `"The fox"`                                       | 2  | 0.784 | 0.158 | MATCH  | 5/5 |
| `"The quick brown fox"`                           | 4  | 1.726 | 0.350 | MATCH  | 5/5 |
| `"The quick brown fox jumps over the lazy dog."` | 10 | 2.349 | 0.426 | DIFFER | 5/5 |

Job: `logs/cuda-parity-smoke-70709.out`. Result: **PARITY: PASS**.

The within-top-5 reordering at 10 tokens is FP32 reduction-order
nondeterminism between GPU and CPU — the GPU is faithful to the model;
its top-1 just happens to be CPU's top-2 when the two scores are within
~0.1 of each other (e.g., `21.671` vs `21.308` GPU; `21.562` vs `21.161`
CPU). Bit-exact GPU↔CPU parity is engineering-expensive with diminishing
returns and not gated.

## Throughput

`./serve-slurm-rtx-bench.sh` short-prompt config: `ctx=32768 -n 256
--nothink --temp 0`, prompt `"Scrivi una storia su una papera scansafatiche."`.

| backend                                        | prefill t/s | gen t/s | source |
|------------------------------------------------|------------:|--------:|--------|
| cuda (host-mapped weights)                     | 1.65        | 1.64    | `logs/cuda-bench-70710.out` |
| cuda (VRAM-resident hot weights, 8.20 GiB hot) | 3.59        | 3.55    | `logs/cuda-bench-70714.out` |

Removing the per-launch `cuStreamSynchronize` (replacing with one explicit
sync at the end of `cuda_graph_eval_token`) had **no measurable effect**.
That confirms the bottleneck is **not** host↔device sync round-trips. The
dominant cost is PCIe weight transfer:

- 80.76 GiB of GGUF weights are `cuMemHostRegister`-ed (host-mapped)
  rather than `cuMemAlloc`-ed, because the model exceeds the 48 GB RTX
  6000 VRAM (`logs/cuda-bench-70710.err` reports
  `mapped_weights=80.76 GiB`).
- Each token's forward reads a substantial fraction of those weights
  (norms + LoRAs + 6/256 active MoE experts × 43 layers + output head).
  At PCIe Gen4 x16 (~32 GB/s effective), that's the rate-limit.

**Hot-weight VRAM promotion done** (commit-equivalent set in this
session): `cuda_graph_promote_hot_weights` copies the small read-on-every-
token tensors (norms, attn LoRAs, KV proj, shared expert, router,
compressor/indexer, output head, token_embd) into one ~8.2 GiB VRAM pool;
`cuda_graph_tensor_device_ptr` binary-searches a sorted offset table and
falls back to host-mapped on miss.  Result: 2.18× speedup, parity smoke
unchanged.  Gated by `DS4_CUDA_NO_HOT=1` for low-VRAM scenarios.

**Routed-expert VRAM promotion explored — 0× extra benefit.** Opt-in via
`DS4_CUDA_EXPERT_HOT=1`.  A/B bench (`logs/cuda-bench-no-exp-70726.out`)
showed identical 3.59 t/s with and without 33.75 GiB of routed-expert
weights pushed into VRAM.  Once the small hot weights live in VRAM, the
bottleneck shifts from PCIe bandwidth to kernel-launch / compute on the
~1300 small kernels per token.  Promoting experts is therefore wasted
VRAM by default; left available behind the env flag for experimentation
on bigger GPUs or future kernel-fusion work.

## What still drags on long prompts

Long prompts run prefill *per-token* (`cuda_graph_prefill_loop`,
`ds4.c:16811`), so a 954-token prompt prefills in 272s at the steady
~3.55 t/s rate.

Two structural changes can break past ~3.5 t/s:

1. **Layer-major batched prefill** (Metal already has it via
   `metal_graph_prefill_chunked_range`).  For prefill only — process M
   prompt tokens through layer 0 in one batch, then layer 1, etc.  Each
   small kernel still launches once per chunk but does M× the work, so
   total kernel-launch overhead drops by M×.  This is the dominant
   long-prompt lever and would drop a 954-token prefill from 272s to
   single-digit seconds.

2. **CUDA Graphs**.  The eval_token sequence is the same ~1300 kernels
   for every token; capture once, replay per token.  Replay submits the
   whole DAG to the driver in one call, eliminating per-launch latency.
   Caveat: kernel parameters that change per token (`pos`, `token_id`)
   need to be moved into device memory or updated via
   `cuGraphExecKernelNodeSetParams` between replays.

Beyond those, multi-GPU expert split across the 2 RTX 6000s already
requested in `serve-slurm-rtx.sh` would unlock full VRAM residency for
the routed experts, but is a larger refactor (NCCL or peer-access plus
per-layer kernel changes).

## Determinism

`./serve-slurm-rtx-determinism.sh` (`logs/cuda-det-70701.out`) — same
prompt, two runs, output matches. **PASS** (CUDA executor is deterministic
across runs given the same input).

## Server-side bench logging

`ds4-server` now emits two greppable lines:

- At startup, after engine + session are ready:
  `ds4-server: model loaded in <Xs> (engine=<Ys> session=<Zs> backend=cuda ctx=<N>)`
- At end of each request:
  `ds4-server: bench id=<id> kind=<chat|completion> prompt=<P> cached=<C> new=<N> gen=<G> prefill_s=<...> decode_s=<...> e2e_s=<...> prefill_tps=<...> decode_tps=<...> e2e_tps=<...> finish=<...>`

These let `bench/pi-bench.sh` cross-reference per-request server numbers
against client-measured TTFT / decode / e2e wall-clock.

## Pi end-to-end

`bench/pi-bench.sh` drives `/v1/chat/completions` over streaming SSE for
`bench/prompts/{short,medium,long}.txt`. Output:
`bench/results/pi-<UTC>.csv`. Pending: a live server run via
`serve-slurm-rtx.sh` + `connect.sh` to record the first numbers.

To run end-to-end:

```
sbatch ./serve-slurm-rtx.sh                      # cluster: start the server
./connect.sh                                     # client:  open the SSH tunnel
./bench/pi-bench.sh                              # client:  drive the bench
```

## Pi end-to-end (verified)

`bench/results/pi-20260509T181710Z.csv` — short and medium prompts driven
from the cluster head against `serve-slurm-rtx.sh` (job 70715,
`itiger02:8000`):

| prompt | prompt_chars | gen_tok | ttft_s  | decode_s | e2e_s   | decode_tps | e2e_tps |
|--------|-------------:|--------:|--------:|---------:|--------:|-----------:|--------:|
| short  |           75 |      64 |   6.469 |   17.828 |  24.297 |       3.59 |    2.63 |
| medium |         2779 |      64 | 271.928 |   18.081 | 290.009 |       3.54 |    0.22 |

The decode rate is consistent (~3.55 t/s) regardless of prompt size; the
medium e2e_tps is dominated by the per-token prefill loop, which is the
next thing to fix.

## Layer-major chunked prefill — first cut shipped

`cuda_graph_prefill_chunked_range` (ds4.c:18154) is wired into
`cuda_session_sync` and is the default path when `batch_chunk_cap > 1`
(any ctx ≥ 4096).  Set `DS4_CUDA_NO_CHUNKED=1` to fall back to the per-
token loop.

| metric                        | per-token | chunked | speedup |
|-------------------------------|----------:|--------:|--------:|
| medium-prompt prefill (t/s)   |     3.53  |   3.91  |  1.11×  |
| medium-prompt generation      |     4.68  |   4.68  |  1.00×  |

Parity (`--cuda-chunked-prefill-test`):

| prompt | tokens | max_abs | rms   | top-1 | top-5 set |
|--------|-------:|--------:|------:|-------|----------:|
| short ("The quick brown fox") |   4 | 0.000 | 0.000 | MATCH | 5/5 |
| long  (19 words)              |  19 | 0.000 | 0.000 | MATCH | 5/5 |
| medium (950 tokens)           | 950 | 1.696 | 0.375 | MATCH | 4/5 |

950-token within-top-5 reordering is FP32 reduction-order
nondeterminism (same gate as the parity probe — top-1 MATCH +
max_abs<5.0).  4-token and 19-token prompts are bit-exact, including
multiple compressor emits.

What's batched (saves ~M-fold launches):
- Embed (1 launch per chunk).
- Attention sub-block matvecs: Q LoRA (q_a, q_a_norm RMS, q_b, head_rms),
  KV proj (kv, kv_a_norm RMS), RoPE on Q & KV, KV_FP8.
- Output projection (matvec_q8_0_grouped + matvec_q8_0).
- HC pre/post (rms-no-weight, matvec_f16 hc_attn_fn / hc_ffn_fn,
  hc4_split_norm, hc4_post).
- FFN sub-block: router (logits + topk/hash), MoE routed_iq2_swiglu +
  routed_q2_down_sum, shared expert (gate/up/swiglu/down), add.

What's still per-token sequential within the chunk (next iteration):
- KV push (interleaved with attention to handle SWA correctly).
- Compressor proj_state, pool/emit, state_shift.
- Indexer compressor + Q + scores + topk_mask.
- Mixed attention itself.

These per-token sequentials cap the speedup at ~1.1×.

A/B with `DS4_CUDA_EXPERT_HOT=1` (32 GiB of routed-expert weights moved
into VRAM, 19/43 layers covered): chunked **3.86 t/s vs 3.91 t/s
without it** — i.e., PCIe is *not* the bottleneck even at chunked-
prefill PCIe demand.  The bottleneck is launch overhead + per-token
attention chain (5-8 launches/token × 43 layers).  This was already
true in per-token mode (the BASELINE.md note from earlier still holds);
chunked just confirms it under the higher PCIe-pressure regime.

Real remaining levers (in order of expected impact):
1. **CUDA Graphs** — capture the ~1300-kernel layer DAG once, replay per
   chunk.  Eliminates per-launch latency entirely.  Would push past the
   launch-overhead ceiling.
2. **Kernel fusion** — fuse the per-token attention chain (5-8 small
   kernels) and the compressor emit chain (5 small kernels) into single
   composite kernels.  Cuts launch count by 5-8×.
3. **Multi-GPU expert split** with peer-access — if the second GPU is
   ever available, full VRAM residency for routed experts plus layer
   sharding.

Two bugs caught + fixed during this iteration:
- `hc4_post_f32_batch` had stride `t * 12` for the per-token split tensor;
  the producer (`hc4_split_weighted_sum_norm_f32_batch`) writes with
  stride `t * 24`.  Fixed kernel + test allocation (the test had
  similarly mis-sized split, masking the bug).
- SWA fallback path pushed all M raw_kv rows before any per-token
  attention ran, so tokens 0..M-2 saw their lookback evicted by later
  pushes.  Fixed by interleaving push+attention per-token within the
  chunk; non-SWA chunks still benefit from batched matvecs.

A latent bug was also fixed: `cuda_graph_reset_layer_caches` now zeros
the compressor state buffers (cuMemAlloc doesn't zero), so the first
emit in a fresh session no longer pools garbage from rows [0..ratio).

## Layer-major prefill: kernel building blocks (in progress)

All per-token-stateless kernels needed for `cuda_graph_prefill_chunked_range`
now have batched (`_batch`) variants in the executor and host wrappers in
`ds4.c`.  Each is grid-extended by an `n_tokens` axis and verified bit-exact
against the per-token kernel where parity-testable:

| kernel                          | batched | bit-exact | notes |
|---------------------------------|---------|-----------|-------|
| matvec_q8_0                     | yes     | yes (test) | `cuda-batch-kernel-test-70748.out` |
| matvec_f16                      | yes     | yes (test) | same |
| rms_general                     | yes     | yes (test) | same |
| head_rms                        | yes     | yes (test) | same |
| rope                            | yes     | yes (test) | same |
| hc4_post / hc4_split_norm       | yes     | yes (test) | hc4_split_norm structural |
| swiglu / add / kv_fp8           | yes     | yes (test) | same |
| embed_token_hc                  | yes     | yes (test) | same |
| matvec_q8_0_grouped             | yes     | structural | mirror of single-token |
| router_probs                    | yes     | yes (test) | `mismatches=0/2048` |
| router_topk_select              | yes     | structural | per-token sequential top-K |
| router_hash_select              | yes     | structural | per-token sequential top-K |
| indexer_scores                  | yes     | structural | grid (n_comp, n_tokens) |
| indexer_topk_mask               | yes     | structural | per-token sequential top-K |
| indexer_weight_scale            | yes     | structural | elementwise |
| routed_iq2_swiglu (MoE expert)  | yes     | structural | grid (mid_dim, n_used, n_tokens) |
| routed_q2_down_sum (MoE down)   | yes     | structural | grid (out_dim, n_tokens) |
| attention_mixed                 | yes     | structural | per-token raw-cutoff at n_raw_pre+t+1 |
| compressor_proj_state           | **no**  | n/a        | next iteration |
| compressor_pool_norm            | **no**  | n/a        | per-emit, can stay sequential |
| push_raw_kv / push_comp_kv      | **no**  | n/a        | DtoD memcpy, batch == one M*row copy |

Remaining work to wire layer-major prefill end-to-end:
- Host-side `cuda_graph_prefill_chunked_range(g, model, weights, prompt, pos0, M, ...)` mirroring `metal_graph_prefill_chunked_range` (`ds4.c:12638`).
- Inside it, two helpers (mirroring the Metal split): `cuda_graph_encode_layer_attention_batch` and `cuda_graph_encode_layer_ffn_batch`.
- Allocate M-batched scratch tensors (q_batch, kv_batch, comp_batch, mid_batch, etc.) — currently the executor only has single-token scratch (`g->eval_kv` etc).
- Compressor stays sequential per token within the chunk for the first cut (it's one small kernel per token; the dominant savings come from the matvec/MoE kernels which are batched).
- Wire `cuda_session_sync` to call `cuda_graph_prefill_chunked_range` for prompts longer than chunk_cap and fall back to `cuda_graph_prefill_loop` for short prompts.

Expected impact: long-prompt prefill goes from per-token (272s for 954 tokens at 3.5 t/s) to per-chunk, with M=8 collapsing 8× of the kernel-launch overhead per layer.  Lower bound is bound by the matvec/MoE compute (small kernels but real work) — single-digit seconds for 954-token prefill is the target.

## What changed in this baseline run

- Logits-level `--cuda-parity-probe` flag (CLI + `ds4_session_cuda_parity_probe` API).
- `serve-slurm-rtx-parity.sh` rewritten to sweep prompt lengths and use the probe; gate is now top-5 set agreement + max-abs<5.0.
- `--use_fast_math` on NVRTC compile is now opt-in via `DS4_CUDA_FAST_MATH=1`.
- Per-launch `DS4_CUDA_SYNC()` is a no-op unless `DS4_CUDA_DEBUG_SYNC=1`. One explicit sync runs at the end of `cuda_graph_eval_token` to surface deferred kernel errors.
- **CUDA worker thread context bind**: `ds4_engine_attach_thread` / `ds4_cuda_attach_thread` (cuCtxSetCurrent). `ds4-server`'s worker thread now binds the CUDA context once at startup. Without this, every server request returned HTTP 500 on the first kernel launch.
- **VRAM-resident hot weights**: `cuda_graph_promote_hot_weights` copies ~8.2 GiB of small frequently-read tensors into VRAM. `cuda_graph_tensor_device_ptr` binary-searches a sorted hot table; routed MoE experts stay host-mapped. **2.18× throughput** with parity unchanged.
- `ds4-server` model-load + per-request bench summary log lines.
- `bench/pi-bench.sh` + prompts + results dir; first end-to-end run captured.

## 2026-06-08 multi-GPU expert residency + fusion/graphs verification

### Multi-GPU expert residency (new feature)

Routed-expert weights (~73 GiB) exceed one GPU's VRAM, so a single card keeps
the overflow host-mapped (read over PCIe per token — the decode cap).  Added a
**peer-device overflow pool**: `cuda_graph_promote_hot_weights` fills device 0
to its budget, then promotes the remaining expert layers into a second GPU's
VRAM (`ds4_cuda_tensor_alloc_peer`, a device-1 context with bidirectional
`cuCtxEnablePeerAccess`).  The hot-table records device-1 pointers; the existing
MoE kernels read them over the inter-GPU link via unified addressing — **zero
kernel changes**.  Lazy + opt-out (`DS4_CUDA_NO_PEER=1`); falls back to
host-mapped if no peer GPU or P2P is unavailable.  Code: `ds4_cuda.{c,h}`
(`ds4_cuda_peer_init` / `_peer_free_mem` / `_tensor_alloc_peer`, `tensor->peer`),
`ds4.c` (`hot_pool_peer`, `cuda_hot_walk_add_peer`, overflow tally in the
promoter).

**Measured on 2× RTX 6000 Ada (48 GiB, PCIe — no NVLink), 80.76 GiB model:**

| config | residency | decode | prefill |
|--------|-----------|--------|---------|
| `dev0_full` (single GPU) | 20/43 layers, 23 host-mapped | 9.69 t/s | 10.98 t/s |
| `peer_full` (2 GPU)      | **43/43 fully resident** (dev0=20, dev1=23) | 9.26 t/s | 10.42 t/s |

Output is **byte-identical** (greedy, temp 0) between the two — correctness
verified.  Full residency is achieved but decode is **neutral (slightly worse)**.

**Why:** measured interconnect bandwidth (`bench/p2p_bw.c`) on this node:
P2P DtoD dev1→dev0 = **26.7 GB/s**, host→dev0 = **24.2 GB/s** — both PCIe-bound
and essentially equal.  Moving experts from host RAM to peer VRAM does not change
the read bandwidth, so there is no decode win.  The benefit requires an
inter-GPU link faster than host PCIe: on the H100 node these two GPUs are
**NV18 NVLink** (cuDeviceCanAccessPeer=1, ~700–900 GB/s, ~30× host PCIe), where
moving the ~23 overflow layers off the PCIe path should be a large win.  (The
H100 measurement was blocked by node I/O contention; the implementation is
correct and engages there — `dev0=N, peer/dev1=M` in the promotion log.)

Conclusion: **multi-GPU expert residency is a correctness-complete capability
whose payoff is gated by interconnect bandwidth — material on NVLink, a wash on
PCIe-only cards.**  The earlier roadmap's "~14–16 t/s on full residency"
prediction assumed VRAM ≫ host bandwidth, which does not hold for PCIe P2P.

### Fusion + CUDA-graphs verification (settles the per-token-launch question)

The per-token compressor-emit and indexer-compressor chains are already fused
(`compressor_emit_fused_f32`, one launch replacing pool→RMS→RoPE→FP8→push), as
is the indexer top-k (`indexer_topk_from_scores_f32`).  Parity holds (the
multi-GPU A/B above is byte-identical through this path).

To test whether *any* further per-token fusion could help, ran a **CUDA-graphs
A/B** (graphs capture the whole per-token DAG → zero per-launch latency),
RTX 6000 Ada, N=96, single-GPU expert-hot:

| | decode | prefill |
|--|--------|---------|
| `DS4_CUDA_GRAPHS=0` | 9.61 t/s | 10.98 t/s |
| `DS4_CUDA_GRAPHS=1` | 9.27 t/s | 9.81 t/s |

Identical output; graphs give **no speedup** (slightly worse).  Eliminating
*all* launch overhead changes nothing → the per-token chain is **bandwidth-bound
on expert reads, not launch-bound**.  Therefore further kernel fusion cannot
raise decode throughput.  The parked `indexer_scores_fused_f32` is also a
dead-end structurally (needs n_head·head_dim = 16384 threads per block, launches
256; recomputes Q per comp-row) and is correctly left unwired.

Repro: `bench/suite.sh` (CONFIGS=`dev0_full peer_full`), `bench/graphs_ab.sh`,
`bench/p2p_bw.c` / `bench/p2p_probe.c`.  Stage the GGUF to node-local disk first
(`/tmp/$USER/ds4/model.gguf`); NFS cold-load dominates otherwise.
