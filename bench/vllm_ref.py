#!/usr/bin/env python3
"""vLLM reference benchmark on RTX 6000 Ada, to anchor the ds4 comparison.

1. Attempt to load DeepSeek-V4-Flash (the model ds4 actually runs) — documents
   whether vLLM can serve it at all.
2. Benchmark a runnable proxy (Mistral-7B) for batch-1 prefill/decode tok/s and
   batched throughput, as a "what a tuned engine does on this GPU" anchor.

Compute nodes are offline: HF_HUB_OFFLINE + direct snapshot paths.
"""
import os, sys, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

HUB = "/project/inniang/hf-cache/hub"
def snap(name):
    d = os.path.join(HUB, name, "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[0])

DS4 = snap("models--deepseek-ai--DeepSeek-V4-Flash")
OLMO7B = snap("models--allenai--OLMo-2-1124-7B-Instruct")
QWEN27B = snap("models--Qwen--Qwen3.6-27B")


def try_deepseek_v4():
    print("=== [1] Attempt vLLM load of DeepSeek-V4-Flash ===", flush=True)
    print(f"path={DS4}", flush=True)
    try:
        from vllm import LLM
        llm = LLM(model=DS4, tensor_parallel_size=2, trust_remote_code=True,
                  enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.9)
        print("DeepSeek-V4-Flash LOADED in vLLM (unexpected!)", flush=True)
        return llm
    except Exception as e:
        print(f"DeepSeek-V4-Flash vLLM load FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        return None


def bench_model(path, name, tp=1):
    from vllm import LLM, SamplingParams
    print(f"\n=== [2] vLLM benchmark: {name} (TP={tp}) ===", flush=True)
    t0 = time.time()
    llm = LLM(model=path, tensor_parallel_size=tp, dtype="float16",
              enforce_eager=False, max_model_len=4096, gpu_memory_utilization=0.85)
    print(f"load_s={time.time()-t0:.1f}", flush=True)

    prompt_short = "Scrivi una storia su una papera scansafatiche."
    # long prompt ~ many tokens for prefill measurement
    prompt_long = ("Write a detailed technical essay about distributed systems, "
                   "consensus algorithms, and fault tolerance. ") * 28

    # --- batch-1 decode tok/s ---
    sp = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    _ = llm.generate([prompt_short], sp)  # warmup
    t0 = time.time()
    out = llm.generate([prompt_short], sp)
    dt = time.time() - t0
    n_out = len(out[0].outputs[0].token_ids)
    print(f"{name} batch1 decode: {n_out} tok in {dt:.3f}s = {n_out/dt:.2f} tok/s (incl prefill)", flush=True)

    # --- prefill tok/s: long prompt, 1 output token ---
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    out_p = llm.generate([prompt_long], sp1)
    n_prompt = len(out_p[0].prompt_token_ids)
    t0 = time.time()
    out_p = llm.generate([prompt_long], sp1)
    dt_p = time.time() - t0
    print(f"{name} prefill: {n_prompt} prompt tok in {dt_p:.3f}s = {n_prompt/dt_p:.1f} tok/s", flush=True)

    # --- throughput: many concurrent requests (continuous batching) ---
    for B in (16, 64):
        sp_t = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
        prompts = [prompt_short + f" (variant {i})" for i in range(B)]
        t0 = time.time()
        outs = llm.generate(prompts, sp_t)
        dt_t = time.time() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{name} throughput B={B}: {total_out} out tok in {dt_t:.3f}s = {total_out/dt_t:.1f} tok/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ds4", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--model", default="olmo", choices=["olmo", "qwen", "all"])
    args = ap.parse_args()
    if not args.skip_ds4:
        try_deepseek_v4()
    which = args.model
    if which in ("olmo", "all"):
        bench_model(OLMO7B, "OLMo-2-7B", tp=1)
    if which in ("qwen", "all"):
        bench_model(QWEN27B, "Qwen3.6-27B", tp=2)
    print("=== vLLM reference done ===", flush=True)
