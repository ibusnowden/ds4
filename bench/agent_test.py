#!/usr/bin/env python3
"""Agentic-engineering smoke test for the optimized ds4-server.

Validates that the block-cooperative MoE rewrite + EXPERT_HOT changes did not
break the model's agentic behaviour: tool-call emission, multi-turn tool loop,
and code generation.  Drives the OpenAI-compatible endpoint directly so it is
reproducible without the interactive Pi TUI.
"""
import json, os, sys, time, urllib.request, urllib.error

BASE = os.environ.get("DS4_BASE", "http://127.0.0.1:8000/v1")
KEY = os.environ.get("DS4_KEY", "dsv4-local")
MODEL = os.environ.get("DS4_MODEL", "deepseek-v4-flash")


def call(messages, tools=None, tool_choice=None, max_tokens=512, think=False):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    if tools: body["tools"] = tools
    if tool_choice: body["tool_choice"] = tool_choice
    if not think: body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read().decode())
    return obj, time.monotonic() - t0


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the full contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "file path"}},
            "required": ["path"],
        },
    },
}

passed = []
failed = []

def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)


print("=== Test 1: tool-call emission ===", flush=True)
msgs = [
    {"role": "system", "content": "You are a coding agent. Use the provided tools to inspect files before answering. Always call a tool first when asked about file contents."},
    {"role": "user", "content": "What port does the server bind to? The configuration is in /app/config.py. Read it first."},
]
obj, dt = call(msgs, tools=[READ_TOOL], max_tokens=256)
choice = obj["choices"][0]
msg = choice["message"]
tcs = msg.get("tool_calls") or []
print(f"  finish_reason={choice.get('finish_reason')} tool_calls={len(tcs)} dt={dt:.1f}s", flush=True)
ok_tc = len(tcs) >= 1 and tcs[0]["function"]["name"] == "read_file"
check("emits a tool_call", ok_tc, f"got {[t['function']['name'] for t in tcs]}" if tcs else "no tool_calls")
args_ok = False
if ok_tc:
    try:
        a = json.loads(tcs[0]["function"]["arguments"])
        args_ok = a.get("path") == "/app/config.py"
        print(f"  tool args: {a}", flush=True)
    except Exception as e:
        print(f"  arg parse error: {e}", flush=True)
check("tool args are valid JSON with correct path", args_ok)

print("\n=== Test 2: multi-turn tool result -> final answer ===", flush=True)
if ok_tc:
    msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
    msgs.append({"role": "tool", "tool_call_id": tcs[0].get("id", "call_0"),
                 "content": "PORT = 8080\nHOST = '0.0.0.0'\nDEBUG = False\n"})
    obj2, dt2 = call(msgs, tools=[READ_TOOL], max_tokens=256)
    final = obj2["choices"][0]["message"].get("content") or ""
    print(f"  final answer ({dt2:.1f}s): {final[:200]}", flush=True)
    check("final answer mentions port 8080", "8080" in final)
else:
    check("final answer mentions port 8080", False, "skipped (no initial tool call)")

print("\n=== Test 3: code generation coherence ===", flush=True)
obj3, dt3 = call([
    {"role": "user", "content": "Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. Return only a code block."}
], max_tokens=400)
code = obj3["choices"][0]["message"].get("content") or ""
print(f"  ({dt3:.1f}s):\n{code[:400]}", flush=True)
check("contains a def fib", "def fib" in code)
# functional check: extract and run the function
func_ok = False
try:
    import re
    m = re.search(r"def fib\(.*?(?=\n\S|\Z)", code, re.S)
    if m:
        ns = {}
        exec(m.group(0), ns)
        func_ok = ns["fib"](10) == 55
except Exception as e:
    print(f"  exec error: {e}", flush=True)
check("generated fib(10) == 55", func_ok)

print(f"\n=== RESULT: {len(passed)} passed, {len(failed)} failed ===", flush=True)
if failed:
    print("FAILED:", failed, flush=True)
    sys.exit(1)
print("ALL AGENTIC CHECKS PASSED", flush=True)
