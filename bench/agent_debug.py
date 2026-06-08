import json, urllib.request

def call(body):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
msgs = [{"role": "user", "content": "Read the file /app/config.py to find the port, using the read_file tool."}]

for label, extra in [("nothink+auto", {"thinking": {"type": "disabled"}, "tool_choice": "auto"}),
                     ("think+auto", {"tool_choice": "auto"}),
                     ("nothink+required", {"thinking": {"type": "disabled"}, "tool_choice": "required"})]:
    b = {"model": "deepseek-v4-flash", "messages": msgs, "tools": [TOOL],
         "max_tokens": 300, "temperature": 0, "stream": False}
    b.update(extra)
    try:
        o = call(b)
    except Exception as e:
        print(f"=== {label}: ERROR {e}")
        continue
    ch = o["choices"][0]; m = ch["message"]
    tcs = m.get("tool_calls") or []
    print(f"=== {label}: finish={ch.get('finish_reason')} tool_calls={len(tcs)}")
    if tcs:
        print("  tool_call:", tcs[0]["function"]["name"], tcs[0]["function"].get("arguments"))
    print("  content:", repr((m.get("content") or "")[:400]))
    rc = m.get("reasoning_content")
    if rc:
        print("  reasoning:", repr(rc[:200]))
    print()
