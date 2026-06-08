import json, urllib.request
def call(body):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read the full contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"}}, "required": ["path"]}}}
msgs = [
    {"role": "system", "content": "You are a coding agent. Use the provided tools to inspect files before answering. Always call a tool first when asked about file contents."},
    {"role": "user", "content": "What port does the server bind to? The configuration is in /app/config.py. Read it first."},
]
b = {"model": "deepseek-v4-flash", "messages": msgs, "tools": [TOOL], "max_tokens": 256,
     "temperature": 0, "stream": False, "thinking": {"type": "disabled"}}
o = call(b)
ch = o["choices"][0]; m = ch["message"]
print("finish:", ch.get("finish_reason"), "tool_calls:", len(m.get("tool_calls") or []))
print("RAW content:")
print(repr(m.get("content")))
if m.get("tool_calls"):
    print("tool_calls:", json.dumps(m["tool_calls"], indent=2))
