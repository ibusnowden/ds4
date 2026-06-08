import json, urllib.request
def call(body):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": "Bearer x", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
TOOLS = [
 {"type":"function","function":{"name":"read","description":"Read a file's contents.",
   "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"ls","description":"List a directory.",
   "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
]
msgs=[{"role":"system","content":"You are a coding assistant. Use the available tools to inspect files before answering."},
      {"role":"user","content":"Read config.py in the current directory and tell me the value of PORT. Answer with just the number."}]
b={"model":"deepseek-v4-flash","messages":msgs,"tools":TOOLS,"max_tokens":400,"temperature":0,"stream":False,"thinking":{"type":"disabled"}}
o=call(b); ch=o["choices"][0]; m=ch["message"]
print("finish:",ch.get("finish_reason"),"tool_calls:",len(m.get("tool_calls") or []))
c=m.get("content") or ""
print("content_len:",len(c))
print("content[:1200]:")
print(c[:1200])
