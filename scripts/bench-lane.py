#!/usr/bin/env python3
"""Lane bench: single/2-stream/4-stream aggregate t/s for the 8012 FlashNext lane.
Recreated Aug 31 2026 (original lived in /tmp, wiped by reboot).
Methodology: non-streaming chat completions, prompt 'Paris essay', max_tokens 600,
ignore_eos, aggregate t/s = total completion tokens / wall time per phase."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8012/v1/chat/completions"
KEY = "sophia"
PROMPT = "Write a detailed essay about the history of Paris."

def gen(timeout=180):
    body = json.dumps({
        "model": "flash-next",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 600,
        "ignore_eos": True,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    return out["usage"]["completion_tokens"], dt, out["choices"][0]["message"]["content"]

def phase(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: gen(), range(n)))
    total_tok = sum(r[0] for r in results)
    wall = max(r[1] for r in results)
    return total_tok / wall

# text sanity
tok, dt, text = gen.__wrapped__() if hasattr(gen, "__wrapped__") else (None, None, None)
body = json.dumps({"model": "flash-next",
                   "messages": [{"role": "user", "content": "Tell me about Paris."}],
                   "max_tokens": 48, "temperature": 0.7}).encode()
req = urllib.request.Request(URL, data=body, headers={
    "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    out = json.loads(r.read())
print(f"text: '{out['choices'][0]['message']['content'][:70]}'")

print(f"single: {phase(1):.1f} t/s")
print(f"2-stream: {phase(2):.1f} t/s")
print(f"4-stream: {phase(4):.1f} t/s")
