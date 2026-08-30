#!/usr/bin/env python3
"""Drop-in lane verification battery.

Verifies a replacement server faithfully matches a retired lane's observable
API contract BEFORE declaring the swap done. Covers the failure modes that a
mere "curl /v1/models works" check misses:

  - auth enforcement (ops scripts assert wrong-key -> 401; a keyless swap-in
    silently accepts everything and breaks those assertions)
  - /health must stay unauthenticated (watchdogs curl it bare)
  - legacy model-name strings (llama-server ignores the model field; vLLM may
    not — know which behavior your harnesses rely on)
  - thinking models: empty content + nonempty reasoning_content at small
    max_tokens is NOT a failure (budget here is 300)
  - vision, when the lane claims multimodal (mmproj loaded)

Proven on the 8/28/26 swap: vLLM Qwen3.8-27B-Int8 -> llama.cpp Flash-Next on
port 8012 (all checks green; runbook: SKILL.md "Access" + "Rollback" sections).

Usage:
  python3 verify_dropin.py --base http://127.0.0.1:8012 --key change-me \
      --model qwen3.8-27b \
      [--image /path/to/test.png] \
      [--extra https://host:8144/v1/models ...]

Exit code 0 only if every check passes. stdlib only.
"""
import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def http(url, key=None, data=None, timeout=15):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, body, timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:80].encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8012")
    ap.add_argument("--key", required=True, help="expected bearer key")
    ap.add_argument("--model", required=True,
                    help="EXACT model string a real harness sends")
    ap.add_argument("--image", help="image path for a vision check")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra URLs (tunnel/funnel) to check with --key")
    args = ap.parse_args()

    results = []

    def check(name, cond, detail=""):
        results.append((name, cond, detail))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    s, _ = http(args.base + "/health")
    check("health no-key  -> 200", s == 200, f"got {s}")
    s, _ = http(args.base + "/health", args.key)
    check("health withkey -> 200", s == 200, f"got {s}")
    s, _ = http(args.base + "/v1/models", "definitely-wrong-key")
    check("models wrongkey -> 401", s == 401, f"got {s}")
    s, body = http(args.base + "/v1/models", args.key)
    served = ""
    if s == 200:
        try:
            served = json.loads(body)["data"][0]["id"]
        except Exception:  # noqa: BLE001
            served = "?"
    check("models rightkey -> 200", s == 200, f"id={served}")

    chat = {"model": args.model, "max_tokens": 300,
            "messages": [{"role": "user",
                          "content": "Reply with exactly: DROP-IN OK"}]}
    s, body = http(args.base + "/v1/chat/completions", args.key, chat, 120)
    content = reasoning = ""
    if s == 200:
        m = json.loads(body)["choices"][0]["message"]
        content = (m.get("content") or "")[:40]
        reasoning = (m.get("reasoning_content")
                     or m.get("reasoning") or "")[:20]
    ok = s == 200 and (content or reasoning)
    check(f"chat exact model string ({args.model})", ok,
          f"content={content!r} reasoning={reasoning!r}")

    if args.image:
        with open(args.image, "rb") as f:
            img = base64.b64encode(f.read()).decode()
        vision = {"model": args.model, "max_tokens": 300, "messages": [{
            "role": "user", "content": [
                {"type": "text", "text": "Describe this image in one line."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + img}}]}]}
        s, body = http(args.base + "/v1/chat/completions", args.key,
                       vision, 180)
        desc = ""
        if s == 200:
            m = json.loads(body)["choices"][0]["message"]
            desc = (m.get("content")
                    or m.get("reasoning_content") or "")[:50]
        check("vision + auth", s == 200 and bool(desc), f"{desc!r}")

    for url in args.extra:
        s, _ = http(url, args.key)
        check(f"extra {url} -> 200", s == 200, f"got {s}")

    fails = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
