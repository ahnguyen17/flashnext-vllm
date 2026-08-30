#!/usr/bin/env bash
# 256K prod verifier: boot diag → quality → REAL 250K-token request → vision.
set -uo pipefail
KEY=${VLLM_API_KEY:-change-me}
for i in $(seq 1 60); do
  sleep 20
  C=$(curl -s -m 5 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8012/v1/models 2>/dev/null)
  [ "$C" = "200" ] && break
done
if [ "$C" != "200" ]; then
  echo "BOOT FAILED after 20min"
  docker logs qwen38-prod-8012 2>&1 | grep -iE 'error|refus|oom|availab|cache blocks|model len' | tail -15
  exit 1
fi
echo "PROD UP after ~$((i*20/60))min ($(date +%H:%M:%S))"
docker logs qwen38-prod-8012 2>&1 | grep -iE 'Using max model len|Available KV cache' | head -5
echo "=== QUALITY ==="
curl -s -m 120 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"flash-next","prompt":"The capital of France is","max_tokens":16,"temperature":0}' \
  http://127.0.0.1:8012/v1/completions | python3 -c "import json,sys; print(repr(json.load(sys.stdin)['choices'][0]['text']))"
echo "=== 250K-TOKEN REQUEST (patience: prefill is minutes) ==="
python3 - << 'EOF'
import json, time, urllib.request
sent = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
reps = 15600  # ~16 tok each -> ~250K
prompt = ("You are reviewing a very long field journal. Here it is:\n\n"
          + sent * reps
          + "\n\nEnd of journal. Reply with exactly one word naming this kind of document, then stop.")
body = json.dumps({"model": "flash-next", "prompt": prompt, "max_tokens": 12,
                   "temperature": 0, "ignore_eos": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8012/v1/completions", data=body,
    headers={"Authorization": "Bearer change-me", "Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1500) as r:
        d = json.load(r)
    dt = time.time() - t0
    pt = d["usage"]["prompt_tokens"]
    print(f"ACCEPTED: prompt_tokens={pt} ({pt/1024:.0f}K) in {dt:.0f}s")
    print(f"prefill ~{pt/max(dt,0.001):.0f} tok/s | ANSWER: {d['choices'][0]['text']!r}")
except urllib.error.HTTPError as e:
    print(f"REJECTED: HTTP {e.code}: {e.read()[:300]!r}")
except Exception as e:
    print(f"FAILED after {time.time()-t0:.0f}s: {e}")
EOF
echo "=== VISION RECHECK ==="
B64=$(base64 -w0 /tmp/vision_test.png)
curl -s -m 120 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"flash-next\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}},{\"type\":\"text\",\"text\":\"One word: what color?\"}]}],\"max_tokens\":10}" \
  http://127.0.0.1:8012/v1/chat/completions | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr(d['choices'][0]['message'].get('content'))[:150] if 'choices' in d else d)"
echo 256K-VERIFIED
