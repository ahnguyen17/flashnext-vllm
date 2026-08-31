#!/bin/bash
# Lane verify+bench for 8012 (permanent home — /tmp copies died at reboot twice).
# Usage: bash ~/scripts/verify-postdriver.sh
start=$(date +%s)
code=000
for i in $(seq 1 120); do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://localhost:8012/health 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 8
done
if [ "$code" != "200" ]; then echo "BOOT FAILED (health=$code)"; exit 1; fi
echo "UP after ~$(( ($(date +%s) - start) / 60 ))min ($(date +%H:%M:%S))"
python3 ~/scripts/bench-lane.py
echo "BENCH-DONE"
echo "JIT during bench: $(docker logs --since 4m qwen38-prod-8012 2>&1 | grep -c jit_monitor || true)"
