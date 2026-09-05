#!/bin/bash
# boot-battery-8012: fires a varied-length warmup battery the moment the prod
# container (re)starts, so first-use kernel loads (spec-decode/post_update
# family, per-shape graphs) happen on shallow traffic instead of the first
# real/deep request. Catches docker --restart crash-loops too (docker events).
# Evidence 9/4: fresh boot + 131K-first => wedge 0/3; boot + short battery =>
# 131K passes. Log: /tmp/boot-battery.log
set -u
LOG=/tmp/boot-battery.log
B=http://127.0.0.1:8012
H='Authorization: Bearer <API-KEY>'
CT='Content-Type: application/json'

gen() {  # gen <prompt> <max_tokens>
  curl -s -m 300 -o /dev/null -w "%{http_code} %{time_total}s" \
    -H "$H" -H "$CT" -d "{\"model\":\"flash-next\",\"temperature\":0,\"max_tokens\":$2,\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}]}" \
    $B/v1/chat/completions
}

battery() {
  echo "[$(date +%H:%M:%S)] battery start" >> "$LOG"
  local t0=$(date +%s)
  # 1. tiny factual (probe-class)
  echo "  tiny:      $(gen 'The capital of France is' 64)" >> "$LOG"
  # 2. medium (spec-verify steps, different output shape)
  echo "  medium:    $(gen 'Name three primary colors and their hex codes.' 200)" >> "$LOG"
  # 3. sustained ~700 (long spec decode, many draft/verify steps)
  echo "  sustain:   $(gen 'Write a detailed 500-word explanation of how pipeline parallelism works in modern LLM inference systems.' 700)" >> "$LOG"
  # 4. second tiny (post-sustain shape re-check)
  echo "  tiny2:     $(gen 'The capital of Japan is' 64)" >> "$LOG"
  echo "[$(date +%H:%M:%S)] battery done in $(( $(date +%s) - t0 ))s" >> "$LOG"
}

wait_healthy() {
  for i in $(seq 1 120); do
    curl -s -m 5 -o /dev/null $B/health && return 0
    sleep 10
  done
  return 1
}

echo "[$(date +%H:%M:%S)] boot-battery daemon armed" >> "$LOG"
docker events --filter container=qwen38-prod-8012 --filter event=start --format '{{.Time}}' | while read -r t; do
  echo "[$(date +%H:%M:%S)] container start event" >> "$LOG"
  if wait_healthy; then
    sleep 3
    if curl -s -m 5 -o /dev/null $B/health; then
      battery
    else
      echo "[$(date +%H:%M:%S)] health flapped, skipping" >> "$LOG"
    fi
  else
    echo "[$(date +%H:%M:%M)] never healthy (20min), skipping" >> "$LOG"
  fi
done
