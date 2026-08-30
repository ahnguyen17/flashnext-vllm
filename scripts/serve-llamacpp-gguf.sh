#!/usr/bin/env bash
# Qwen3.8-Flash-Next (176B MoE, UD-Q4_K_XL + mmproj vision) — PROD lane on port 8012.
# Drop-in replacement for the old Qwen3.8-27B-Int8 vLLM lane:
#   same port (8012), same auth (Bearer change-me), any model name accepted.
# Old lane rollback: ~/scripts/prod-restart-8012.sh (relaunches 27B vLLM).
# Tunnel: tailscale serve --bg --https=8144 http://127.0.0.1:8012
# Baseline: 34-36 tok/s decode @ 256K (repo: ahnguyen17/flashnext-170hx-baseline)
set -euo pipefail
# 170HX pair = nvidia-smi indices 2,3 (3090s are 0,1 — llama.cpp must NOT see them)
export CUDA_VISIBLE_DEVICES=2,3
cd ~/tools/llama.cpp-q4exp

exec ./build/bin/llama-server \
  -m ~/models/Flash-Next-NVMe/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --tensor-split 62,38 \
  -ngl 99 \
  --override-tensor per_layer_token_embd=CPU \
  -c 262144 \
  --parallel 1 \
  -fa on \
  -b 4096 -ub 2048 \
  -t 32 \
  --jinja \
  --mmproj ~/models/Flash-Next-NVMe/UD-Q4_K_XL/mmproj-F16.gguf \
  --alias flash-next \
  --api-key change-me \
  --host 0.0.0.0 --port 8012
