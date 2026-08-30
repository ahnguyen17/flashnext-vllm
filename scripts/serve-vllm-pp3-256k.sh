#!/usr/bin/env bash
# PROD 8012: vLLM Flash-Next AWQ-INT4 lane (no-MTP, proven v20 recipe).
# Swapped in 2026-08-30 ~04:10 from llama.cpp GGUF lane (rollback below).
# 48-54 t/s sustained, clean text, VISION LIVE (ViT in checkpoint, tested
# with image → correct description). Context 262144 (256K): 12 full-attn
# layers (interval 4) × 2 KV heads × 256 dim bf16 → per-rank KV/token
# PP0=4KB PP1=14KB PP2=6KB; default pools at util 0.85 hold 588K+ tokens
# on the binding rank (PP1) → 2.2x margin. NO --kv-cache-memory flag (it
# caps ranks globally and would strangle PP1 below 256K).
# Rollback: docker rm -f qwen38-prod-8012 && bash ~/scripts/serve-flashnext-8012.sh
# Tunnel survives (points at port): tailscale serve https=8144 -> 127.0.0.1:8012
set -euo pipefail
docker rm -f qwen38-prod-8012 2>/dev/null || true
docker run --runtime nvidia -d --gpus '"device=1,2,3"' \
  --ipc=host --shm-size=96g --cap-add=SYS_PTRACE \
  --name qwen38-prod-8012 -p 8012:8000 \
  --restart unless-stopped \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache-v19:/root/.cache/vllm \
  -v ~/models/FlashNext-AWQ:/model:ro \
  --entrypoint vllm \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env VLLM_API_KEY=change-me \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  --env VLLM_PP_LAYER_PARTITION=8,28,12 \
  qwen38-flash-next:pp3fix7 \
  serve /model --served-model-name flash-next qwen3.8-27b --max-model-len 262144 \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.85 --max-num-seqs 2 \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --mamba-cache-mode align \
  --enable-prefix-caching --trust-remote-code --generation-config auto
echo "vLLM prod 8012 (256K ctx) launched $(date +%H:%M:%S)"
