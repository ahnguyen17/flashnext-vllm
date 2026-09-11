#!/usr/bin/env bash
# PROD since 2026-09-10: 262K native + MTP-3 with RANK REORDER — rank0 moved off the sm_86
# (desktop-class) card onto the 64 GB sm_80 card. Same image, same per-PHYSICAL-card layer
# counts — only pipeline ORDER changes. Full write-up: docs/deployment/rank-reorder-pp3.md
#
# Prerequisites (unchanged from the 262k-mtp recipe): image pp3fix26 (sites 8-10, 13,
# 17-18, 23, 26, 30, 31), PLE CPU offload, MTP k=3, prefix caching OFF.
#
# THE TWO DELTAS:
#   1. CUDA_VISIBLE_DEVICES remap INSIDE the container, on top of the PCI-ordered base.
#      On this rig the PCI base is cuda:0=RTX 3090 (24G), cuda:1=170HX 64G, cuda:2=170HX 40G.
#      "1,0,2" puts the 64G card at PP0, the 3090 at PP1, the 40G card at PP2 (drafter tail).
#      PROBE YOUR OWN ENUMERATION FIRST (torch.cuda.get_device_properties) — the nvidia
#      container runtime hands devices to the container in PCI-bus order regardless of
#      CUDA_DEVICE_ORDER on the host, so FASTEST_FIRST does NOT work as a lever here.
#   2. VLLM_PP_LAYER_PARTITION re-numbered 8,28,12 -> 28,8,12 so every PHYSICAL card keeps
#      its exact layer count from the old recipe (64G=28L, 3090=8L, 40G+drafter=12L).
#      Capacity/KV pool are identical to the previous prod; only pipeline order moves.
#
# Rollback: delete the CUDA_VISIBLE_DEVICES line and set the partition back to 8,28,12
# (with the device list still PCI-ascending).
set -euo pipefail
docker rm -f qwen38-prod-8012 2>/dev/null || true
docker run --runtime nvidia -d --gpus '"device=0,1,2"' \
  --ipc=host --shm-size=96g --cap-add=SYS_PTRACE \
  --name qwen38-prod-8012 -p 8012:8000 \
  --restart unless-stopped \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache-v19:/root/.cache/vllm \
  -v vllm-triton-cache:/root/.triton \
  -v ~/models/FlashNext-AWQ:/model:ro \
  --entrypoint vllm \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES=1,0,2 \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_API_KEY=change-me \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  --env VLLM_PP_LAYER_PARTITION=28,8,12 \
  --env VLLM_PP_WARMUP=1 \
  qwen38-flash-next:pp3fix26 \
  serve /model --served-model-name flash-next qwen3.8-27b --max-model-len 262144 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --mamba-cache-mode align \
  --no-enable-prefix-caching \
  --trust-remote-code --generation-config auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "vLLM PP3 262K + MTP-3, rank-reordered (PP0 = 64G sm_80) launched"
