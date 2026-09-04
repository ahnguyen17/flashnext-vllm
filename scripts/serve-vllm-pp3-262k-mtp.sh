#!/usr/bin/env bash
# PROD: vLLM Flash-Next AWQ-INT4, 262,144 NATIVE ctx + MTP-3 — pp3fix25 (site-26 + site-30 boot warmup), promoted 2026-09-04.
# site-30 = VLLM_PP_WARMUP=1 runs kernel warmup at PP>1 (env-gated site-9/10 bypass; unset = old skip behavior).
# --no-enable-prefix-caching is MANDATORY: nightly images default it ON; cache-ON wedges this rig (see docs).
# Base = the proven v20 no-MTP PP3 recipe + site-26 draft-table ring sync (MTP at PP>1).
# Deltas vs the v20 no-MTP recipe:
#   1. --speculative-config mtp k=3 (site-26 fixes draft-table propagation at PP>1)
#   2. dual served-model aliases: `flash-next` AND `qwen3.8-27b` (client compat)
#   3. nothing else moves: partition 8,28,12, seqs 4, util 0.85 (2x pool margin;
#      drafter weights + draft KV land on PP2's slack — costs ~9% of pool)
# VLLM_GDN_DECODE_KERNEL=triton is baked into the pp3fix22 image env; passed
# explicitly here for portability to images that don't bake it.
# Rollback to the 786K YaRN no-MTP lane: scripts/serve-vllm-pp3-786k.sh
set -euo pipefail
docker rm -f qwen38-prod-8012 2>/dev/null || true
docker run --runtime nvidia -d --gpus '"device=1,2,3"' \
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
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env VLLM_API_KEY=change-me \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  --env VLLM_PP_LAYER_PARTITION=8,28,12 \
  --env VLLM_PP_WARMUP=1 \
  qwen38-flash-next:pp3fix25 \
  serve /model --served-model-name flash-next qwen3.8-27b --max-model-len 262144 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --mamba-cache-mode align \
  --no-enable-prefix-caching \
  --trust-remote-code --generation-config auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "vLLM PP3 262K + MTP-3 (pp3fix22, util 0.85) launched"
