#!/usr/bin/env bash
# PROD: vLLM Flash-Next AWQ-INT4, 786,432-token context via YaRN 3.0 (2026-09-01).
# Base = the proven v20 no-MTP PP3 recipe with 3 deltas:
#   1. --max-model-len 786432 + YaRN override (factor 3.0 over native 262144)
#   2. --gpu-memory-utilization 0.85 -> 0.92 (binding rank PP1 needs 10.99 GiB KV
#      for 786K tokens; util 0.90 gives 10.6 -> clean boot-fail with
#      "estimated maximum model length is 758912"; 0.92 -> pool 846,453 tokens)
#   3. nothing else changes (partition 8,28,12, seqs 4, mamba align)
# NOTE: this vLLM lineage reads the Transformers-v5 `rope_parameters` key
# (NOT `rope_scaling`), and --hf-overrides deep-merges nested PretrainedConfig,
# so the text_config form below is safe. A missing/ignored override fails fast:
# the engine rejects max_model_len > derived (262144) before weight load.
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
  qwen38-flash-next:pp3fix7 \
  serve /model --served-model-name flash-next --max-model-len 786432 \
  --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings":262144}}}' \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.92 --max-num-seqs 4 \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --mamba-cache-mode align \
  --trust-remote-code --generation-config auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "vLLM PP3 786K (YaRN 3.0, util 0.92) launched"
