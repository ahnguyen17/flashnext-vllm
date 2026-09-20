#!/bin/bash
# PROD 8012: vLLM Flash-Next AWQ-INT4, 262,144 NATIVE ctx + MTP-3 — pp3fix27-mambaalign (site-32).
# 2026-09-19: cache-OFF baseline preserved in serve-flashnext-vllm-8012.sh.bak-96k-cacheoff.
#   This variant RE-ENABLES prefix caching (--enable-prefix-caching + --prefix-match-unit 400,
#   unit must divide block 1600). Image carries site-32 mamba-align fix (port of
#   akumaburn/vllm-dflash2 1c7542a4c5): align sites must read mamba_block_size (1648/1600),
#   not cache_config.block_size (shrinks to smallest KV-group block after groups are built).
#   Without it, first prefix hit overruns the mamba block table -> CUDA IMA / wrong resume column.
# GPU pinning + rank order per 9/10 reorder: PP0=170HX-64G, PP1=3090, PP2=170HX-40G+drafter.
set -euo pipefail
docker rm -f qwen38-prod-8012 2>/dev/null || true
docker run --runtime nvidia -d --gpus '"device=GPU-17287294-fae8-b240-5c44-1514a55874b5,GPU-cde98005-379e-d9da-e952-11e0f6424bb3,GPU-b5223098-14ee-c9c2-8dc5-73a494e5cf87"' \
  --ipc=host --shm-size=96g --cap-add=SYS_PTRACE \
  --name qwen38-prod-8012 -p 8012:8000 \
  --restart unless-stopped \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache-v19:/root/.cache/vllm \
  -v vllm-triton-cache:/root/.triton \
  -v /home/f0ol/models/FlashNext-AWQ:/model:ro \
  --entrypoint vllm \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env CUDA_VISIBLE_DEVICES=1,0,2 \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_API_KEY=sophia \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  --env VLLM_PP_LAYER_PARTITION=28,8,12 \
  --env VLLM_PP_WARMUP=1 \
  qwen38-flash-next:pp3fix27-mambaalign \
  serve /model --served-model-name flash-next qwen3.8-27b --max-model-len 262144 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.85 --max-num-seqs 4 \
  --distributed-timeout-seconds 3600 \
  --cpu-distributed-timeout-seconds 3600 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --prefix-match-unit 400 \
  --trust-remote-code --generation-config auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
echo "PROD 8012: 262K native + MTP-3, pp3fix27-mambaalign (site-32) + prefix caching ON (pmu400), $(date +%H:%M:%S)"
