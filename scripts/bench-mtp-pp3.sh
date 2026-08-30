#!/usr/bin/env bash
# v33: site-23+24 — placeholders schedule freely (drafter engagement restored),
# phantom slots roll back on all three accountants. Expect: clean text AND
# SPEC-OUT lines (engagement) AND speed >= v32's 36 t/s.
set -euo pipefail
docker rm -f qwen38-pp3-bench 2>/dev/null || true
docker run --runtime nvidia -d --gpus '"device=1,2,3"' \
  --ipc=host --shm-size=96g --cap-add=SYS_PTRACE \
  --name qwen38-pp3-bench -p 8003:8000 \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache-v19:/root/.cache/vllm \
  -v ~/models/FlashNext-AWQ:/model:ro \
  --entrypoint vllm \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_API_KEY=change-me \
  --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  --env VLLM_PP_LAYER_PARTITION=8,28,12 \
  qwen38-flash-next:pp3fix21 \
  serve /model --served-model-name flash-next --max-model-len 32768 \
  --pipeline-parallel-size 3 --gpu-memory-utilization 0.85 --max-num-seqs 2 \
  --distributed-timeout-seconds 600 \
  --cpu-distributed-timeout-seconds 600 \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching --trust-remote-code --generation-config auto
echo "v37 (slots+ledger) launched $(date +%H:%M:%S)"
