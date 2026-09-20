#!/bin/bash
# PROD 8012: vLLM Flash-Next AWQ-INT4, 262,144 NATIVE ctx + MTP-3 (pp3fix22, site-26) — promoted 2026-09-02 00:10.
# PROD 8012 candidate: vLLM Flash-Next AWQ-INT4, 262,144 NATIVE ctx + MTP-3 (2026-09-01 night).
# Base = proven v20 256K recipe + pp3fix22 (site-26 draft-table ring sync — MTP at PP).
# util 0.85 (proven at 262K, 2x pool margin; drafter weights+KV land on PP2 slack).
# 2026-09-05: GPU1 3090 (old PP0/cuda:0) physically pulled from rig — re-pinned to UUIDs:
#   cuda:0/PP0 = GPU0 3090 (GPU-1728…), cuda:1/PP1 = 170HX 64G (GPU-cde9…), cuda:2/PP2 = 170HX 40G (GPU-b522…).
#   UUIDs immune to index renumber; PCI_BUS_ID keeps ascending order; partition 8,28,12 unchanged (both 3090s 24GB).
# Rollback (786K no-MTP): docker rm -f qwen38-prod-8012 && bash ~/scripts/serve-flashnext-vllm-8012-786k-nomtp.sh.bak
# 2026-09-10 REORDER PROMOTED: rank0 off the 3090 (allover326 lesson — pure-sm80 rank0).
#   PCI base: cuda:0=3090, cuda:1=170HX-64G, cuda:2=170HX-40G. CUDA_VISIBLE_DEVICES=1,0,2
#   remaps → PP0=170HX-64G (28L), PP1=3090 (8L), PP2=170HX-40G+drafter (12L).
#   Same per-PHYSICAL-card layers as old 8,28,12 → identical capacity (pool 505K class).
#   Measured 9/10: decode 103-106 @32K / 100 @64K / 87 @131K; 4-conc 177.8 agg;
#   FIRST-DEEP NEEDLES 2/2 NO WEDGE (fresh-process lottery gone with rank0 off sm_86).
#   Rollback: delete the CUDA_VISIBLE_DEVICES line + partition back to 8,28,12.
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
echo "PROD 8012: 262K native + MTP-3, pp3fix26 + VLLM_PP_WARMUP=1 (site-30 boot warmup), prefix-cache OFF explicit $(date +%H:%M:%S)"
