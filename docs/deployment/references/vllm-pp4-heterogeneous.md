# vLLM PP=4 on Heterogeneous GPUs (64G + 40G + 2×24G 3090s) — 8/29

the operator's call after the PP=2 unfittable verdict: "can we add a 3090 or two?" — the two
3090s are ALREADY in the host (GPU0/GPU1 next to the 170HX pair), so the expansion
cost $0 hardware. SKILL.md has the verdict summary; this file has the capacity
math, placement rules, and exact launch recipe.

## Capacity ledger

| Config | Total VRAM | vs. author reference (2×64G = 128G) |
|---|---|---|
| PP2 (64+40) | 104 GB | short by single-digit GB (died in graph capture) |
| PP3 (+1×3090) | 128 GB | exact parity with author's proven shape |
| PP4 (+2×3090) | 152 GB | +24 GB headroom |

Weight mass observed on PP2 loads: **~1.6 GB/layer** (AWQ-INT4; 32/16 split →
51.41G/33.45G), plus drafter + lm_head + graphs (several GB) on the LAST rank only.

## Rig GPU map (audited 8/29 — ALWAYS re-audit before claiming GPUs)

| GPU | Card | State |
|---|---|---|
| 0 | RTX 3090 (24G) | **SHARED** — `qwen3.5-4b-uncensored` llama-server (port 8090, ~4.5G standing, up for days, has vision). Live lane — do NOT kill. |
| 1 | RTX 3090 (24G) | idle |
| 2 | CMP 170HX (64G) | prod GGUF (when up) |
| 3 | CMP 170HX (40G) | prod GGUF (when up) |

Audit command: `nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory,process_name --format=csv,noheader` then `/proc/PID/cmdline` + `/proc/PID/cgroup` (a docker path in cgroup = containerized tenant; `docker ps` name lookup follows).

## Rank placement rules (the generalizable core)

1. **Rank order = order in the `--gpus '"device=..."'` list** (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`). Map ranks to cards deliberately.
2. **The LAST rank carries lm_head + MTP drafter + speculator CUDA graphs + final norm** — the memory-heaviest tail. Give it the card with the most CLEAN headroom (here: the 40G 170HX), NOT the smallest card. A 24G 3090 as last rank would re-create the suffocation that killed PP2.
3. **Taxed GPUs (standing tenants) get light MID-pipeline ranks** — few layers, no head burden. GPU0 (4.5G tenant) took 8 layers = ~12.8G under ~19G free.
4. **`--gpu-memory-utilization` is ONE value across all ranks** — the ceiling is set by the most-taxed GPU (free/total ≈ 0.78 here); budget layers so every rank fits at that common util.
5. Balance layers by VRAM, not evenly: 20/8/8/12 (rank0 64G→20L, rank1 taxed 3090→8L, rank2 clean 3090→8L, rank3 40G→12L+drafter).

## Exact launch (PP4, 8/29)

```bash
docker run --runtime nvidia -d --gpus '"device=2,0,1,3"' \
  --ipc=host --shm-size=96g --cap-add=SYS_PTRACE \
  --name qwen38-pp4-bench -p 8001:8000 \
  -v vllm-hf-cache:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  -v ~/models/FlashNext-AWQ:/model:ro \
  --entrypoint vllm \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --env VLLM_PLE_CPU_OFFLOAD=1 \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env VLLM_API_KEY=change-me \
  --env VLLM_PP_LAYER_PARTITION=20,8,8,12 \
  qwen38-flash-next:pp2v2 \
  serve /model --served-model-name flash-next \
  --max-model-len 32768 --pipeline-parallel-size 4 \
  --gpu-memory-utilization 0.78 --max-num-seqs 4 \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --trust-remote-code --generation-config auto
```

No `--async-scheduling` (OFF for first boots — warmup sensitivity), local model
path mount instead of HF hub id, `--entrypoint vllm` mandatory (pp2v2's commit
clobbered it to `python3 /fix.py` — symptom: `can't open file '/workspace/serve'`).

Site-7 PLE patch verified PP4-safe by inspection: guard triggers only at
`pp_size==1` (PleOffloadWorker); a 4-element partition env passes through the
normal path (`/opt/vllm/vllm/distributed/utils.py`, `get_pp_indices`).

## Bench-window ops sequence (proven 8/29)

1. **Pause the prod watchdog cron FIRST** (`8012-nightly-watchdog`, */15) — else it relaunches prod into the GPUs mid-bench.
2. **Kill prod by EXACT PID** from `nvidia-smi --query-compute-apps` (was 726429). NEVER broad `pkill llama-server` — matches the 8090 lane and any other tenant.
3. Verify GPUs released (compute-apps shows only the 8090 tenant).
4. Launch + arm ONE watcher (background, notify_on_complete): poll :8001, container Running check, crash-pattern grep, on UP → dump KV-cache lines + run decode bench.
5. Restore: `~/scripts/serve-flashnext-8012.sh` (~160s to healthy; thinking-model health check needs max_tokens ≥300 or read reasoning_content), then RESUME the watchdog cron.

## Stale-watcher forensics (post-compaction)

Watchers armed before a context compaction deliver LATE notifications describing
already-handled events. Before declaring a new failure mode: docker log timestamps
are container UTC (local PDT = UTC−7); a trailing `logout` line in `docker logs`
output means the CONTAINER exited — including when YOUR OWN earlier `docker rm -f`
killed it (8/29: a "mystery relaunch" was just my own cleanup timestamped between
two watcher reports). Reconcile wall-clock order of your own actions first.

## Status

Launched 8/29 ~10:55 PDT, watcher armed (`proc_9d8260fdf44c`), bar: **≥50 t/s decode
or the AWQ book closes.** PP2 failure signatures for comparison: OOM at
CaptureDescriptor = graph headroom; NCCL collective-timeout dump ~10 min after a
rank stall = cold-compile watchdog kill; negative `kv-cache-memory` in startup
lines = impossible partition. Result → append here + update SKILL.md verdict line.
