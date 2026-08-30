# PP4/PP3 execution log — 8/29 evening window (the operator: "Do it now")

Context: PP4 plan per `vllm-pp4-heterogeneous.md` executed 8/29 ~17:30 PDT. Prod 8012 down, watchdog paused, 8090 lane on GPU0 audited + untouched.

## PP4 result: DIED IN WEIGHT LOADING — not VRAM

All three non-first ranks (PP1=taxed 3090, PP2=clean 3090, PP3=40G) failed ~2 min in, identical frame:

```
weight_utils.py:970 safetensors_weights_iterator
    param = f.get_tensor(name)
ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'
```

Rank0 (64G) showed no error — but likely just hadn't reached the failing point before executor teardown (teardown hit at the first worker death). Treat as "the loader breaks under >2 GPU worker ranks", not as a rank-0 survivor.

## Isolation ladder (proven, reuse for any "loader works at N ranks but not M" mystery)

1. **Full single-proc sweep** — every tensor of every shard, materialized and freed:
   ```python
   for fp in sorted(glob('/model/*.safetensors')):
       with safe_open(fp, framework='pt', device='cpu') as f:
           for k in f.keys():
               t = f.get_tensor(k); del t   # try/except per tensor
   ```
   Result: 38 shards × ~10.7K keys, ZERO failures → files pristine (don't re-download, don't re-verify hashes).
2. **Hub-cache volume check** — `vllm-hf-cache` was EMPTY → prior PP2 runs also served from a local mount → mount path is not the variable.
3. **PP2 smoke, IDENTICAL mount + image + env** (`device=2,3`, partition 32,16, util 0.78) → loads clean at ~2 s/shard → failure isolated to rank COUNT, not files/image/env/author patches.
4. **patch_pp.py review** — all 6 sites are rank-parity checks (`is_first_rank`/`is_last_rank`), none touch the loader, no hardcoded 2s → not the fork's patches. Failing code is stock vLLM `safetensors_weights_iterator`.

Verdict: fixing PP4 = surgery inside the vLLM fork's stock loader (likely torchao-flattened-tensor or concurrent-mmap interaction under ≥4 readers). Unbounded; not attempted. **The fork's practical ceiling is 2 GPU ranks until proven otherwise.**

## PP2 smoke @ util 0.78 — third failure mode, same root cause

Died ~7 min: `ValueError: No available memory for the cache blocks` (CPU-backend reservation; VllmWorker-1 = 40G card). PP2 on 104G now has THREE distinct death signatures — partition OOM, capture-phase NCCL watchdog, cache-block exhaustion — all the same structural deficit. Don't re-test PP2 splits; the math is closed.

## PP3 — the one-3090 config (the operator's stepwise preference: add one card, measure, then scale)

128G = exactly the author's proven 2×64G capacity. **Rank placement rule refined: the MTP-drafter tail rank goes on the 40G card; the clean 3090 (GPU1) takes a light mid rank; the taxed 3090 (GPU0, 8090 lane) stays OUT entirely.**

- Devices `2,1,3` → rank0=64G (26L), rank1=clean 3090 (8L), rank2=40G (14L + drafter)
- `VLLM_PP_LAYER_PARTITION=26,8,14`, util **0.85** (0.78 starves the tail rank's cache blocks), seqs 2, len 32768, MTP-3, no async
- Launcher: `~/pp3-launch.sh` (tears down smoke container, launches `qwen38-pp3-bench` on :8003)
- Launched 8/29 17:53 PDT, watcher armed (proc session), **verdict pending at skill-write time**

Decision ladder given to the operator: UP + ≥50 t/s → one 3090 suffices, plan migration; UP + <50 t/s → AWQ closes with a measured ceiling; loader crash same as PP4 → fork ceiling 2 ranks, AWQ waits for a 2×64G day.

## Next-session checklist (if PP3 verdict unresolved)

1. `docker ps -a --filter name=qwen38-pp3-bench` + `docker logs qwen38-pp3-bench | grep -E 'KV cache size|Maximum concurrency|startup complete|Error'`
2. If it served: decode numbers should be in the transcript/Discord; else bench :8003 (`Bearer change-me`, nonce'd prompts, temperature 0)
3. Restore prod either way: kill bench by exact PID (compute-apps), `~/scripts/serve-flashnext-8012.sh` (~160 s to healthy), resume `8012-nightly-watchdog` cron
4. Record the number here + SKILL.md verdict line; memory update for the final call
