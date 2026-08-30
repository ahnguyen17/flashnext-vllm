# PP3 endgame: first full boot + request-path triton fix (8/29 late night)

Continues `pp3-final-verdict.md` (ends at v9/pp3fix3 with PP1 "alive" in a JIT grind).
Lineage: `pp2` → `pp2v2` (+site-7 PLE worker) → `pp3fix` (+site-8) → `pp3fix2` (+site-9, entrypoint trap) → `pp3fix3` (guards sed-fixed to `.world_size`) → `pp3fix4` (+site-10) → `pp3fix5` (+site-11).

## v9 dead end — flat-cache spin discriminates grinding from looping

Two liveness signals, only one is truth:
- CPU jiffies `/proc/PID/stat` fields 14+15: delta ~3000/30s = one core busy ("alive")
- triton cache file count: `find /root/.triton -type f | wc -l` two samples 60s apart

v9's PP1: jiffies alive BUT cache +0/min → **re-init loop, not compilation**. A compiler
that ships no artifacts never finishes. Full-depth py-spy then placed BOTH PP0 and PP1 at
`warmup.py:358` — sampler bookkeeping (`post_update_num_computed_tokens` under
`sample_tokens`) immediately after the prefill warmup — BEFORE the decode loop site-9 skips.
PP2 (12 layers) had passed the same code: shape-luck.

## Site-10 (pp3fix4)

Skip the whole `warmup_kernels` body at PP>1, right after the `is_encoder_only` early return:

```python
try:
    from vllm.distributed.parallel_state import get_pp_group as _gpp10
    if _gpp10().world_size > 1:
        logger.info("site-10: skipping kernel warmup at PP>1")
        import torch as _t10
        _t10.accelerator.synchronize()
        return
except Exception:
    pass
```

Consequence: first request pays ALL cold JIT. Measured: 64-token first request = **305.3 s**
(mostly JIT, streamed to completion). Bench design must include a throwaway warmup request
before measured cases.

## v10 — the boot that ended the "can it even start" question

23:51:37 UTC 8/29: `Application startup complete`. KV 272,286 tokens, 8.31× concurrency
@32K/req. Request #1 completed (above). Request #2 (`ignore_eos`, 256 tok, temp 0):

```
TimeoutError: RPC call to sample_tokens timed out.
  ← shm_broadcast dequeue timeout in EngineCore
  ← Worker never returned from sample_tokens RPC
```

APIServer then logged EngineDeadError and shut the container down cleanly.

## The unified root cause

Every silent wedge tonight — warmup hangs (v3–v9), the request-2 death — is ONE disease:
**triton runtime `_init_handles` spin (100% CPU, flat cache, never returns) for
`_post_update_num_computed_tokens_kernel` on certain dynamic shapes on this host.**
The kernel itself (input_batch.py:~646) is a trivial scatter-add:

```python
@triton.jit
def _post_update_num_computed_tokens_kernel(idx_mapping_ptr, num_computed_tokens_ptr, query_start_loc_ptr):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + (query_end - query_start))
```

## Site-11 (pp3fix5) — pure-torch replacement

Replace the launch body of `post_update_num_computed_tokens(...)` in
`vllm/v1/worker/gpu/input_batch.py`:

```python
def post_update_num_computed_tokens(idx_mapping, num_computed_tokens, query_start_loc) -> None:
    # site-11 (pp3fix5): triton _init_handles wedges at 100% CPU forever on this
    # host (py-spy: warmup AND live request path -> sample_tokens RPC timeout).
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    query_len = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    num_computed_tokens.scatter_add_(0, idx_mapping.long(), query_len)
```

Same op (per-batch indexed increment), scatter_add_ handles duplicate indices safely,
stays on-GPU, called once per step — negligible cost.

## If it recurs elsewhere

Signature: `TimeoutError: RPC call to <method> timed out` → the named method's worker-side
stack (py-spy) shows a triton kernel entry → substitute the kernel with a pure-torch
equivalent. Do NOT chase OOM/watchdog theories first — this disease mimics both.

## Config that finally booted (v10/v11 — v4 config + patches)

devices `1,2,3` (PCI-ascending: 3090-clean → 64G → 40G+drafter), partition `8,28,12`,
util 0.85, seqs 2, `--max-model-len 32768`, `--distributed-timeout-seconds 3600`,
`--cpu-distributed-timeout-seconds 3600`, MTP-3, PLE offload on, compile-cache volume
`vllm-cache` mounted (carries ALL prior compile progress — boot ~8 min warm).
