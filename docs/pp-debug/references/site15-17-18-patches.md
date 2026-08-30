# Sites 15 / 17 / 18 — the MTP ring deadlock fix (2026-08-30)

Patch scripts of record live at `~/site15.py`, `~/site17.py`, `~/site18.py` (host).
Apply pattern: `docker run -d --entrypoint sleep <base_img> infinity` → `docker cp` script →
`docker exec python3 /tmp/sN.py` → verify with grep + `ast.parse` →
`docker commit --change 'ENTRYPOINT ["vllm"]' <ctr> qwen38-flash-next:<next>`.

## Site-15 — pure-torch post_update on non-last PP ranks (pp3fix8)
File: `vllm/v1/worker/gpu/model_runner.py`
- Module-level `_post_update_pure_torch(...)`: boolean-mask valid rows, last-sampled gather,
  ragged token scatter via `repeat_interleave` + within-block arange, `num_computed += query_len - num_rejected`.
  One host sync (`ns.sum().item()`) — acceptable per decode step.
- `postprocess_sampled()`: `if self.is_last_pp_rank:` keeps triton `post_update(...)`, else calls the port.
- MANDATORY gate: bit-exact equivalence test of both paths before deploying — build ONE input
  dict, `clone()` per arm (consecutive `make()` calls advance the RNG and fake a mismatch —
  this false-positived once). Test masked rows (-1 in idx_mapping), zero-sampled reqs, ragged appends.

## Site-17 — ring on main stream (pp3fix9)
File: `vllm/v1/worker/gpu/pp_utils.py`
- `receive()` / `broadcast()`: drop `with torch.cuda.stream(self.broadcast_stream)` +
  `wait_stream` + `record_stream` dances; run the two `torch.distributed.broadcast` calls on the
  main stream; record the PendingRecv event on main stream.
- Rationale: sibling-communicator + side-stream interleaving vs inter-stage p2p is a deadlock
  hazard class. NECESSARY but INSUFFICIENT — v25 hung identically; the real gate was site-18.

## Site-18 — unconditional ring participation (pp3fix10) ← THE FIX
File: `vllm/v1/worker/gpu/pp_utils.py`
- `receive()`: `compute_need_sampled_mask` returning None no longer skips — substitute
  `np.zeros(num_reqs, bool)` and ALWAYS enqueue the 2 broadcasts. Mask stays a per-request
  DATA filter at consume time (`get_prev_sampled_outputs` already filters).
- `broadcast()`: remove the early-return on mask None; ALWAYS broadcast; pad/truncate
  `sampled_token_ids` to `max_sample_len` with -1 (consumers skip -1) so numel matches the
  receivers' `[num_reqs, max_sample_len]` buffer.
- Principle: **collective participation must be unconditionally symmetric across ranks;
  rank-local state (pipeline-lagged `num_computed_tokens_np`) must never gate whether a
  collective is issued — only what its payload means.**

## Verification chain that led here
1. v20 (no-MTP) served → bug isolated to spec path.
2. v21 (site-15): hang MOVED to the port's first sync → disease upstream of post_update.
3. v22 (drop GDN env): silent triton fallback, identical hang → env archaeology dead end.
4. v23 (--enforce-eager): identical hang → graphs/compile exonerated.
5. v24 (600s watchdog + NCCL dump): **named BROADCAST SeqNum=1 NumelIn=4 on pp_broadcast,
   ranks 0/1 enqueued-never-started, rank 2 absent** → comms deadlock, skip-path suspected.
6. v25 (site-17 main-stream ring): identical → interleaving wasn't the gate.
7. Static read of PPHandler: only skip path = mask-based early return on rank-local state
   → site-18. v26 = decisive boot.

## If v26 still hangs
- Check the NCCL dump again: a DIFFERENT collective/SeqNum = progress; same = the skip has
  another gate (audit `sample_tokens` last-rank path for other early returns).
- Layer-partition bisect (4/32/12, `--max-model-len 16384` if PP1 cache-block OOMs at 32 layers)
  distinguishes placement vs comms — but comms now proven, so weight accordingly.
- The mask's rank-local inputs (`num_computed_tokens_np`) can be made scheduler-derived
  (identical on all ranks) if pad-filtering proves insufficient.
