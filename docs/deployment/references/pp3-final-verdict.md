# PP3 v2–v5 final ladder — 8/29 night session (closes the AWQ PP question)

Outcome: **capacity solved at 128G; fork's MTP speculator capture deadlocks at PP>1; AWQ parked.**
Applies to `qwen38-flash-next:pp2v2` image + `~/models/FlashNext-AWQ` + any PP>1 launch of the
`vektorprime/qwen38-flash-next-pp2` recipe on this rig.

## The two root-cause discoveries

### 1. Device-ordering scramble (the ">2-rank loader bug" that wasn't)
With `CUDA_DEVICE_ORDER=PCI_BUS_ID` (set by the recipe's launch.sh), docker `--gpus '"device=..."'`
lists are RENUMBERED ascending by PCI bus id. Board order is GPU0=03:00 (3090, carries the 8090
studio-director lane ~4.5G), GPU1=04:00 (3090 clean), GPU2=81:00 (170HX 64G), GPU3=82:00 (170HX 40G).
- `--gpus 2,0,1,3` (PP4 intent: rank0=64G) → actual: rank0=taxed 3090 → 20 layers (32G) OOM in ~2 min.
- `--gpus 2,1,3` (PP3 intent: rank0=64G,26L) → actual: rank0=clean 3090,26L → `torch.OutOfMemoryError ... GPU 0 has a total capacity of 23.56 GiB` at ~1 min.
- Sibling ranks die ~6s later with `ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'` at `weight_utils.py:970 safetensors_weights_iterator` — **teardown noise, not the cause.** The 38-shard/every-tensor single-proc sweep was clean and PP2 loaded the same mount: the loader was never broken.
- **THE TELL:** an OOM mentioning a card size that doesn't match rank0's intended card ⇒ verify mapping before touching anything else. `nvidia-smi --query-compute-apps` during load shows which bus each rank actually landed on.
- **RULE: `--gpus` lists must be PCI-ascending (`1,2,3` etc.); encode the desired rank order in `VLLM_PP_LAYER_PARTITION` instead** (rank N takes partition[N]).

### 2. Gloo vs NCCL timeout knobs are SEPARATE
- `--distributed-timeout-seconds` raises only device/NCCL process-group timeouts.
- The gloo (CPU control-plane) PG keeps its 1800s default ⇒ **silent death at exactly 30:00 into any
  stall** (`gloo/transport/tcp/unbound_buffer.cc:78 Timed out waiting 1800000ms`). Raise BOTH:
  `--distributed-timeout-seconds 3600 --cpu-distributed-timeout-seconds 3600`.
- Prior signature: default 10-min NCCL watchdog kills peers ~11 min into a stalled capture
  (ProcessGroupNCCL dump → SIGABRT on the healthy rank). A watchdog death means "slow or hung",
  not which.

## The attempts (all PP=3 unless noted, image pp2v2, `--entrypoint vllm`, PLE CPU offload)

| ver | devices / partition / util | outcome |
|---|---|---|
| PP4 | `2,0,1,3` / 20,8,8,12 / 0.78 | rank0→taxed 3090, instant OOM cascade (UntypedStorage noise) |
| v1 | `2,1,3` / 26,8,14 / 0.85 | rank0→3090 OOM at ~1 min (THE TELL line) |
| v2 | `1,2,3` / 8,26,14 / 0.85 | **ALL phases pass**: weights, PLE table, KV 59,261 tok (1.81× @32K), rank0 graphs 5/5+2/2 → NCCL 10-min watchdog kills peers 11 min into rank2 speculator capture |
| v3 | v2 + `--distributed-timeout-seconds 3600` | died at exactly 30:00 — gloo 1800s fuse (knob #2 discovered) |
| v4 | `1,2,3` / 8,28,12 / 0.85 | KV 269,165 tok (8.21× @32K), tail ~9.5G free → **identical 30:07 gloo death ⇒ deadlock is memory-independent** |
| v5 | `0,1,2` / 10,11,27 / 0.86 (both timeouts 3600; studio-director stopped) | rank1 `No available memory for the cache blocks` ~10 min — capacity arithmetic, never reached speculator |

## The fork bug (upstream-issue material)
MTP speculator capture (`speculator.py:148 Capturing model for speculator...` on the LAST rank)
hangs with zero log progress for ≥30 min at PP>1, then whichever fuse fires kills the engine.
Proven memory-independent: v3 (5.5G tail free) and v4 (9.5G tail free, KV 4.5× larger) died
identically. Rank0 completes all its CUDA-graph capture normally while rank2 never returns from
the speculator capture. Every other boot phase (weights, 95G PLE table in host RAM, KV sizing,
rank0 FULL+PIECEWISE graphs) works at 3 ranks. Author's verified config is PP2 on 2×64G —
possibly only with a warm compile cache. Evidence bundle: v3/v4 docker logs (timestamps
18:05:59→18:36 / 19:13:18→19:43:25), partition + free-memory numbers above.

## Capacity arithmetic that worked (AWQ-INT4, 48 layers, observed)
- ~1.6 GiB per layer weights; last-rank fixed overhead (drafter + lm_head + norms) ≈ 8.5–10.2G.
- Read per-rank truth from the `Free memory on device (X/Y GiB) ... Actual usage is ... Current kv cache memory in use is Z` startup lines — engine auto-shrinks KV to fit; small KV number = thin rank.
- 3090 practical cap ≈ 10–11 layers at util 0.85–0.86 (KV-profiling activations exceed a 2G estimate; 11L died, 10L fits) — less if the card hosts another lane (one global util flag: the most-taxed GPU sets the ceiling).
- 40G tail: 14L + drafter works (KV shrinks to ~1.3G); 12L tail frees KV dramatically (269K tok).
- 64G tail: 27–28L + drafter ≈ 99% of 0.86×64 — razor, cache-blocks OOM territory.
- Topology matrix: 64+40 (104G) unfittable for ANY servable PP2 config; +1 clean 3090 (128G) fits with headroom; 3090+3090+64G (112G, drafter on 64G) arithmetic-closed only at >99% util — rejected.

## Ops notes that saved the night
- Pause `8012-nightly-watchdog` cron BEFORE killing prod; resume after verified restore. Watcher
  scripts: background terminal + `notify_on_complete`, 60s poll loop with crash-pattern greps
  (include `Timed out waiting`), container-exit check, auto-bench on HTTP 200.
- Kill prod by exact PID from `nvidia-smi --query-compute-apps` — broad pkill also kills the 8090
  lane (`/app/llama-server`, container `studio-director`, evictable via `docker stop`, reversible).
- Late notifications from watchers armed in an earlier (compacted) session describe PAST events —
  verify current state (`docker ps`, compute-apps) before reacting; one described my own cleanup.
- Restore: `bash ~/scripts/serve-flashnext-8012.sh` (~160s to 200), gen check with max_tokens ≥200
  (thinking models return empty content at 10).

---

## UPDATE 8/29 late-night: root cause FOUND + FIXED — the "fork bug" was the warmup, and the speculator capture was innocent

Reopened on the operator's "make it work". Reproduced the v4 hang (identical config) and dumped live stacks
~3 min into the stall — `docker exec qwen38-pp3-bench bash -c 'pip install -q py-spy; py-spy dump --pid <PID>'`
(PIDs from `ps -eo pid,args | grep VLLM::` → EngineCore / Worker_PP0/1/2 + PleOffloadWorker).

**The stacks:**
- `Worker_PP2` (hung): `recv (distributed_c10d.py:2799) ← recv_object ← irecv_tensor_dict (parallel_state.py:1148) ← execute_model (gpu_worker.py:1217) ← _run_decode_step (warmup.py:390) ← warmup_kernels (warmup.py:421)`
- `Worker_PP0` + `Worker_PP1` (both ACTIVE, not blocked): triton `_init_handles` in `postprocess_num_computed_tokens` — mid-warmup, progressing fine
- `EngineCore`: healthily waiting on the collective_rpc response

**Mechanism:** after CUDA-graph capture, vLLM's kernel warmup replays `execute_model` locally on
EVERY rank (warmup.py builds a ladder of decode steps: spec / non-spec / mixed batch shapes). Under
PP>1, `execute_model` on non-first ranks starts with a REAL pipeline `irecv_tensor_dict`. The
spec-decode warmup steps run **different forward counts per rank** — the drafter's extra passes on
the last rank — so rank2's receive count exceeds rank1's send count and rank2 blocks in `recv`
forever. Silent (no log line after `speculator.py:148`'s successor phase), memory-independent
(v3 5.5G tail vs v4 9.5G tail, identical 30:07 death), PP>1-only. Also explains the author's PP2
"verified" claim: a warm compile cache shortens/skips exactly the stall-prone replay phases.

**The fix — site-8 (`warmup.py`, 8 lines after `num_spec_steps = model_runner.num_speculative_steps`):**
```python
try:
    from vllm.distributed.parallel_state import get_pp_group as _gpp
    if _gpp().pp_size > 1:
        num_spec_steps = 0
except Exception:
    pass
```
Warm the non-spec paths only (1 forward per step on every rank → sends/recvs match 1:1); runtime
spec decode is scheduler-coordinated and untouched. Cost: drafter kernels cold until first real
request (one-time JIT seconds) — acceptable.

**Build gotcha (bit twice tonight):** `docker commit` of a container that RAN a fix script bakes
`ENTRYPOINT python3 / CMD /fix.py` into the image → `serve ...` becomes `python3 serve` → instant
`can't open file '/workspace/serve'`. Always `docker commit -c 'ENTRYPOINT ["vllm"]'` (or pass
`--entrypoint vllm` at launch).

**Meta-lesson (the highest-value takeaway):** a silent multi-rank hang whose only symptom is a
timeout fuse is a 5-minute py-spy away from the exact blocking line — dump stacks BEFORE
hypothesizing (two full attempts were spent theorizing about capture-phase VRAM that was fine).
Related: when sibling ranks show exotic errors seconds after one rank's plain OOM, the OOM is the
cause and the exotica are teardown noise — always find the chronologically FIRST error.

**v7 OUTCOME (site-8 alone is NOT sufficient):** the patch held — py-spy showed the hang moved
PAST the spec-flagged steps (line 390 → 400, non-spec) — but rank2 still starved in the identical
`irecv` while ranks 0/1 wedged at 100% SM (GPUs 1&2 pegged, rank2's card 0%) inside
`sample_tokens (model_runner.py:1806) → postprocess_num_computed_tokens → post_update_num_computed_tokens
(input_batch.py:670) → triton _init_handles`. Two dumps 60s apart at the SAME frame + 100% SM =
spin, not progress. So the count-divergence theory was incomplete: **the warmup decode-step
machinery does not converge multi-rank at PP3 at all** (spec or non-spec). Depth lesson: PP1's
first (shallow) dump showed generic triton frames; the true `sample_tokens` caller only appeared
at stack depth 14 — dump deep.

**The fix — site-9 (`warmup.py`, guard the decode-step loop):** skip the ENTIRE
`for step_indices, step_spec_flags in decode_steps: _run_decode_step(...)` loop when `pp_size > 1`
(keep prefill warmup — every run converged it; keep the cleanup forward — one matched
forward/rank). Decode kernels JIT cold on the first real request (seconds, once); correctness
untouched. Image **`qwen38-flash-next:pp3fix2`** = pp3fix + site-9.

**Entrypoint trap, third bite, new costume:** building pp3fix2 from a container run with
`--entrypoint sleep` and committing WITHOUT `-c 'ENTRYPOINT ["vllm"]'` baked `sleep` in →
`sleep: unrecognized option '--served-model-name'`, exit 1 at ~1 min with ZERO vLLM log lines.
That exact one-line signature = entrypoint clobber, not a config bug. **Recovery without redoing
the patch: the exited container's filesystem IS the patched image — `docker commit -c
'ENTRYPOINT ["vllm"]' <exited-container> <tag>` repairs it in place.** Rule: EVERY commit derived
from an `--entrypoint sleep`/`python3` build container must pass `-c 'ENTRYPOINT ["vllm"]'`.

**v8/v8b:** v8 died instantly on the entrypoint clobber (above); v8b = recommitted pp3fix2
(EP verified `vllm`, both patch markers grepped) with the v4 config (`1,2,3` / `8,28,12` / 0.85 /
32K / MTP-3 / both timeouts 3600) — **boot outcome PENDING at write time.** Next session:
`docker logs qwen38-pp3-bench | grep -E 'startup complete|KV cache'`; if up, the bench watcher
posts decode t/s (compare vs 36 t/s GGUF floor and the ≥50 bar); if it wedges AGAIN, py-spy first
(`scripts/py-spy-dump-ranks.sh qwen38-pp3-bench`) — remaining suspects are the cleanup forward
and `worker_sample_tokens` on non-last ranks (PP1 ran `sample_tokens` — a middle rank — which is
itself suspicious and unexplored).
