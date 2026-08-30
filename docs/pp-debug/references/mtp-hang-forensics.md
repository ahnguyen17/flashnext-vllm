# MTP Hang Forensics — FlashNext-AWQ PP3 campaign (2026-08-30, rounds 2–3)

## Build ledger (container `qwen38-pp3-bench`, port 8003)

| Build | Image | Change vs prev | Verdict |
|---|---|---|---|
| v17 | pp3fix6 | +7200s RPC timeout (sites 11/12 still in) | hang inside site-12 port at :631; "200 OK"s were client-disconnect artifacts, 1 token lifetime |
| v18 | pp3fix4 (NO ports 11/12) | reverted kernel ports | same hang at ORIGINAL :627 in `cuModuleLoadData` — ports exonerated for the hang |
| v19 | pp3fix7 | +site-13 per-rank TRITON_CACHE_DIR, fresh cache volume | identical hang — cross-arch cache-poisoning theory dead |
| v20 | pp3fix7 | **no `--speculative-config`** | **WORKS: 57–59 t/s single, ~105 t/s 2-conc, TTFT 0.3s** — MTP path is the disease carrier |
| v21 | pp3fix8 | +site-15 pure-torch post_update (non-last ranks, bit-exact) | hang MOVES to `_post_update_pure_torch:187` (first implicit sync) → forward never completes → spinner is upstream |
| v22 | pp3fix8 | dropped fossil `VLLM_GDN_DECODE_KERNEL=triton` | hang IDENTICAL; boot log still `GDN decode kernel: triton` — cuda op not built, silent fallback; GDN-env theory dead |
| v23 | pp3fix8 | +`--enforce-eager` (no CUDA graphs, no torch.compile) | hang identical — graph/compile theories dead |
| v24 | pp3fix8 | +600s watchdogs + `NCCL_DEBUG=INFO` + `TORCH_NCCL_TRACE_BUFFER_SIZE=2048` + `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | PENDING at write time — comms-deadlock naming test (`~/pp24-launch.sh`) |

## Stack signatures (all PP0 = physical GPU1, 3090, sm_86)
- v10–v19: `_init_handles (triton/compiler/compiler.py:469)` ← `post_update (input_batch.py:627)`; native: 11 anon libcuda frames ending in `cuModuleLoadData`; 100% SM, ~91% CPU, R state; no Xid
- v21+: `_post_update_pure_torch (model_runner.py:187)` = `vi = idx_mapping[valid].long()` — boolean-mask indexing = implicit sync → waiting on wedged stream
- PP1/PP2 always idle in `irecv_tensor_dict`/`recv` (starved downstream ranks); PP1's 100% GPU = NCCL spin-wait, normal

## Free isolation probes (all PASSED on the 3090 — scripts on host)
- `~/probe13.py`: plain compile+load+launch of `_post_update_kernel`, tensor variant — GPU0 & GPU1
- `~/probe14.py` (SCENARIO env): +CUDA-graph-bearing context, +triton launch hooks
- `~/probe15.py`: +19GB expandable-segments reservation, fragmented (every other GB freed), graphs+hooks
- `~/probe16.py`: **None specialization** (non-last-rank variant — what non-last ranks actually load)
- `~/probe17.py triton4`: `fused_recurrent_gated_delta_rule_packed_decode` at T=4, one shared state slot, indices [0,0,0,0] — 2.26s OK (out was all-zeros on random inputs; ran is what matters)
- `~/probe18.py`: `causal_conv1d_update` spec-mode (R=1, 4 accepted tokens, qsl=[0,4], max_query_len=4, state_len=8) — 2.0s OK
- `~/equiv15.py`: bit-exact triton-vs-pure-torch equivalence, masked + all-valid cases

Probe-building pitfalls hit:
- **Derive shapes from the kernel's own validator**: `fused_recurrent.py` infers HV,V,K from `initial_state.shape[-3:]` — first attempt with Hk=16 heads in state failed validation before testing anything (expected out was (4,1,16,128) instead of (4,1,48,128))
- `exec(open(f).read())` breaks `@triton.jit` definitions (inspect needs a real file) — probe code must be docker cp'd as a file
- Two `timeout N docker run … ; echo exit:$?` — the wrapper always exits 0; check the inner exit line, not the shell's

## The reasoning chain (as it actually unfolded)
1. v10's warmup emitted 64 tok in 301.8s ≈ default RPC timeout 300s → timeout-race theory (correct for v10's REQ2 500s; the 7200s env is kept in all launches)
2. v17 fake-successes (200 OK on disconnect) exposed via `/metrics` `generation_tokens_total=1.0`
3. v18/v19 killed the port-regression and cache-poisoning theories
4. v20 isolated the disease to the MTP request path
5. v21's pure-torch substitution moved the hang to the first sync — the module-load "hang" was never the load
6. v22: env archaeology found the fossil `VLLM_GDN_DECODE_KERNEL=triton` override; dropping it changed nothing (cuda op absent → silent triton fallback)
7. v23: eager mode → graphs and torch.compile eliminated
8. probe17/probe18: GDN recurrent + spec-mode conv kernels pass standalone at exact spec shapes → every kernel-level theory exhausted
9. Remaining unprobed delta: multi-process comms. Hypothesis: **spec-step communication deadlock** — MTP adds reverse-direction draft traffic that no-MTP never generates; NCCL send-kernel spin = 100% SM with zero progress; PP1/PP2 blocked in recv; everything works in single-process isolation. v24's fast watchdog + NCCL trace dump is designed to NAME the stuck collective.
10. Fallback if comms ruled out: layer-partition bisect (4/32/12 → narrow) — first serving config wins, `--max-model-len 16384` if PP1 (64G) hits cache-block OOM at 32 layers

## Spec-path code map (pp3fix8, /opt/vllm)
- `vllm/v1/worker/gpu/model_runner.py` — `update_pp_decode_requests` (:1062) → `postprocess_sampled` (:1507); called early in `execute_model` (:1544) with sampler outputs from pp_size steps ago (`pp_handler.get_prev_sampled_outputs`)
- `vllm/v1/worker/gpu/input_batch.py` — `_post_update_kernel` (:544); wrapper `post_update` (:607). Per-req semantics: skip idx<0; last_sampled = sampled[ns-1]; total_len += ns; append ns tokens at OLD total_len; num_computed += qlen − num_rejected (qsl None → 0)
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` — dispatch `_forward_core` (:1244); non-spec → `_forward_core_decode_non_spec` (:1632) → triton `fused_recurrent_gated_delta_rule_packed_decode` (:1672); spec → `_forward_core_decode_spec_fused_norm` (:1686) → spec-mode `causal_conv1d_update` → `ops.fused_gdn_decode_post_conv_mtp` — **op NOT BUILT** (`vllm/_custom_ops.py:2780`, wrapper has no fallback) → engine's actual spec route past :1400 unverified; find where it really goes before round 3 ends
- `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` — `causal_conv1d_update` (:1096); spec mode needs `state_len ≥ width−1 + max_query_len−1` (=6); grid (batch, cdiv(dim, BLOCK_N))
- `vllm/envs.py` — `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` default 300, applies to BOTH execute_model and sample_tokens RPCs

## Model dims (for probe inputs)
Hk=16, Hv=48, D=128 → key_dim 2048, value_dim 6144, qkv row 10240; conv width 4; recurrent state fp32 `[slots, 48, 128, 128]`; a/b/A_log/dt_bias sized to Hv=48

## v10 anomaly
One 64-token spec success at 301.8s — never reproduced. Don't build theories on it; note and move on.

## Ops state at last write (~00:15)
- 8012 prod DOWN **by explicit user mandate** ("not using downstream consumers at the moment. I want the bug to be fixed") — do NOT auto-restore mid-campaign; ask first
- Watchdog cron `5776c61d67ad` PAUSED — resume after final restore
- v24 container up since 00:12, watcher → `~/pp24-watch.log`
- Restore when done: `~/scripts/serve-flashnext-8012.sh` (~165s to healthy), then `cronjob resume 5776c61d67ad`
