# Round 2 endgame — v19 falsifies cache poisoning, v20 no-MTP WORKS (8/30 night)

Continues `pp3-round2-v18-cache-poisoning.md` (v19 armed at that write).

## v19 verdict — cache theory DEAD

pp3fix7 (site-13 per-rank `TRITON_CACHE_DIR`, confirmed executing — worker stack line numbers shifted by exactly the +8 patch lines) + fresh `vllm-cache-v19` volume + `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`. Booted ~13 min, then warm-64 froze for the full 1800s client ceiling at the same frame as always: PP0 `_init_handles (triton/compiler/compiler.py:469)` under `post_update (input_batch.py:627)` — the ORIGINAL kernel, no ports, own empty cache dir. The 3090 compiled its own kernel and still hung the LOAD.

Retracted: shared-cache cross-arch cubin poisoning as the root cause. Site-13 stays in the image as hygiene (mixed sm_80/sm_86 rigs sharing one cache dir is still a real upstream footgun).

## Native stack + GPU state

`py-spy dump --pid <PP0> --native`:
```
0x… (libcuda.so.610.43.03)   × 11 anonymous frames
cuModuleLoadData (libcuda.so.610.43.03)
loadBinary (cuda_utils.cpython-312-…so)
_init_handles (triton/compiler/compiler.py:469)
launch_metadata (triton/compiler/compiler.py:485)
run (triton/runtime/jit.py:760)
post_update (vllm/v1/worker/gpu/input_batch.py:627)
```
GPU1 (the PP0 card): 100% util at 1950 MHz SM clock — the driver is actively spinning, not blocked in a syscall. `dmesg` clean, zero Xid. `ps`: worker in R state; note `pcpu` is a LIFETIME average (91.6% ≠ current).

## Isolation probes — exonerate everything individually

Generalized pattern (script: `scripts/kernel-load-isolation-probe.py`): run the EXACT kernel from the crash stack standalone in the same image on an idle GPU, timed:

```
docker run --rm --gpus '"device=0"' --entrypoint python3 \
  -v /path/probe.py:/probe.py -e TRITON_CACHE_DIR=/tmp/tc-probe \
  <vllm-image> /probe.py
```

Results on an idle 3090 (GPU0, studio-director's card):
- compile+load+launch `_post_update_kernel` with decode-shaped dummies: **1.98s** (second call 0.0001s)
- same under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + live 4GB VMM alloc: **2.12s**

⇒ Kernel innocent, silicon innocent, driver CAN load it, VMM not the trigger. The hang needs the full engine context (multi-rank pipeline, CUDA graphs, PLE offload worker) and/or physical GPU1 — every failing run used GPU1 as PP0; probes ran on GPU0. Un-tested-at-close: same probe on GPU1 with the engine down (queued for round 3).

## v20 — the working server

Config = v19 verbatim MINUS `--speculative-config` (no MTP ⇒ `post_update` never runs). Image pp3fix7, devices 1,2,3, partition 8,28,12, util 0.85, 32K, seqs 2, both distributed timeouts 3600, env 7200. Launcher `~/pp20-launch.sh`, container `qwen38-pp3-bench`, port 8003, key `change-me`, volume `vllm-cache-v19`.

| Case | Result |
|---|---|
| warm-64 (cold) | 64 tok, ttft 8.1s, 32.7 t/s |
| warm-64 repeat | 63 tok, ttft 0.3s, 59.5 t/s |
| decode-256 ignore_eos | 57.8 t/s |
| decode-512 ignore_eos | 57.1 t/s sustained |
| 1024-tok long | 48.1 t/s |
| 2 concurrent 512s | 52.3 + 53.4 ≈ 105 t/s aggregate |

**Beats the ≥50 t/s bar with zero MTP; 1.6× the GGUF lane's 36 t/s on a 6.5× bigger model.**

## Final bug scope (for the upstream issue / round 3)

- Sole casualty: triton module-load of `_post_update_kernel` (MTP spec-decode bookkeeping, `input_batch.py`) spins forever inside `cuModuleLoadData` — only on the PP0 3090, only in full engine context, deterministic across ≥6 runs (v12–v19), no error, no Xid.
- Standing evidence: py-spy frozen-line ×3, native libcuda stack, isolation exonerations, v10's one 301.8s success (nondeterministic completion), no-MTP control serving flawlessly.
- Round-3 levers: probe on GPU1 (engine down) → if it hangs there alone, it's the CARD not the context; unpatched-driver test (nukes 170HX unlock — full-outage decision for the operator); 2×64G day (drops 3090s, author's proven shape); upstream issue (all evidence collected, offer standing).

## Artifacts on disk (kept)

- Image `qwen38-flash-next:pp3fix7` = sites 8-10 + 13 (the serving image). pp3fix5/pp3fix6 carry the RETRACTED sites 11/12 — never serve from them.
- Volume `vllm-cache-v19` (warm per-rank triton caches — speeds re-boots), `vllm-hf-cache`.
- `~/pp20-launch.sh` (the recipe), `~/pp18-watch.sh` (unbuffered ladder watcher, tee to file), `~/probe13.py` (session-local probe; generalized copy in this skill's scripts/).
- Model: `~/models/FlashNext-AWQ` 174G verified.

## Ops close-out (verified)

v20 torn down → `~/scripts/serve-flashnext-8012.sh` → 8012 health 200 in 165s + gen test OK → cron `5776c61d67ad` (8012-nightly-watchdog) resumed. studio-director untouched throughout (v6+ configs never touch GPU0). Placement defaulted to "restore 8012" after a 10-min clarify silence — production-safe default; AWQ re-serves with one command.
