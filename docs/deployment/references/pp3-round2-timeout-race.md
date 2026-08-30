# PP3 Round 2 — the timeout race (8/30 early hours)

## What closed round 1

- **v15** — PP2 pure-170HX isolation control (devices `2,3`, partition `33,15`, pp3fix6, 32K, seqs 2, port 8003): died ~11 min at init, `No available memory for the cache blocks`.
- **v16** — maximum drafter relief (`34,14`, 16K ctx, seqs 1): identical init death.
- Conclusion: the 40G+drafter PP2 tail cannot fund KV at ANY split (29/19, 32/16, 33/15, 34/14 all failed) → the PP2-without-3090s control is unbootable; the 3090 correlation stayed a theory. Bench containers removed, images parked, prod restored and verified.

## The round-2 discovery

Trigger: the operator — "go for round two and finish the fix."

Re-read of round-1 evidence: **v10's warmup request SUCCEEDED in 301.8s.** Warmup durations across v10/v11: 300.7 / 301.8 / 305.3s. Deadlocks don't occasionally succeed at a constant duration.

Chain to the constant (the reusable recipe):

1. `docker run --rm --entrypoint bash <img> -c "grep -rn 'RPC call to' /opt/vllm/vllm"` → `v1/executor/multiproc_executor.py:432: raise TimeoutError(f"RPC call to {method} timed out.")`
2. The raise sits in `collective_rpc`'s `get_response` — dequeue timeout derived from the `timeout` param's deadline.
3. `sample_tokens` and `execute_model` (same file, ~:340–357) both pass `timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`.
4. `envs.py:239` (getter ~:1773): `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, default **"300"**.

Physics: the 3090 rank (sm_86 through the cmpunlocker-patched driver) pays **~300s per triton kernel-specialization `_init_handles` module-load**; 170HX ranks load ~instantly. Cost is one-time per specialization per process — v10's 64 tokens flowed at speed after the first ~300s step. The recipe author's rig (no 3090s, stock driver) never loses the race, which is why upstream never saw it.

Retractions vs round 1:
- "_init_handles spins forever" → it completes at ~300s. Every py-spy "wedge" was a mid-init snapshot (probe12 caught +255s; engine died ~+273s ≈ 300s after the RPC began).
- Sites 11/12 (pure-torch ports of the two bookkeeping kernels) were correct but symptom-level — keep them (cheap insurance, and genuinely better semantics), but they were never the cure.
- Unresolved: v10's same-shape REQ2 died despite REQ1's success → some request-to-request kernel axis differs (EOS vs ignore_eos path? sampler histogram shapes?). The v17 soak phase enumerates these empirically instead of theorizing.

## v17 (armed 18:55 8/29 PDT; verdict pending at this write)

- Image `qwen38-flash-next:pp3fix6` (sites 7–12), container `qwen38-pp3-bench`, port 8003.
- Config = v14 verbatim: devices `1,2,3` (PCI-ascending), `VLLM_PP_LAYER_PARTITION=8,28,12`, util 0.85, 32K, seqs 2, MTP-3, both distributed timeouts 3600, compile-cache + hf-cache volumes.
- **ONE delta: `--env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`.**
- Launcher `~/pp17-launch.sh`; watcher `~/pp17-watch.sh` (background, notify_on_complete): 20-min boot wait → **SOAK** (6 diverse shapes — short / ~350-tok medium / ~1.5K-tok long / no-EOS / two varied; 1500s client ceiling each; a py-spy trigger file `/tmp/v17-pyspy` dumps EngineCore+VllmWorker stacks inside the container on stall) → timed ladder (warm-64 / decode-256 / sustained-512).
- Success shape: first soaks ~300–600s, then collapse to seconds = race theory proven; the sustained-512 number vs the ≥50 t/s bar decides the AWQ book.
- Failure shape: identical 1-token death WITH a 7200s budget → race theory dead; escalate to the reserve lever or the upstream issue.

## Bench-window ops runbook (8/30 refinements)

- Pause the 8012 watchdog with cronjob **action='pause'** (job `5776c61d67ad`). GOTCHA: cronjob **action='update' with a "PAUSED…" prompt only rewrites the text — the job stays ENABLED and keeps firing** (bit me 8/30; always verify `state: "paused"` in the tool response).
- Prod down by exact PID from `ss -tlnp | grep ':8012'` (the `pkill -f` self-match footgun stands).
- studio-director (GPU0, port 8090) runs llama.cpp `server-cuda` — it exercises NO triton, so its health proves nothing about the 3090-triton theory. Leave it up; the bench uses devices 1,2,3 only.
- Restore: `bash ~/scripts/serve-flashnext-8012.sh` (background, ~160s to healthy) → verify `/health` 200 → cronjob action='resume' `5776c61d67ad` → sanity gen with max_tokens ≥ 300 (small budgets can return empty `content` on thinking models — not a failure).

## Reserve lever (untested)

Drop `CUDA_DEVICE_ORDER=PCI_BUS_ID` entirely: CUDA's default FASTEST_FIRST enumeration ranks the 170HXs ahead of the 3090s, so a `--gpus '"device=0,1,2"'` list can make **PP0 = 170HX-64 with the 3090 as the TAIL rank**. MUST probe per-index `torch.cuda.get_device_name(i)` inside the actual container first (tie-breaking between the two 170HXs is not guaranteed), then size `VLLM_PP_LAYER_PARTITION` to the probed order. Risk: the tail rank carries LM head + MTP drafter; a 24G 3090 tail is tight (v5 showed 3090s cap at ~10–11 layers at util 0.85–0.86). Only worth trying if the timeout raise doesn't cure the request path — it does not remove the 3090 from the pipe, it only moves it out of the PP0 seat.
