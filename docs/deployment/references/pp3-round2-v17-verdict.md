# v17 Live-Fire Verdict — Timeout Env + Kernel Ports = Livelock; Sites 11/12 Retracted

Night of 8/30, round two ("go for round two and finish the fix"). Supersedes the optimistic framing in `pp3-round2-timeout-race.md`, which ended at v17's launch.

## What ran

v17 = v14 shape verbatim (devices `1,2,3` PCI-ascending: rank0=clean 3090, rank1=170HX-64G, rank2=170HX-40G+MTP drafter; `VLLM_PP_LAYER_PARTITION=8,28,12` @ util 0.85; 32K ctx; seqs 2; MTP-3; both distributed timeouts 3600) on image **pp3fix6 (sites 7-12)** plus the round-2 discovery: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`. Launched 18:55 via `~/pp17-launch.sh`; watcher `~/pp17-watch.sh` (boot wait, 6-shape soak, 1500s client ceiling per request, background py-spy dumper on failure trigger).

## Result: FAILED — and it falsified "the kernel ports are innocent"

- Boot took ~30 min (slower than the usual 8-11, but clean): API up, CUDA graphs captured PIECEWISE + FULL, KV pool healthy.
- Two `POST /v1/completions 200 OK` lines appeared in docker logs — **FAKE**: uvicorn logs 200 when the handler ends, including client-disconnect. The engine's lifetime counters never moved past `generation_tokens_total=1.0` / `prompt_tokens_total=29.0`.
- soak1-short burned its FULL 1500s client timeout; EngineCore sat idle in `dequeue -> get_response` (multiproc_executor.py:430) the whole time — waiting on a worker response that never came.
- **PP0 (the 3090) livelocked at `post_update (input_batch.py:631)` — inside site-12's pure-torch port.** Verified by the frozen-line test: 3 py-spy dumps 20s apart, identical frame each time, process in R state at 91.6% CPU with GPU at 100% SM. High CPU + high SM is NOT proof of progress.
- Requests that did "finish" completed with exactly 1 token — premature finalization: the scheduler closes the request while the worker grinds on phantom work.

## Corrected causal model

| Build | Kernel ports (11/12) | Request path |
|---|---|---|
| v10 / pp3fix4 | absent | **WORKED** — 64 tok in 301.8s; later requests died only on the 300s-RPC-ceiling vs ~300s-init race |
| v13-v17 / pp3fix5-6 | present | exactly 1 token, then hard livelock inside the port |

Sites 11/12 were symptom-patches for what was actually a timeout race, and they introduced two real bugs: the `post_update` port livelock on the 3090 rank, and premature request finalization. **"Correct but symptom-level" is RETRACTED.**

## v18 — the clean untried combination

**Image `qwen38-flash-next:pp3fix4` (sites 8-10 ONLY) + `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`, everything else v14-verbatim.** Rationale: v10 proved the unported request path generates through all three ranks + MTP drafter; the only thing that killed v10 was the 300s RPC ceiling vs the ~300s one-time-per-specialization triton module-load on the 3090. Raise only the ceiling and every cold init completes.

- Scaffold: copy `~/pp17-launch.sh` to `~/pp18-launch.sh`, swap the image tag to `pp3fix4` (the write was cancelled mid-turn when the operator interrupted — recreate rather than assuming it exists).
- Bench ladder: warmup 64-tok (absorbs the first ~300s init), a same-shape repeat (should be FAST — proves the warm path), then decode_256 and decode_sustained_512. Per-request client timeout 1800s+.
- **Never reuse pp3fix5/pp3fix6 for a serving build without excising sites 11/12.** They remain useful only as evidence of the regression.

## Verification toolkit — dead vs slow vs fake-success (all proved tonight)

1. **py-spy frozen-line test**: dump the SAME pid 3x ~20s apart. Identical frame = livelock. Moving line = slow progress. This is the discriminator the CPU-jiffies probe (v9) cannot make — jiffies say "alive", the frozen line says "alive but looping".
2. **`/metrics` completion oracle**: `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` advance ONLY on request completion. Static counters + requests nominally "in flight" = nothing is being generated. Cheapest dead-vs-alive check; no process attach needed.
3. **Fake 200-OKs**: uvicorn's `POST ... 200 OK` fires on client-disconnect too. Never count log lines as completions — cross-check the token counters.
4. **Buffered-watcher blindness**: `bash watch.sh 2>&1` as a background process surfaced 0 output lines for ~50 min despite `flush=True` prints; everything flushed only at kill (the deathbed output carried the real verdict: `soak1-short elapsed 1546s | worst so far 1501s`). During a live run, ground truth = `docker logs` + `/metrics` + py-spy — and ALWAYS tee watcher output to a file so tailing works.

## Ops gotchas sharpened

- **cronjob tool**: `action='update'` (even rewriting the prompt text) does NOT pause a job — state stays `scheduled` and armed (bit at 18:52: the 8012 watchdog stayed armed for its 19:00 tick mid-window). Follow with `action='pause'` and VERIFY `"state": "paused"` in the response before starting GPU work the watchdog would fight.
- Killing a stuck watcher is safe and useful: its buffered stdout flushes on death and often contains the verdict you couldn't see while it ran.

## Ops state at this write (8/30 ~19:50)

- v17 container `qwen38-pp3-bench`: still up, stuck, awaiting teardown.
- prod 8012: DOWN (window open). Restore = `bash ~/scripts/serve-flashnext-8012.sh`, verify 200 + a 300+tok generation (thinking models burn small max_tokens on reasoning_content), then resume cron `5776c61d67ad`.
- `8012-nightly-watchdog`: PAUSED 18:53.
- studio-director (8090 lane, GPU0): up, untouched all round (v6+ configs never used GPU0).
- the operator interrupted mid-teardown; v18 go/no-go pending his reply. Recommendation on the table: proceed with v18.