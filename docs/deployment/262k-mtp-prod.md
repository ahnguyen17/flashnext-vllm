# 262K + MTP-3: production promotion (2026-09-02)

The fourth prod lane for this model on this rig, and the first with speculative
decoding: **262,144-token native context + MTP k=3**, image `qwen38-flash-next:pp3fix22`
(sites 8–10, 13, 17–18, 23, 26), `--gpu-memory-utilization 0.85`, `--max-num-seqs 4`,
partition `8,28,12`. Promoted 2026-09-02 00:10, replacing the 786K YaRN no-MTP lane
after a ~20-minute swap window: one boot, full validation battery, zero failures.

Launcher: [`scripts/serve-vllm-pp3-262k-mtp.sh`](../../scripts/serve-vllm-pp3-262k-mtp.sh)

## The site-26 fix in one paragraph

Under pipeline parallelism the speculator's `propose()` runs only on the LAST rank,
but verification happens on every rank — ranks 0/1 read zeroed draft-token tables
(`[-1]` markers), producing the corrupted text and 0%-acceptance behavior that parked
MTP at the end of round 3. **Site-26 adds a third broadcast to the site-17/18 input
ring that distributes the draft-token table from the last rank to all ranks before
verification.** Patch: [`scripts/site26-pp-draft-table-sync.py`](../../scripts/site26-pp-draft-table-sync.py)
(touches `spec_decode/mtp.py` + `worker/model_runner.py`). No public recipe existed
for MTP-at-PP; this is the missing piece. First validated boot: 96.6 tok/s
single-stream, 52% acceptance, clean text.

## Why 262K + MTP over 786K no-MTP

The YaRN 3.0 lane proved 786K works (needle-clean to 737K retrieval) but decode at
depth is bandwidth-bound — ~10 tok/s at 538K filled. MTP multiplies decode throughput
everywhere it matters (1.5× warm; more at depth, where the accept-window amortizes
the attention sweep). Every ingredient was already proven separately, so nothing was
extrapolated:

- the 262K memory shape ran prod for days at util 0.85 with 2.2× pool margin (v20)
- site-26 made MTP-at-PP3 work (see above)
- the drafter's ~4–5 GB lands on PP2 (the 40 GB tail rank), which has the slack

Measured on the promoted lane: KV pool **488,863 tokens = 1.86× the window**; the
drafter cost only ~38K tokens of pool (~9%).

## Config deltas vs the v20 no-MTP recipe

1. `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
2. dual served-model aliases: `flash-next` and `qwen3.8-27b` both resolve (client compat)
3. nothing else moves: same partition, seqs, timeouts, tools, auth, vision contract

## Validation battery (promoted lane, 2026-09-02)

| Check | Result |
|---|---|
| KV pool | 488,863 tokens (1.86× window; drafter −38K ≈ 9%) |
| Warm single-stream decode | **85.6 tok/s** (no-MTP recipe: 57–59) |
| Sustained 512-token generation | 71.3 tok/s, zero doubled-text patterns |
| 2 concurrent streams | 88.9 tok/s aggregate (96 no-MTP — small spec discount) |
| Draft acceptance | 55.6% mean (1,902 / 3,423 draft tokens) |
| Stop behavior | clean `finish=stop` at `max_tokens`; no runaway |
| Tool calling | OpenAI tools pass (`qwen3_xml` parser) |
| Auth contract | wrong-key → 401, `/health` unauthenticated |
| Needle-in-haystack @ 36K / 187K | HIT / HIT (thinking budget 600) |
| Prefix caching at depth | 187K re-request answered in 10.9 s |
| DRAFTSYNC telemetry | flowing on both non-last ranks; 478 SPEC-OUT events |

## Gotchas banked this window

- **Thinking-budget needle artifact (the false miss):** a needle test with
  `max_tokens: 220` "fails" — the model burns the entire budget on
  `reasoning_content` before the answer can form. Same test at a 600 budget hits.
  On this model family, suspect the thinking budget before suspecting retrieval.
- **`ignore_eos` bench artifact:** greedy + `ignore_eos` lets generation walk into
  degenerate continuations (role-token loops) after the answer. Fine for throughput
  benches; never quote such output as a quality signal. Real stop behavior is clean.
- **First request after boot absorbs cold JIT** (minutes at PP3, worst on the 3090
  rank) — benchmark warm, always.
- **Concurrency spec discount:** 2-stream aggregate is slightly BELOW no-MTP
  (88.9 vs 96) — verification overhead pays off at small batches, not saturated
  ones. Fine for single-agent workloads; re-bench before serving many streams.

## Rollback

The 786K no-MTP lane is one command:
[`scripts/serve-vllm-pp3-786k.sh`](../../scripts/serve-vllm-pp3-786k.sh)
(YaRN 3.0, util 0.92, validated to 737K retrieval). Mechanics per the lane-swap
playbook: `docker rm -f qwen38-prod-8012 && bash <launcher>`.

## 786K + MTP: tested and parked (2026-09-02)

Tested over two boots (~20 min each) the night after the 262K promotion. It
**boots** — pool 817K-826K tokens (1.04-1.05× the window) at util 0.92, drafter
included — and passes every shallow check: 80-86 tok/s warm, 75.7 sustained @512,
56-79% acceptance, DRAFTSYNC flowing, 401/tools/stop clean.

It **fails at depth**: the first deep request (~300K-500K tokens) kills the engine.
py-spy `--native` put PP1 at 100% SM frozen in `cuLaunchKernel → index_put_kernel`
under `_forward_core` (`qwen_gdn_linear_attn.py:1520`) — the
`ssm_state[prefill_state_indices]` write in the chunked delta-rule path's
**spec-split branch** (`split_non_spec`), which only engages when the final prefill
chunk carries draft slots. The trigger needs depth × MTP together: the no-MTP 786K
lane needle'd 737K clean (pure prefill path); the 262K+MTP lane needle'd 187K clean.
Observed boundary: clean ≤187K, fast-fail at ~300K (HTTP error, body uncaptured),
20-minute livelock at ~498K (frozen-line dumps + flat compile caches = spin, not
compile).

**Verdict: parked pending a fork fix** to the GDN spec-split state write. The 262K
native + MTP lane remains prod. A fix here is the second upstream-reportable
PP+MTP finding (after site-26).
