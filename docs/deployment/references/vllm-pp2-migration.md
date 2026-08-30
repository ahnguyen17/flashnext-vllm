# vLLM PP=2 migration — Flash-Next AWQ-INT4 on the 170HX pair (staged 8/28, ATTEMPTED 8/29 — hardware-blocked)

Recipe source: `github.com/vektorprime/qwen38-flash-next-pp2` — Dockerfile +
`patch_pp.py` (6-site PP patch) + verified launch flags, written 2026-08-29 on
2× CMP 170HX **64G** (stock, no cap mod). Author-reported perf: **>50 t/s TG
with MTP k=3**. The repo itself was never benchmarked beyond that note.

## EXECUTION LOG 8/29 (four attempts, then stopped — verdict below)

Working setup common to all attempts: image `qwen38-flash-next:pp2v2` (base
build + our 7th patch site, see below), `--gpus '"device=2,3"'`,
`-v ~/models/FlashNext-AWQ:/model:ro`, `--entrypoint vllm`,
`serve /model --served-model-name flash-next --api-key change-me
--pipeline-parallel-size 2 --gpu-memory-utilization 0.9 --mamba-cache-mode align
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code
--generation-config auto`, env `VLLM_PLE_CPU_OFFLOAD=1
VLLM_GDN_DECODE_KERNEL=triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_DEVICE_ORDER=PCI_BUS_ID`, `--ipc=host --shm-size=96g`. Local model mount
avoids HF cache duplication — do NOT pass the repo id (it re-downloads).

1. **29/19, seqs 8, async** → PleOffloadWorker died instantly:
   `get_pp_indices: len(partitions)=2 does not match pp_size=1`. The PLE
   offload worker builds its model in a SINGLE-process context and still reads
   the process-global partition env. **7th patch site (ours)**: in
   `vllm/distributed/utils.py get_pp_indices`, when `len(partitions) !=
   pp_size` and `pp_size == 1`, substitute `partitions = [num_hidden_layers]`
   instead of raising. Applied via temp container + `docker commit` → tag
   `pp2v2`. Verified by marker-string check inside the image.
2. **32/16, seqs 8, async, no compile volume** → weights loaded (PP0 51.41G /
   PP1 33.45G, PLE registered, registrations complete) then died ~10 min into
   graph capture: PP1 SIGABRT (exit -6) inside MTP warmup recv; peer saw gloo
   "Connection closed by peer". Classic **cold-compile NCCL watchdog kill** —
   PP1's decode-graph compile stalled the pipeline recv past the watchdog
   window. Fixes: mount `-v vllm-compile-cache:/root/.cache/vllm` (persistent
   compile cache — without it EVERY restart recompiles from zero) and drop
   `--async-scheduling` for the first boot.
3. **32/16, seqs 8, no async, compile volume** → got furthest: PIECEWISE
   11/11 + FULL 5/5 graphs captured on PP0, speculator capture started on
   PP1… then remote-worker crash at 16:48. The startup lines explain why:
   PP1 reported `--kv-cache-memory=-70409932` (**−0.07 GiB — negative**),
   KV pool 7,822 tokens, `--max-model-len auto` reduced 262144 → **9,600**
   (only 0.35 GiB available for KV). PP0 had 3.06G KV + ~5G spare. The
   speculator (MTP drafter) needs memory ON THE LAST RANK — PP1 had none.
4. **33/15, seqs 2, max-model-len 32768** — last try to buy the small card
   room (GDN/mamba cache is per-seq: seqs 8→2 frees several GB). **Final
   outcome: CRASHED 16:47 — NCCL collective timeout on BOTH ranks (watchdog
   dump, `Last enqueued NCCL work: -1`) — the same negative-headroom root
   cause in its third mask. 4/4 attempts dead; verdict closed.** Even a boot
   would have been lab curiosity only, not a 256K prod lane.

**Verdict (memory math):** PP1 budget at util 0.9 = 35.6G must hold stage
weights + MTP drafter (~2–3G) + peak activation (~1.3G) + graphs (~0.3G) +
KV. Useful KV needs weights ≤ ~30G → ≤ ~13-14 layers → PP0 ≥ 34 layers ≈
54.9G+ weights > its 57.6G budget once act/graphs land. Every rebalance moves
the bust. **The recipe is a 2×64G shape; a 176B + MTP + PLE-offload lane
cannot give the 40G card a useful role.** Do not re-attempt on this pair
without a new knob (an MTP-free profile is pointless — MTP is the point).

**Other traps hit:** `docker commit` of a container run with
`--entrypoint python3` bakes that entrypoint into the image — relaunches then
pass `serve` to python3 (`can't open file '/workspace/serve'`); always pass
`--entrypoint vllm` explicitly. Benching vLLM needs client-side stream timing
(TTFT = first content chunk; decode t/s = n/(total−ttft)) — vLLM has no
`/props`/`/completion`; and under MTP never count SSE deltas, only
`usage.completion_tokens`. RAM note: PLE (95G) + engine ≈ 104G in container;
host sat at ~7.5G/8G swap with si/so ~40/140 KB/s — mild, NOT thrashing;
check `vmstat` before concluding RAM is the killer (a mis-parsed `free`
column once read as "107G swap used" what was actually mem-used).

**If a second 64G card ever lands:** follow the repo verbatim (even 24/24
split, their flags incl. `--async-scheduling`), expect >50 t/s MTP-3 decode,
then prod-lane-swap onto 8012. Assets already on disk: AWQ at
`~/models/FlashNext-AWQ` (38/38 size-verified), image `qwen38-flash-next:pp2v2`
(all 7 patch sites), compile-cache volume `vllm-compile-cache`.

## Their verified shape (what the repo proves works on 64+64)

- Model: `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (~174 GB, 38 safetensors) — 512-expert AWQ-INT4 MoE (~85 GiB GPU-resident) + 95 GiB **bf16 PLE table** (host RAM via `VLLM_PLE_CPU_OFFLOAD=1`) + MTP module
- vLLM PR #53899 @ `a5530b9` + 6-site patch; `--pipeline-parallel-size 2`
- Weight split PP0 39.48 G / PP1 45.53 G; KV pool 14.97 GiB → 1,196,679 tokens (GDN hybrid KV is cheap); `--max-model-len auto` resolves 262144
- Cold start ~9 min; PLE first fill streams shards during load
- Required flags: `VLLM_GDN_DECODE_KERNEL=triton`, `--mamba-cache-mode align`, `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, `--enable-prefix-caching`, `--async-scheduling`, `--ipc=host --shm-size=96g`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Repo gotchas worth remembering (from their reproduction guide)

- `PYTORCH_ALLOC_CONF` (without `_CUDA_`) is silently ignored → allocator fragmentation OOM crash-loop that LOOKS like real OOM. Check the env NAME first.
- `--gpus '"device=0,1"'` — nested quoting mandatory (device LIST, not count).
- Image `ENTRYPOINT ["vllm"]` swallows one-off commands — use `--entrypoint python3`.
- Base image lacks `git` and `python` (only `python3`).
- `patch_pp.py` does exact-string matching and prints WARNING (not failure) on pattern miss — a WARNING means upstream moved; **never trust an image that printed one**.
- `--restart always` + crash-loop can wedge GPUs mid-release — `docker rm -f` the loop, verify cards free, relaunch.
- `num_speculative_tokens>1` re-runs one MTP layer k times; k=3 is what the author measured >50 t/s with.

## The generalizable vLLM lesson

Upstream "requires PP=1" refusals on feature+parallelism combos are often
**conservative guards / rank-assumption bugs, not architecture limits**. When
the gated component lives on exactly one rank (here: the PLE table on layer 1 →
rank 0), gating on `get_pp_group().is_first_rank` at the guard sites makes
PP>1 safe. Their six sites: model_state PLE gate, gpu_worker offload-config
validator, ple_offload connector (init/_setup_layers/_launch), model_runner
`_setup_ple_offload`, HC-mixer persistent-weight skip on non-last ranks, MTP
drafter input branch (`is_first_rank` → `or hidden_states is not None`).
Corollary learned the hard way: **any `VLLM_*` partition/parallel env var is
read by EVERY process in the engine** — single-process side contexts (PLE
worker, speculator capture) must tolerate it (our 7th site).
