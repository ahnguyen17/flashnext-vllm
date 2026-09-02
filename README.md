# FlashNext on vLLM — 176B MoE, 256K context, on a 3090 and two mining cards

Everything from a working production deployment of **Qwen3.8-Flash-Next** (176B-parameter
MoE with hybrid attention — 36 GDN linear-attention layers + 12 full-attention layers —
and a native vision encoder), served with **vLLM pipeline-parallel across three
mismatched GPUs**: an RTX 3090 (24 GB) and two unlocked NVIDIA CMP 170HX mining
cards (64 GB + 40 GB).

Measured end-to-end, not estimated: **262,144-token context, live vision,
~4,500 tok/s prefill, 85 tok/s warm decode with MTP-3 speculative decoding**
(57–59 no-MTP). The same rig also runs a llama.cpp GGUF
fallback lane; both lanes are documented with the same yardstick.

## Results (2026-09-02, same rig, same context window, same model)

| | llama.cpp GGUF lane | **vLLM PP3 lane (prod)** |
|---|---|---|
| Weights | UD-Q4_K_XL, ~104 GB across 2× 170HX | AWQ-INT4, ~176 GB across 3090 + 2× 170HX |
| Context | 262,144 | 262,144 (native; 786K via YaRN 3.0 — see docs) |
| Decode | 34–36 tok/s | **85 tok/s warm, 71 sustained @512 (MTP-3)**; 57–59 no-MTP |
| MTP draft acceptance | — | 55.6% mean (1.9 of 3 draft tokens) |
| Prefill | ~605 tok/s | **~4,494 tok/s** (7.4×) |
| Real 250K-token request | — | accepted: 249,633 prompt tokens, 56 s prefill |
| Retrieval at depth | — | needle HIT at 36K and 187K (MTP on); to 737K on the YaRN lane |
| Concurrency | 1 sequence | 4 sequences — warm aggregate: 96 (2 streams) no-MTP / 89 with MTP |
| Tool calling | — | OpenAI tools + `tool_choice` (parser: `qwen3_xml`, matches the model's XML template) |
| Vision | via `mmproj-F16.gguf` sidecar | native, in-checkpoint ViT |
| Cold boot | ~3 min | ~11 min |
| Crash recovery | manual | auto (`--restart unless-stopped`) |

Raw evidence of the passing 256K verification: [`results/verify-256k.log`](results/verify-256k.log).

## Repo layout

```
scripts/
  serve-vllm-pp3-262k-mtp.sh # the prod vLLM lane: 262K native + MTP-3 (sanitized)
  serve-vllm-pp3-786k.sh    # the 786K YaRN 3.0 no-MTP lane (rollback / long-context day)
  serve-vllm-pp3-256k.sh    # the original no-MTP recipe the above derive from
  site26-pp-draft-table-sync.py  # the PP draft-table ring-sync patch (MTP at PP>1)
  serve-llamacpp-gguf.sh    # the llama.cpp GGUF fallback lane
  verify-256k.sh            # boot diag -> quality -> REAL 250K-token request -> vision
  bench-mtp-pp3.sh          # speculative-decoding bench launcher
results/
  verify-256k.log           # passing run output
docs/
  deployment/               # full deployment log: bring-up, PP4->PP3 partitioning,
                            # deadlock hunt, lane-swap playbook, isolation probes
  pp-debug/                 # MTP speculative-decoding forensics: NCCL deadlock
                            # patches (sites 15/17/18), corruption hunt, fork bug map
```

## Quickstart (vLLM lane)

Prereqs: Docker + NVIDIA container toolkit; the fork image (see caveat below); the
AWQ checkpoint at a path you mount as `/model`.

```bash
export VLLM_API_KEY=your-secret      # scripts default to 'change-me'
bash scripts/serve-vllm-pp3-262k-mtp.sh  # ~9-11 min to come up
bash scripts/verify-256k.sh           # full verification incl. a real 250K-token request
```

**Image caveat:** the recipe uses image `qwen38-flash-next:pp3fix22` — a locally built
vLLM fork image (CUDA 13 production base) carrying FlashNext/Qwen4Exp architecture
support, PLE CPU offload, Triton GDN decode kernels (baked in as image env), our PP3
deadlock fixes (sites 8–10, 13, 17–18, 23), and the site-26 draft-table ring sync
that makes MTP work under pipeline parallelism.
It is not on a registry; the patches it contains are in
[`docs/pp-debug/references/site15-17-18-patches.md`](docs/pp-debug/references/site15-17-18-patches.md),
[`docs/pp-debug/references/spec-corruption-hunt.md`](docs/pp-debug/references/spec-corruption-hunt.md),
and [`scripts/site26-pp-draft-table-sync.py`](scripts/site26-pp-draft-table-sync.py)
so you can recreate or port them.

## The 256K context math (why it fits on 24 GB + 104 GB)

Full derivation in [`docs/pp-debug/references/context-and-vision-math.md`](docs/pp-debug/references/context-and-vision-math.md). Short version:

- The architecture is hybrid: only **12 of 48 layers** are full attention (1-in-4
  interval). The other 36 are GDN — constant-size state per sequence, no per-token KV.
- Full-attn layers have 2 KV heads × 256 dim, bf16. At partition `8,28,12`:
  per-token KV is **PP0 = 4 KB, PP1 = 14 KB, PP2 = 6 KB**.
- Default pools at `--gpu-memory-utilization 0.85` hold ~603K / 588K / 2M tokens;
  the binding rank (PP1, the 64 GB card) holds 588K tokens against a 262,144 need —
  **2.2× headroom**, no partition surgery required.
- With MTP-3 on, the drafter (weights + draft KV, last rank) costs ~38K tokens of
  pool (~9%): measured 488,863-token pool = **1.86× the window** at util 0.85.
- **Do not pass `--kv-cache-memory`** with asymmetric partitions: it caps every rank
  globally and strangles the fat rank below the target.
- First multi-sequence generations after boot measure ~4x slow (CUDA-graph capture per batch shape happens on first use) — benchmark warm.
- 256K boots transiently OOM-retry on the 24 GB rank during memory profiling
  (expandable-segments warnings) — benign, it recovers. Boot takes ~11 min.
- If you OOM for real, the ladder is: `--max-num-batched-tokens 4096` →
  repartition `6,30,12` → drop to 192K.
- **Prefix caching is disabled deliberately.** On this GDN architecture, long
  multi-turn **vision** conversations (agent loop: screenshot each step, growing
  history) reproducibly wedge the engine core at ~step 8-9 — `/v1/models` stays
  200 while completions hang forever (3/3 reproduced; 12/12 steps pass with the
  flag off). Log signature: Triton JIT of `_bilinear_pos_embed_kernel` mid-flight.
  Matches vLLM issue #45238 (GDN prefix-cache). Cost of leaving it off: late-step
  re-prefill (~+1-3 s/step at ~30K ctx). Re-enable only after the upstream fix.

## MTP speculative decoding: fixed and in production (site-26)

MTP k=3 has been **on in production since 2026-09-02**: 85 tok/s warm single-stream
(57–59 no-MTP), 55.6% draft acceptance, clean text, clean stop behavior.

The last blocker after the 37-build campaign: under pipeline parallelism the
speculator's `propose()` runs only on the last rank, so the other ranks verified
against zeroed draft-token tables — corrupted text, 0% acceptance. **Site-26** adds a
third broadcast to the site-17/18 input ring that distributes the draft-token table
to every rank before verification:
[`scripts/site26-pp-draft-table-sync.py`](scripts/site26-pp-draft-table-sync.py).
No public recipe existed for MTP-at-PP; this is the missing piece.

Validation numbers, gotchas (thinking-budget needle artifact, concurrency spec
discount), rollback, and the tested-and-parked 786K+MTP verdict:
[`docs/deployment/262k-mtp-prod.md`](docs/deployment/262k-mtp-prod.md).
The full debugging campaign history is in [`docs/pp-debug/`](docs/pp-debug/).

## Hardware notes

- GPU map: PP0 = RTX 3090 (16.1 GB of weights), PP1 = CMP 170HX 64 GB (45.9 GB),
  PP2 = CMP 170HX 40 GB (21.9 GB). `VLLM_PP_LAYER_PARTITION=8,28,12`,
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`, device lists ascending — all three matter.
- The CMP 170HX cards are mining cards unlocked to A100-class compute (including
  the Xid-31 WPR2-region driver fix). Unlocking is its own project; treat it as a
  prerequisite, not part of this repo.
- `--ipc=host --shm-size=96g` and `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200` are
  load-bearing for multi-rank startup, not cosmetic.

## License

MIT — scripts and docs. Model weights and the vLLM fork carry their own licenses.
