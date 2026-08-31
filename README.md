# FlashNext on vLLM — 176B MoE, 256K context, on a 3090 and two mining cards

Everything from a working production deployment of **Qwen3.8-Flash-Next** (176B-parameter
MoE with hybrid attention — 36 GDN linear-attention layers + 12 full-attention layers —
and a native vision encoder), served with **vLLM pipeline-parallel across three
mismatched GPUs**: an RTX 3090 (24 GB) and two unlocked NVIDIA CMP 170HX mining
cards (64 GB + 40 GB).

Measured end-to-end, not estimated: **262,144-token context, live vision,
~4,500 tok/s prefill, 48–54 tok/s decode.** The same rig also runs a llama.cpp GGUF
fallback lane; both lanes are documented with the same yardstick.

## Results (2026-08-30, same rig, same context window, same model)

| | llama.cpp GGUF lane | **vLLM PP3 lane (prod)** |
|---|---|---|
| Weights | UD-Q4_K_XL, ~104 GB across 2× 170HX | AWQ-INT4, ~176 GB across 3090 + 2× 170HX |
| Context | 262,144 | 262,144 |
| Decode | 34–36 tok/s | **48–54 tok/s** (+40%) |
| Prefill | ~605 tok/s | **~4,494 tok/s** (7.4×) |
| Real 250K-token request | — | accepted: 249,633 prompt tokens, 56 s prefill |
| Concurrency | 1 sequence | 4 sequences — warm aggregate: 96 (2 streams) / 134 (3) / 169 (4) tok/s |
| Tool calling | — | OpenAI tools + `tool_choice` (parser: `qwen3_xml`, matches the model's XML template) |
| Vision | via `mmproj-F16.gguf` sidecar | native, in-checkpoint ViT |
| Cold boot | ~3 min | ~11 min |
| Crash recovery | manual | auto (`--restart unless-stopped`) |

Raw evidence of the passing 256K verification: [`results/verify-256k.log`](results/verify-256k.log).

## Repo layout

```
scripts/
  serve-vllm-pp3-256k.sh    # the prod vLLM lane (docker run recipe, sanitized)
  serve-llamacpp-gguf.sh    # the llama.cpp GGUF fallback lane
  verify-256k.sh            # boot diag -> quality -> REAL 250K-token request -> vision
  bench-mtp-pp3.sh          # speculative-decoding bench launcher (experimental)
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
bash scripts/serve-vllm-pp3-256k.sh   # ~11 min to come up
bash scripts/verify-256k.sh           # full verification incl. a real 250K-token request
```

**Image caveat:** the recipe uses image `qwen38-flash-next:pp3fix7` — a locally built
vLLM fork image (CUDA 13 production base) carrying FlashNext/Qwen4Exp architecture
support, PLE CPU offload, Triton GDN decode kernels, and our PP3 deadlock fixes.
It is not on a registry; the patches it contains are in
[`docs/pp-debug/references/site15-17-18-patches.md`](docs/pp-debug/references/site15-17-18-patches.md)
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

## MTP speculative decoding: honest status

MTP k=3 **engages** on this fork (62–68 tok/s measured) but draft tokens never become
real token IDs end-to-end — a design bug in the fork's spec-decode plumbing
(`[-1]` placeholders flow all the way to verification). The served lane therefore
runs **without** speculation: clean 48–54 tok/s beats corrupt 62–68. The entire
37-build debugging campaign — deadlock forensics, phantom-slot ledger, rollback
races, the final input-gather analysis — is in [`docs/pp-debug/`](docs/pp-debug/).

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
