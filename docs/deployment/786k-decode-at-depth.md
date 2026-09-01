# Decode at Depth: Why Long-Context Generation Slows Down (2026-09-01)

Measured on the 786K YaRN lane: warm decode at fresh context is ~50 t/s, but at
538K tokens of context it drops to **~10 t/s**. This doc explains why, why it is
**not** a YaRN/configuration artifact, when it matters, and what the levers are.

## The mechanism: decode is memory-bandwidth-bound, and depth multiplies the bytes

Every generated token attends over the *entire* accumulated context. For the
hybrid architecture's ~12 full-attention layers (2 KV heads × 256 dim × K+V ×
bf16 ≈ 2 KiB per token per layer, distributed across the pipeline), a single
decode step at 538K context streams:

| Rank | KV per token | KV read per step @538K | Bandwidth floor |
|---|---|---|---|
| 64 GB rank (28 layers) | ~14 KB | ~7.3 GiB | ~5 ms @ ~1.55 TB/s |
| 40 GB rank (12 layers) | ~6 KB | ~3.1 GiB | ~2.3 ms @ ~1.35 TB/s |
| 24 GB rank (8 layers) | ~4 KB | ~2.1 GiB | ~2.2 ms @ ~0.94 TB/s |

Those reads happen **every step**, serially through the pipeline (PP step time ≈
sum of rank times), on top of weight reads, MoE routing, and GDN state updates.
Fresh context: ~20 ms/step → ~50 t/s. At 538K: ~100 ms/step → ~10 t/s. The
bandwidth arithmetic lands almost exactly on the measurement.

Two mitigating properties of this architecture:

- **The GDN layers (~3/4 of the stack) are recurrent** — constant-size state,
  O(1) per token at any depth. A pure-transformer at 538K would degrade far
  worse; this is the hybrid design doing its job.
- The sparse/indexer retrieval path (PLE) is built to stay cheap at depth —
  which is why *prefill* stays near-linear (4,468 → 2,460 → 1,833 t/s at
  314K/538K/737K) even as decode pays the full KV-read tax.

## This is not a YaRN artifact

The 256K configuration pays the identical tax at the same depth — its window
just never got there. Note the measurement trap: a spec-sheet number like
"47 t/s @256" can mean 256 **output** tokens at short context, not 256K
**context** depth. Always check which before comparing.

## When it matters

| Workload | Impact |
|---|---|
| Short-context chat / agents (≤50K) | None — depth never engages, ~50 t/s |
| Query a 500K archive | Fine — prefill once (~4–7 min), then 100–300-token answers cost 10–30 s |
| Long *outputs* at full depth | The pain case — a 2,000-token synthesis at ~538K ≈ 3.5 min |
| Agent loops with 300K+ history | Drags — 500-token reasoning turns cost ~50 s each |

## Levers (if it ever bites)

1. **`--kv-cache-dtype fp8`** — halves KV bytes, which halves the dominant
   per-step read tax: expect ~10 → ~14–16 t/s at depth. The same flag is the
   prerequisite for a 1M window on this memory budget, so one change buys both.
   (Unverified on the QSA/indexer path of this fork — one boot to find out.)
2. **MTP / speculative decoding** — a throughput multiplier that stacks on top
   of any KV-cache improvement (engagement proven separately on this rig;
   pending a fork draft-id fix).
3. More bandwidth — the actual wall. Deeper pockets or more cards.

Companion doc: [`786k-yarn-context.md`](786k-yarn-context.md) — the config
recipe and the validation ladder that produced these numbers.
