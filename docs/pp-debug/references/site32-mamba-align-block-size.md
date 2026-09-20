# Site 32 — mamba-align block-size fix (ported) + prefix caching re-enabled on 8012

**Date:** 2026-09-19/20 · **Status:** see Validation below · **Image:** `qwen38-flash-next:pp3fix27-mambaalign`

## Source and credit

Ported from **[akumaburn/vllm-dflash2](https://github.com/akumaburn/vllm-dflash2)** commit
`1c7542a4c5` (2026-09-18, Claude-coauthored) — original author a.eslampanah@live.com.
That fork carries three commits on upstream vLLM 0.29.0-dev; this is the first, plus
our base-specific notes. Patch file: [`patches/mamba-align-block-size.patch`](../../patches/mamba-align-block-size.patch).

## The bug

After KV cache groups are built, `cache_config.block_size` becomes the smallest
prefix-cacheable group's block size. Two Mamba align sites read it instead of
`cache_config.mamba_block_size`:

1. `MambaHybridModelState.add_request` seeds a resumed request's running-state column
   as `(num_computed - 1) // block_size`. On a large prefix hit that indexes the block
   table far past the row → CUDA IMA (`WARP_OUT_OF_RANGE_ADDRESS`) on the **first
   prefix-cache hit** — or silently resumes from a wrong state column.
2. `Scheduler._mamba_BLOCK_aligned_split` aligned prefill chunks to the small block
   size, so chunk ends miss Mamba block boundaries → breaks the slot-p = state-after-
   (p+1)·block_size invariant.

On our prod base (nightly 6f7df92a8e): scheduler.py:392 and mamba_hybrid.py:116.
The bug is masked while prefix caching is disabled (align mode forced to 'none') —
which is exactly why 8012 has been running cache-OFF since the prefix-cache misses
were first root-caused. On this base `platforms/interface.py:922` sets
`mamba_block_size = block_size` under align mode, so the patched reads are
well-defined; the fix is inert with caching off and load-bearing with caching on.

## What shipped

- Image `pp3fix27-mambaalign` = pp3fix26 + patch applied in-container + `docker commit`
  (commit message carries full credit). No venv rebuild; `/opt/vllm` editable install.
- `scripts/serve-8012-prefixcache.sh` — prod variant with `--enable-prefix-caching
  --prefix-match-unit 400` (unit must divide block 1600).
- Rollback unchanged: `scripts/serve-8012-cacheoff.sh` (pp3fix26, cache OFF).

## Validation (2026-09-20, prod shape, grown-prefix probe `~/scripts/prod-grown-prefix.py`)

| Scenario | cache OFF (pp3fix26) | cache ON + site-32 (pp3fix27) |
|---|---|---|
| A: turn-2 grown +2K @40K | 10.07s | see results ledger |
| B: booster turn-2 | 29.22s | see results ledger |
| C: double-booster turn-2 | 42.74s | cache-ON numbers pending first run |

Baseline (cache OFF) captured 2026-09-20 ~00:00 PT; cache-ON numbers to be appended
after the first post-restart probe. A no-IMA pass on the first repeat send is itself
the regression test for the ported fix (the unpatched fork crashes there).
