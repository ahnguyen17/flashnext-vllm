# site-31: Full-range 262K on the mixed-arch rig (QSA width de-specialization)

**Date:** 2026-09-04 · **Image:** `pp3fix26` (on pp3fix25) · **Status:** PROD

## Problem

Three incident classes on the sm_86 + sm_80 (patched-libcuda) rig shared one
root cause: **first-use triton kernel module-loads during inference on the
3090 rank are a lottery** — success / infinite `_init_handles` spin / Xid-31
crash. Deep context made it acute because the QSA kernels baked
`PAGE_TABLE_WIDTH` (block-table width = context depth in pages, ~1600-token
pages → width 5..~164 blocks) as a `tl.constexpr`. Every new depth band
compiled a *different* kernel: a fresh first-use load per band. The 250K band
lost 0/3 (deterministic wedges).

(Correction of an earlier hypothesis: `num_splits` is *batch*-coupled, not
depth-coupled — the width constexpr was the depth mint.)

## Fix 1 — site-31 patch (`patches/site31-qsa-width-despecialize.py`)

In three kernels (`_qsa_sparse_paged_gqa_splitk_kernel`,
`_qsa_mqa_paged_kernel`, `_build_qsa_metadata_kernel` — the last also baked the
table strides): demote the width and width-coupled scalars
(`page_table_width`, `stride_table_req`, `num_columns`) to plain runtime args
and mark them `do_not_specialize` (name-based; the pattern already existed in
`qsa_cache.py`). All usages are clamps/masks — no `tl.arange` — so runtime is
safe. Result: **one cubin per kernel at every depth**, compiled and loaded at
boot warmup (site-30 `VLLM_PP_WARMUP=1`).

## Fix 2 — boot-battery warmup (`scripts/boot-battery-warmup.sh`)

Residual class after Fix 1: the *first deep request after a fresh boot* could
still wedge (131K needle: 0/3) — the spec-decode/post_update kernel family
first-uses mid-prefill (final-chunk + draft-slot path). Evidence: with a short
varied battery run first (4 requests incl. a 700-token sustain), the same
needle passes.

The script watches container (re)start events, waits for `/health`, then fires
a 12-second battery. Run it as a persistent user service (Linger enabled):

```ini
[Unit]
Description=Boot-battery warmer for the vLLM prod lane
After=docker.service
[Service]
ExecStart=/bin/bash /opt/scripts/boot-battery-warmup.sh
Restart=always
[Install]
WantedBy=default.target
```

Catches manual relaunches *and* `--restart` crash-loops.

## Validation (2026-09-04, live lane, cache-OFF, MTP-3)

| Test | Before | After |
|---|---|---|
| 250K needle ×3 | 0/3 (wedge each) | **3/3** — 96/104/138s, exact retrieval, `jit+0` (zero new kernels at 254K tokens) |
| 131K, first deep req after cold restart | 0/3 (wedge) | **2/2** — 44s, exact hits |
| 250K, same cold boot | — | **PASS** — 111s @ 260,046 tokens |
| Shallow battery | 85 t/s class | unchanged (81 t/s, 77 sustained) |
| Boot time / KV pool | 521s / 505K tok | 510s / 505K tok |

## Corrections to earlier claims (falsified today)

1. "Cache-hit loads don't spin" — **false**. Compiles persist in the triton
   volume, but every process re-rolls module *loads*; a loser shape wedged 0/3
   on pure cache-hit loads.
2. "Warm-sweep bands are permanent across reboots" — **overstated**. Compiles
   persist; loads re-roll per process start. Boot-time loading (warmup /
   battery) and shape-space collapse (site-31) are the structural escapes.

## Forensic notes

- **jit-timeline-by-window**: map timestamped `jit_monitor` log lines (kernel
  names) to attempt windows; last kernel to compile before a freeze is the
  prime suspect; `jit+0` at depth proves shape coverage.
- **Telemetry trap**: `/metrics` scrapes can fail during heavy load (all-None)
  while working when idle — during live runs use the engine log's
  `Running: N reqs` lines as the zombie oracle.
- **py-spy pid discovery must be comm-based** (`/proc/*/comm == python3`);
  `pgrep -f <pattern>` self-matches the dump shell and silently captures
  nothing.

## Residuals

- The exact kernel behind the 131K first-request wedge was not stack-named
  (two py-spy captures grabbed the wrong process). The battery fix is
  empirical, validated on a full cold-restart cycle.
- 786K remains parked (site-27 requires prefix-cache ON; cache-ON is a
  3-strike no-go on this rig — separate disease).
- Upstream driver report (patched-libcuda module-load race on sm_86, three
  incident classes + today's) remains draft-worthy.
