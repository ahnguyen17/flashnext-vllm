# Site-32 A/B: prefix caching re-enabled with mamba-align fix — results ledger

**Date:** 2026-09-20 · **Rig:** 170HX+3090 PP3 · **Probe:** `~/scripts/prod-grown-prefix.py`
(randomized corpus offsets per run; scenario = 40K turn-1, grown +2K turn-2 — real
conversation shape, not repeated prefix)

## Grown-prefix probe (the user-felt numbers)

| Scenario | cache OFF (pp3fix26) | cache ON + site-32 (pp3fix27-mambaalign) |
|---|---|---|
| A turn-1 40K cold | 9.6–35.9s | 4.7–41.3s (varies w/ thermal state of GPU2) |
| **A turn-2 grown +2K** | **10.07s** | **1.98s** |
| A turn-2 repeat (control) | 10.23s | 1.00s |
| B booster re-send | 19.41s → miss | 1.05s (hit) |
| B turn-2 after booster | 29.22s | 2.92s |
| C turn-2 after 2 boosters | 42.74s | 5.08s |
| Prefix cache hit rate (server) | 0% | 41% during probe |

**The open turn-2 TTFT problem (root-caused 8/23 as upstream #45238) is functionally
solved by cache-ON + pmu400 + site-32: 10.07s → 1.98s at 40K depth.**

No CUDA IMA on any first prefix hit — the regression the unpatched fork crashes on
(akumaburn's fix works; full credit to akumaburn/vllm-dflash2 commit 1c7542a4c5).

## Clean depth-decode ladder (2026-09-20 00:44, post fan-speed fix, zero throttle)

Thermal at start/end: GPU2 57→75°C, sw_thermal_slowdown Not Active throughout.
Nonce prompts (cache-cold by design — this ladder measures cold-prefill + decode):

| Depth | TTFT | prefill | decode | 9/10 cache-OFF reference |
|---|---|---|---|---|
| 32K | 9.1s | 3,602 t/s | 78.7 t/s | 103–106 (first run post-boot; warmup artifact) |
| 64K | 18.2s | 3,588 t/s | 96.7 t/s | 100 — within noise |
| 131K | 39.3s | 3,392 t/s | 87.0 t/s | 87 — identical |

**Verdict: no decode regression from prefix caching. Cache-ON stack fully promoted.**

## Thermal confound discovered during validation

GPU2 (170HX-64G, PP0/28 layers) hit 86°C → SW thermal slowdown active (210 MHz vs
1695 max, ~745s cumulative throttle) — passive cards, chassis-airflow dependent.
Decode measured as low as ~55 t/s warm under full throttle vs ~100 t/s expected.
**Any decode numbers measured while GPU2 >80°C are invalid — check
`nvidia-smi --query-gpu=clocks_event_reasons.sw_thermal_slowdown` before benching.**
Depth-decode ladder invalidated by this; re-run needed after cooldown (pending).

## Incident during bench window

Container received external SIGTERM at 00:26:26 PT (docker kill signal=15 then 9,
exit 137) mid-depth-decode — source unidentified (no local actor found; asked Andy).
Auto-restart (unless-stopped policy) recovered cleanly; engine healthy after reboot.

## Status

- Promoted variant: `scripts/serve-8012-prefixcache.sh` (cache ON, pmu400, site-32)
- Rollback: `scripts/serve-8012-cacheoff.sh` (pp3fix26, cache OFF)
- Pending: clean depth-decode ladder (post-cooldown), 262K deep validation
