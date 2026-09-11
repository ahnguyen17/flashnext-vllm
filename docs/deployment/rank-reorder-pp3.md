# Rank reorder: PP0 off the sm_86 card (2026-09-10)

A two-line change to the 262K+MTP prod recipe that cost zero capacity and returned
+15% shallow decode, +34% 4-stream aggregate — and, more importantly, removed the
**fresh-process first-deep wedge lottery** that had been this rig's most persistent
stability disease.

Launcher: [`scripts/serve-vllm-pp3-262k-mtp-rank-reorder.sh`](../../scripts/serve-vllm-pp3-262k-mtp-rank-reorder.sh)

## Motivation

An independent results repo ([allover326/qwen38-flash-next-170hx-results](https://github.com/allover326/qwen38-flash-next-170hx-results))
benchmarked the same model on a pure sm_80 rig (2× CMP 170HX, no desktop-class card):
107 tok/s single-stream decode, 5,781 tok/s prefill at 249K tokens, flat across the
whole context window. Our mixed rig (RTX 3090 + 170HX 64G + 170HX 40G) served the
3090 as PP0 — every pipeline step is latency-chained to the slowest rank, and the
3090 has roughly a third of the 170HX's memory bandwidth. The pure-sm_80 numbers
were the control experiment our rig couldn't run on itself.

## The mechanism (and the lever that doesn't exist)

The intended lever was `CUDA_DEVICE_ORDER=FASTEST_FIRST`. **It does not work in
containers**: the nvidia container runtime hands devices to the container in PCI-bus
order regardless of the host-side env, and our 3090 sits on the lowest PCI bus, so
it enumerates as `cuda:0` under both orderings (probed and verified before launch).

The working mechanism is a `CUDA_VISIBLE_DEVICES` **remap inside the container**,
applied on top of the PCI-ordered base:

```
PCI base:   cuda:0 = RTX 3090 24G   cuda:1 = 170HX 64G   cuda:2 = 170HX 40G
CUDA_VISIBLE_DEVICES=1,0,2
Effective:  cuda:0 = 170HX 64G (PP0, 28L)  cuda:1 = 3090 (PP1, 8L)  cuda:2 = 170HX 40G (PP2, 12L + drafter)
```

`VLLM_PP_LAYER_PARTITION` is re-numbered `8,28,12 → 28,8,12` so every **physical**
card keeps its exact layer count from the previous recipe — the change is pure
pipeline order, capacity and KV pool unchanged. Probe your own enumeration with
`torch.cuda.get_device_properties(i)` before transplanting; the device list itself
must stay PCI-ascending (see the device-renumbering gotchas in the PP4 post-mortem).

## Measured (same image pp3fix26, same day, warm)

| Metric | PP0=3090 (old) | PP0=64G 170HX (new) |
|---|---|---|
| decode @32K filled | 89.9 t/s | **103–106 t/s** |
| decode @64K filled | 91.2 t/s | **100.2 t/s** |
| decode @131K filled | 91.5 t/s | 86.7 t/s |
| 4-stream aggregate | 132.3 t/s | **177.8 t/s** |
| 2-stream aggregate | 142.9 t/s | **149.2 t/s** |
| long-512 single | 73.5 t/s | **77.5 t/s** |
| boot time | ~510–530 s | **350 s** (promote boot) |
| first-deep needle (32K + 131K, fresh process) | racy wedge lottery | **2/2 first-try, zero wedges** |

The 131K tail is the one regression (−5%); everything else wins. The single
regression wasn't enough to offset the stability gain.

## Why the stability result matters more than the throughput

Every prior wedge class on this rig — the Xid-31 storm, the `_init_handles` spins,
the first-deep lottery — shared one fingerprint: **first-use triton kernel
specialization during inference on the sm_86 rank under the patched driver**
(see `docs/` field reports; upstream PR #36). Moving PP0 off sm_80 didn't just
speed the lane up; the fresh-process first-deep needle pair passed first-try,
which had never happened on a cold boot before. The 3090 still serves as PP1 —
this is not a cure for the sm_86 module-load race, but the pipeline's most
load-bearing rank no longer hosts it.

## Bench-methodology notes

- First-pass prefill at any new depth band reads ~40% low (per-shape JIT mint);
re-run warm before quoting numbers.
- Same warm-shape discipline as every recipe on this rig: first request per
output/batch shape after boot pays graph capture.

## Rollback

Delete the `CUDA_VISIBLE_DEVICES` env line and restore `VLLM_PP_LAYER_PARTITION=8,28,12`
(the device list stays PCI-ascending). One relaunch, ~6–9 min boot.
