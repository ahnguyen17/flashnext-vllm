# site-31: QSA page-table width de-specialization (262K full-range fix)

On mixed-architecture rigs (sm_86 + sm_80 under a patched libcuda), first-use
triton kernel module-loads during inference can spin indefinitely or Xid-crash.
The QSA kernels baked `PAGE_TABLE_WIDTH` (block-table width = context depth in
pages) as a `tl.constexpr`, so every new context depth minted a fresh cubin —
a first-use load lottery per depth band. De-specializing the width (and
width-coupled scalars) into runtime args with `do_not_specialize` collapses all
depths onto one cubin per kernel, loaded at boot warmup.

Result: 250K-token needles went 0/3 (wedge) -> 3/3 (96-138s, exact needle
retrieval, zero mid-flight JIT at depth).

Companion: `scripts/boot-battery-warmup.sh` — fire a short varied-length
request battery after every engine (re)start (systemd/docker-events watcher).
Covers the remaining first-request kernel-load class (spec-decode/post_update
family) that no boot warmup shape-set reaches.
