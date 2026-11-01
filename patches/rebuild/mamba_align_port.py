#!/usr/bin/env python3
"""Port of akumaburn mamba-align fix (site-32) for the PR-53899 reconstruction base.
Same two functional edits as mamba-align-block-size.patch; tolerant anchors.
Exits 0 if both applied OR already present; exits 1 if an anchor is ambiguous/missing."""
import ast, pathlib

def sub(path, pairs):
    p = pathlib.Path(path)
    src = p.read_text()
    changed = False
    for old, new in pairs:
        if new in src:
            print(f"[mamba-port] {path}: already applied")
            continue
        n = src.count(old)
        assert n == 1, f"[mamba-port] {path}: anchor count {n} for {old[:60]!r}"
        src = src.replace(old, new)
        changed = True
    if changed:
        ast.parse(src)
        p.write_text(src)
        print(f"[mamba-port] {path}: applied + ast ok")

# 1) scheduler.py — align split must use mamba_block_size
sub("vllm/v1/core/sched/scheduler.py", [(
    "        if start >= prefill_end:\n            return num_new_tokens\n\n        block_size = self.cache_config.block_size",
    "        if start >= prefill_end:\n            return num_new_tokens\n\n"
    "        # Not cache_config.block_size: it becomes the smallest group block size\n"
    "        # (e.g. a DFlash sliding-window drafter's), below the Mamba block size.\n"
    "        block_size = self.cache_config.mamba_block_size\n"
    "        assert block_size is not None"
)])

# 2) mamba_hybrid.py — resumed-request state column must use mamba_block_size
import re
p2 = pathlib.Path("vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
src2 = p2.read_text()
if "mamba_block_size" in src2 or "_mamba_bs" in src2:
    print("[mamba-port] mamba_hybrid: already applied (site-27d equivalent or newer)")
else:
    pat = re.compile(r"(num_computed_tokens\s*-\s*1\)\s*//\s*self\.cache_config)\.block_size")
    hits = pat.findall(src2)
    assert len(hits) == 1, f"[mamba-port] mamba_hybrid: regex hits {len(hits)}"
    src2 = pat.sub(r"\1.mamba_block_size", src2)
    ast.parse(src2)
    p2.write_text(src2)
    print("[mamba-port] mamba_hybrid: applied + ast ok")
print("[mamba-port] done")
