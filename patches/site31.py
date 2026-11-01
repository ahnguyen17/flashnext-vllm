#!/usr/bin/env python3
"""site-31: de-specialize QSA page-table WIDTH (depth-minted triton kernels).

Root cause: block-table width (in pages) scales with context depth, and it is
baked as a triton *constexpr* (plus stride/column runtime args that triton
specializes on). Every new depth band therefore mints a first-use kernel load
on the 3090 rank — the racy lottery under the patched libcuda.

Fix: make the width (and width-coupled scalars) plain runtime args and mark
them do_not_specialize, so ONE cubin serves every depth. Usages are clamps and
mask comparisons only — no tl.arange dependence — so runtime is safe.

Files (relative to vllm package root):
  models/qwen4_exp/nvidia/ops/qsa.py   (2 kernels + 2 launchers)
  models/qwen4_exp/common/qsa_cache.py (metadata kernel signature only)

Idempotent; ast-gated; dry-run: pass a host dir containing bare filenames.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "# site-31"

OPS_EDITS = [
    # (description, old, new)
    (
        "decorator A: _qsa_mqa_paged_kernel do_not_specialize",
        "@triton.jit\ndef _qsa_mqa_paged_kernel(",
        '@triton.jit(do_not_specialize=["page_table_width", "stride_table_req", "num_columns"])  # site-31\ndef _qsa_mqa_paged_kernel(',
    ),
    (
        "signature A: insert runtime page_table_width",
        "    num_requests,\n    score_divisor,",
        "    num_requests,\n    page_table_width,  # site-31\n    score_divisor,",
    ),
    (
        "signature A: drop PAGE_TABLE_WIDTH constexpr",
        "    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,\n    NUM_HEADS: tl.constexpr,",
        "    PAGE_SIZE: tl.constexpr,\n    NUM_HEADS: tl.constexpr,",
    ),
    (
        "body A: clamp uses runtime width",
        "        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)",
        "        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)  # site-31",
    ),
    (
        "launcher A: pass width positionally",
        "        page_table.shape[0],\n        float(score_divisor),\n        PAGE_SIZE=k_cache.shape[1],\n        PAGE_TABLE_WIDTH=page_table.shape[1],",
        "        page_table.shape[0],\n        page_table.shape[1],  # site-31\n        float(score_divisor),\n        PAGE_SIZE=k_cache.shape[1],",
    ),
    (
        "decorator B: split-K kernel do_not_specialize",
        "@triton.jit\ndef _qsa_sparse_paged_gqa_splitk_kernel(",
        '@triton.jit(do_not_specialize=["page_table_width", "stride_table_req"])  # site-31\ndef _qsa_sparse_paged_gqa_splitk_kernel(',
    ),
    (
        "signature B: insert runtime page_table_width",
        "    num_cache_blocks,\n    num_requests,\n    TOPK: tl.constexpr,",
        "    num_cache_blocks,\n    num_requests,\n    page_table_width,  # site-31\n    TOPK: tl.constexpr,",
    ),
    (
        "signature B: drop PAGE_TABLE_WIDTH constexpr",
        "    PAGE_SIZE: tl.constexpr,\n    PAGE_TABLE_WIDTH: tl.constexpr,\n    GROUP_SIZE: tl.constexpr,",
        "    PAGE_SIZE: tl.constexpr,\n    GROUP_SIZE: tl.constexpr,",
    ),
    (
        "body B: mask uses runtime width",
        "            & (logical_page < PAGE_TABLE_WIDTH)",
        "            & (logical_page < page_table_width)  # site-31",
    ),
    (
        "body B: clamp uses runtime width",
        "            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),",
        "            + tl.minimum(logical_page, page_table_width - 1),  # site-31",
    ),
    (
        "launcher B: pass width positionally",
        "        block_table.shape[0],\n        TOPK=logical_indices.shape[1],\n        PAGE_SIZE=k_cache.shape[1],\n        PAGE_TABLE_WIDTH=block_table.shape[1],",
        "        block_table.shape[0],\n        block_table.shape[1],  # site-31\n        TOPK=logical_indices.shape[1],\n        PAGE_SIZE=k_cache.shape[1],",
    ),
]

CACHE_EDITS = [
    (
        "signature C: strides become runtime",
        "    block_table_stride_0: tl.constexpr,\n    block_table_stride_1: tl.constexpr,",
        "    block_table_stride_0,  # site-31\n    block_table_stride_1,  # site-31",
    ),
    (
        "signature C: columns becomes runtime",
        "    num_block_table_columns: tl.constexpr,",
        "    num_block_table_columns,  # site-31",
    ),
    (
        "decorator C: extend do_not_specialize",
        '        "work_search_steps",\n    ]\n)',
        '        "work_search_steps",\n        "block_table_stride_0",  # site-31\n        "block_table_stride_1",  # site-31\n        "num_block_table_columns",  # site-31\n    ]\n)',
    ),
]

EXPECT_MARKERS = {"ops_qsa.py": 9, "qsa_cache.py": 6}


def apply(path: Path, edits) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"[site-31] {path.name}: markers already present — SKIP")
        return
    for desc, old, new in edits:
        n = text.count(old)
        assert n == 1, f"[site-31] {path.name}: anchor for '{desc}' matched {n} times (want 1)"
        text = text.replace(old, new)
    ast.parse(text)  # syntax gate before writing
    path.write_text(text)
    print(f"[site-31] {path.name}: {len(edits)} edits applied, ast OK")


def verify(root: Path) -> None:
    ops = (root / "nvidia" / "ops" / "qsa.py").read_text()
    cache = (root / "common" / "qsa_cache.py").read_text()
    assert "PAGE_TABLE_WIDTH" not in ops, "ops_qsa.py still references PAGE_TABLE_WIDTH"
    assert "num_block_table_columns: tl.constexpr" not in cache
    assert "block_table_stride_0: tl.constexpr" not in cache
    ast.parse(ops)
    ast.parse(cache)
    for name, text in (("ops_qsa.py", ops), ("qsa_cache.py", cache)):
        got = text.count(MARKER)
        want = EXPECT_MARKERS[name]
        assert got == want, f"{name}: {got} markers, want {want}"
    # runtime arg must be threaded end-to-end in both kernels
    assert ops.count("page_table_width") == 7, ops.count("page_table_width")
    print("[site-31] verify OK: 0 width constexprs remain; markers 8+6; ast clean")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/vllm/vllm/models/qwen4_exp")
    apply(root / "nvidia" / "ops" / "qsa.py", OPS_EDITS)
    apply(root / "common" / "qsa_cache.py", CACHE_EDITS)
    verify(root)


if __name__ == "__main__":
    main()
