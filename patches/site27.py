#!/usr/bin/env python3
"""site-27: prefix-caching correctness family — port of zebgop-ops/qwen38-flashnext-pp
patches 0010 + 0011 + 0012 + 0013 (their production-validated set; 0010 alone
soaked clean on their rig, the rest are adjacent-path hardening).

Upstream fixes, all verified against OUR pp3fix22 tree 2026-09-03:
  27a  mamba_utils.py:  copy kernel indexes SOURCE req-slot block tables by
      req_idx, never stale batch row (THE fix — deferred postprocess on non-last
      PP ranks walked current tables with a stale batch mapping, copying
      recurrent state through other requests' block ids -> NaN corruption).
  27b  mamba_utils.py:  second kernel (align precopy) same req_idx indexing.
  27c  single_type_kv_cache_manager.py (MambaManager.find_longest_cache_hit):
      port of unmerged vllm#48375 — drop the final matched page on cache hit
      for EAGLE/MTP (its state snapshot may contain rejected draft tokens).
  27d  mamba_hybrid.py: state-seed divisor = mamba group block size, not
      attention/CLI block size (vllm#53142 — resume-over-cached-prefix seeded
      out-of-range block_table column -> garbage block id in align precopy).
  27e  qwen4_exp/nvidia/ple_layer.py: zero-init speculative-extension columns
      [conv_state_len:] on prefill state init (tail is never-written VRAM =
      NaN reservoir; spec-decode rollback gather can reach it first).

Usage: python3 site27.py [DIR]   (DIR = host dir with bare filenames for
dry-run; default = container /opt/vllm paths)
"""
import ast, sys

FILES = {
    "mamba_utils": "/opt/vllm/vllm/v1/worker/mamba_utils.py",
    "stkcm": "/opt/vllm/vllm/v1/core/single_type_kv_cache_manager.py",
    "mamba_hybrid": "/opt/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
    "ple_layer": "/opt/vllm/vllm/models/qwen4_exp/nvidia/ple_layer.py",
}

EDITS = [
    # ---- 27a: mamba_utils.py, gathered-table row indexing ----
    (FILES["mamba_utils"],
     "    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx\n",
     "    # site-27a (port zebgop-ops 0010): the captured block tables are the\n"
     "    # SOURCE per-request-slot tables, so always index them by req_idx.\n"
     "    # The previous batch-row indexing read the CURRENT step's gathered\n"
     "    # tables at a (possibly stale) batch row -- under PP the deferred\n"
     "    # postprocess runs pp_size steps after its batch was gathered, so\n"
     "    # batch rows point at DIFFERENT requests and the align state copy\n"
     "    # walks another request's block ids.\n"
     "    bt_row_idx = req_idx\n"),
    # ---- 27b: mamba_utils.py, second kernel (align precopy) ----
    (FILES["mamba_utils"],
     "    token_bias = tl.load(token_bias_ptr + req_idx)\n"
     "    _copy_mamba_state_block(\n"
     "        state_idx,\n"
     "        batch_idx,\n",
     "    token_bias = tl.load(token_bias_ptr + req_idx)\n"
     "    _copy_mamba_state_block(\n"
     "        state_idx,\n"
     "        req_idx,  # site-27b (0010): source tables are req-indexed\n"),
    # ---- 27c: MambaManager honors drop_eagle_block ----
    (FILES["stkcm"],
     '        assert dcp_world_size == 1, "DCP not support mamba now."\n'
     '        assert pcp_world_size == 1, "PCP not support mamba now."\n'
     "        block_hashes = resolve_block_hashes(\n",
     '        assert dcp_world_size == 1, "DCP not support mamba now."\n'
     '        assert pcp_world_size == 1, "PCP not support mamba now."\n'
     "        # site-27c (port zebgop-ops 0011 / unmerged vllm#48375):\n"
     "        # EAGLE/MTP requires the final matched page of a cache hit be\n"
     "        # dropped -- its recurrent-state snapshot may include draft\n"
     "        # tokens that verification later rejects; resuming from it\n"
     "        # corrupts the GDN/mamba state. Lowering max_length by one\n"
     "        # mamba block == upstream's max_num_blocks-1 for the coarse\n"
     "        # loop and also bounds the partial-unit branch.\n"
     "        if drop_eagle_block:\n"
     "            max_length = max(0, max_length - kv_cache_spec.block_size)\n"
     "        block_hashes = resolve_block_hashes(\n"),
    # ---- 27d: state-seed divisor = mamba block size ----
    (FILES["mamba_hybrid"],
     "            self._mamba_state_idx_gpu[req_index].fill_(\n"
     "                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size\n",
     "            # site-27d (port zebgop-ops 0012, vllm#53142): divisor must\n"
     "            # be the MAMBA group's block size, not the attention/CLI\n"
     "            # block size -- on hybrids they differ, and a resume over a\n"
     "            # cached prefix then seeds an out-of-range block_table\n"
     "            # column -> garbage block id read by the fused align\n"
     "            # precopy (silently wrong GDN state where the read stays\n"
     "            # mapped). _mamba_spec is set by the first batch's\n"
     "            # preprocess; fresh requests seed from 0 either way.\n"
     "            _mamba_bs = (\n"
     "                self._mamba_spec.block_size\n"
     "                if self._mamba_spec is not None\n"
     "                else self.cache_config.block_size\n"
     "            )\n"
     "            self._mamba_state_idx_gpu[req_index].fill_(\n"
     "                (new_req_data.num_computed_tokens - 1) // _mamba_bs\n"),
    # ---- 27e: zero-init PLE spec-extension tail on prefill ----
    (FILES["ple_layer"],
     "            existing_state[..., : self.conv_state_len] = safe_next_state\n"
     "            conv_state.index_copy_(0, state_indices, existing_state)\n",
     "            existing_state[..., : self.conv_state_len] = safe_next_state\n"
     "            # site-27e (port zebgop-ops 0013): zero-init the\n"
     "            # speculative-extension columns [conv_state_len:] when\n"
     "            # prefill (re)inits a state block. Prefill/decode only\n"
     "            # write [0:conv_state_len]; on a fresh cache block the tail\n"
     "            # is uninitialized VRAM (NaN patterns), and the spec-decode\n"
     "            # rollback gather can reach into it before the first full\n"
     "            # spec write heals it. Zero = 'no history'.\n"
     "            if existing_state.shape[-1] > self.conv_state_len:\n"
     "                _tail = existing_state[..., self.conv_state_len :]\n"
     "                existing_state[..., self.conv_state_len :] = torch.where(\n"
     "                    update_mask.view(num_prefills, 1, 1).expand_as(_tail),\n"
     "                    torch.zeros_like(_tail),\n"
     "                    _tail,\n"
     "                )\n"
     "            conv_state.index_copy_(0, state_indices, existing_state)\n"),
]


def main():
    host_dir = sys.argv[1] if len(sys.argv) > 1 else None
    for path, old, new in EDITS:
        p = f"{host_dir}/{path.rsplit('/', 1)[-1]}" if host_dir else path
        src = open(p).read()
        tag = "site-27" + new.split("site-27", 1)[1][:1]
        if new in src and old not in src:
            print(f"[{p}] {tag}: already applied")
            continue
        n = src.count(old)
        assert n == 1, f"[{p}] {tag}: anchor count {n} (need 1)"
        src = src.replace(old, new)
        ast.parse(src)  # syntax gate
        open(p, "w").write(src)
        print(f"[{p}] {tag}: applied + ast-gated OK")
    print("site-27: all edits done")


if __name__ == "__main__":
    main()
