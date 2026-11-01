#!/usr/bin/env python3
"""site-28: per-rank KV cache budgets — port of zebgop-ops patch 0014.

--kv-cache-memory is a single value for every worker, but under PP the ranks
are heterogeneous (PLE machinery on rank 0, MTP drafter on the last rank,
different per-token KV costs), so one global value must fit the tightest rank
and strands memory everywhere else (our own rule: "never set --kv-cache-memory
on asymmetric partitions" — this replaces that limitation). Each rank takes
its own absolute byte budget from VLLM_KV_CACHE_MEMORY_RANK<i>.

Their measured: +19.4% KV pool on heterogeneous PP ranks.
Our prod boot log 9/3 suggests: PP0 2.25G -> 5.07G possible, PP2 4.35G ->
9.48G possible (read fresh values at window time).

Usage: python3 site28.py [DIR]   (DIR = host dir with gpu_worker.py for
dry-run; default = container path)
"""
import ast, sys

PATH = "/opt/vllm/vllm/v1/worker/gpu_worker.py"
OLD = (
    "        maybe_apply_startup_plan(self)\n"
    "\n"
    "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
)
NEW = (
    "        maybe_apply_startup_plan(self)\n"
    "\n"
    "        # site-28 (port zebgop-ops 0014): per-rank KV budget override.\n"
    "        # See skill vllm-pp-debug; env VLLM_KV_CACHE_MEMORY_RANK<i> sets\n"
    "        # rank i's absolute KV byte budget (global --kv-cache-memory\n"
    "        # still wins if set).\n"
    "        import os as _os\n"
    "        _pp_rank = get_pp_group().rank_in_group\n"
    "        _override = _os.getenv(f\"VLLM_KV_CACHE_MEMORY_RANK{_pp_rank}\")\n"
    "        if _override and not self.cache_config.kv_cache_memory_bytes:\n"
    "            self.cache_config.kv_cache_memory_bytes = int(_override)\n"
    "            logger.info(\n"
    "                \"site-28 per-rank KV budget: pp_rank=%d takes %s GiB \"\n"
    "                \"from VLLM_KV_CACHE_MEMORY_RANK%d\",\n"
    "                _pp_rank, format_gib(int(_override)), _pp_rank,\n"
    "            )\n"
    "\n"
    "        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:\n"
)


def main():
    host_dir = sys.argv[1] if len(sys.argv) > 1 else None
    p = f"{host_dir}/gpu_worker.py" if host_dir else PATH
    src = open(p).read()
    if NEW in src and OLD not in src:
        print(f"[{p}] site-28: already applied")
        return
    n = src.count(OLD)
    assert n == 1, f"[{p}] site-28: anchor count {n} (need 1)"
    src = src.replace(OLD, NEW)
    ast.parse(src)
    open(p, "w").write(src)
    print(f"[{p}] site-28: applied + ast-gated OK")


if __name__ == "__main__":
    main()
