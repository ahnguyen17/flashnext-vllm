#!/usr/bin/env python3
"""site-33: PLE offload worker must not inherit VLLM_PP_LAYER_PARTITION.

The PLE worker runs in its own world-1 distributed group; the asymmetric
partition env (28,8,12 for the 3-rank GPU engine) is meaningless there and
crashes get_pp_indices when the worker's meta model is constructed
(len(partitions)=3 != pp_size=1) under MTP configs. Pop it at proc entry."""
import ast, pathlib

P = pathlib.Path("/opt/vllm/vllm/v1/ple_offload/worker.py")
src = P.read_text()
if "site-33" in src:
    print("[site-33] already applied"); raise SystemExit(0)

OLD = '        """Load PLE weights, accept registrations, and run the request loop."""\n        decorate_logs("PleOffloadWorker")\n'
NEW = ('       """Load PLE weights, accept registrations, and run the request loop."""\n'
       '        # site-33: this process is its own world-1 group; the GPU engine\'s\n'
       '        # asymmetric partition env would crash get_pp_indices here.\n'
       '        import os as _os33\n'
       '        _os33.environ.pop("VLLM_PP_LAYER_PARTITION", None)\n'
       '        decorate_logs("PleOffloadWorker")\n')
NEW = NEW.replace('       ""', '        ""', 1)
n = src.count(OLD)
assert n == 1, f"anchor count {n}"
src = src.replace(OLD, NEW)
ast.parse(src)
P.write_text(src)
print("[site-33] applied + ast ok")
