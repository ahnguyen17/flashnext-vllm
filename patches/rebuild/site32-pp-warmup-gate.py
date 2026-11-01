#!/usr/bin/env python3
"""site-32 reconstruction: env-gate kernel warmup at PP>1 (port of the original
site9/site10/site30 behavior for the rebuilt image).

Original behavior: warmup skipped at PP>1 (3090-rank triton module-load
pathology wedged it) unless VLLM_PP_WARMUP=1, in which case warmup runs at
BOOT where VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 tolerates the slow
first-use compiles and seeds the persistent triton cache."""
import ast, pathlib

P = pathlib.Path("/opt/vllm/vllm/v1/worker/gpu/warmup.py")
src = P.read_text()

if "site-32-recon" in src:
    print("[site-32-recon] already applied")
    raise SystemExit(0)

OLD = "    if model_runner.is_encoder_only:\n        return\n"
NEW = ("    if model_runner.is_encoder_only:\n        return\n"
       "    # site-32-recon: at PP>1, kernel warmup triggers the 3090-rank\n"
       "    # triton module-load pathology. Skip unless VLLM_PP_WARMUP=1 (boot-\n"
       "    # time warmup then seeds the persistent cache under the 7200s fuse).\n"
       "    from vllm.distributed.parallel_state import get_pp_group as _gpp32\n"
       "    if _gpp32().world_size > 1 and os.environ.get(\"VLLM_PP_WARMUP\", \"\") != \"1\":\n"
       "        logger.info(\"site-32-recon: skipping kernel warmup at PP>1\")\n"
       "        return\n")

n = src.count(OLD)
assert n == 1, f"anchor warmup-head count {n}"
src = src.replace(OLD, NEW)

IMP_OLD = "import numpy as np\nimport torch\n"
IMP_NEW = "import numpy as np\nimport os\nimport torch\n"
assert src.count(IMP_OLD) == 1, "import anchor"
src = src.replace(IMP_OLD, IMP_NEW)

ast.parse(src)
P.write_text(src)
print("[site-32-recon] applied + ast ok")
