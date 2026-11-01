#!/usr/bin/env python3
"""site-34: gptq_marlin_repack compat with the prebuilt _C op.

The rebuild links the PR-tree Python against the nightly base's PRECOMPILED
_C extension, whose gptq_marlin_repack predates the is_a_8bit argument.
is_a_8bit=False (the AWQ-int4 path this model uses) maps exactly onto the
5-arg legacy call; only the 8-bit-activation path needs the new binary."""
import ast, pathlib

P = pathlib.Path("/opt/vllm/vllm/_custom_ops.py")
src = P.read_text()
if "site-34" in src:
    print("[site-34] already applied"); raise SystemExit(0)

OLD = ("    return torch.ops._C.gptq_marlin_repack(\n"
       "        b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit\n"
       "    )\n")
NEW = ("    # site-34: prebuilt _C predates the is_a_8bit arg; the False path\n"
       "    # is exactly the legacy 5-arg behavior (AWQ int4).\n"
       "    try:\n"
       "        return torch.ops._C.gptq_marlin_repack(\n"
       "            b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit\n"
       "        )\n"
       "    except (TypeError, RuntimeError):\n"
       "        assert is_a_8bit is False, \"this _C build lacks is_a_8bit support\"\n"
       "        return torch.ops._C.gptq_marlin_repack(\n"
       "            b_q_weight, perm, size_k, size_n, num_bits\n"
       "        )\n")
n = src.count(OLD)
assert n == 1, f"anchor count {n}"
src = src.replace(OLD, NEW)
ast.parse(src)
P.write_text(src)
print("[site-34] applied + ast ok")
