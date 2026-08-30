#!/usr/bin/env bash
# py-spy-dump-ranks.sh <container-name> [dump-depth]
# Dump python stacks of every VLLM worker rank inside a (possibly hung) container.
# THE tool for silent multi-rank hangs: shows the exact blocking line per rank.
# Usage notes:
#   - py-spy is pip-installed into the live container (ephemeral, no image change)
#   - "idle" thread state + recv/poll frames = blocked (waiting on a peer)
#   - "active" at the SAME frame across two runs 60s apart = spinning, not progressing
#   - correlate with nvidia-smi SM%: 100% + frozen frame = kernel-launch spin;
#     0% + recv frame = comm starvation
#   - dump DEPTH matters: shallow dumps hide the real caller (a PP1 stack looked
#     like generic triton compile at depth 8; its true frame `sample_tokens`
#     appeared at depth 14) — default depth here is 20
set -euo pipefail
C="${1:?usage: py-spy-dump-ranks.sh <container> [depth]}"
D="${2:-20}"
docker exec "$C" bash -c "
pip install -q py-spy 2>/dev/null || true
for p in \$(ps -eo pid,args | grep -E 'VLLM::|EngineCore|PleOffload' | grep -v grep | awk '{print \$1}'); do
  echo \"===== PID \$p : \$(ps -p \$p -o args= | head -c 60)\"
  py-spy dump --pid \$p 2>&1 | sed -n \"1,\$((D+4))p\"
done"
