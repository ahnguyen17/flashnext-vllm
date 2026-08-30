#!/usr/bin/env python3
"""Kernel-load isolation probe — when a triton kernel 'hangs' inside a multi-GPU
vLLM engine, run the EXACT kernel standalone on an idle GPU before theorizing.

Usage (inside the vLLM image, on any idle GPU):
  docker run --rm --gpus '"device=0"' --entrypoint python3 \
    -v this_file:/probe.py -e TRITON_CACHE_DIR=/tmp/tc-probe <image> /probe.py

If FIRST call completes in seconds here but hangs in the engine, the kernel/
driver/silicon are innocent — the trigger is engine context (multi-rank,
CUDA graphs, sibling procs) or the specific physical card. Try the probe on
the failing run's actual card with the engine down, and with
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (pass --vmm).

Pattern proven 8/30 on the FlashNext AWQ PP3 campaign: _post_update_kernel
loads in 1.98s standalone (2.12s under a 4GB VMM alloc) yet spins forever in
cuModuleLoadData inside the engine — see flash-next-170hx skill,
references/pp3-round2-v19-v20-solved.md.
"""
import argparse, time

ap = argparse.ArgumentParser()
ap.add_argument('--vmm', action='store_true', help='allocate 4GB with expandable_segments first')
ap.add_argument('--kernel', default='post_update',
                help='vllm.v1.worker.gpu.input_batch function to exercise')
args = ap.parse_args()

import torch
assert torch.cuda.is_available()
print(f'device: {torch.cuda.get_device_name(0)}', flush=True)

if args.vmm:
    x = torch.empty(4 * 1024**3, dtype=torch.uint8, device='cuda')
    x[-1] = 1
    torch.cuda.synchronize()
    print('4GB expandable-segment alloc OK', flush=True)

from vllm.v1.worker.gpu import input_batch
fn = getattr(input_batch, args.kernel)
kernel = getattr(input_batch, f'_{args.kernel}_kernel', None)
print(f'exercising {args.kernel} (kernel obj: {kernel})', flush=True)

# Decode-shaped dummies (match the shape class that fails in-engine; adjust
# vocab/max_model_len to the model under test).
max_reqs, vocab, mml = 2, 150000, 512
tensors = dict(
    idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device='cuda'),
    num_computed_tokens=torch.zeros(max_reqs, dtype=torch.int32, device='cuda'),
    last_sampled_tokens=torch.zeros(max_reqs, dtype=torch.int64, device='cuda'),
    output_bin_counts=torch.zeros(max_reqs, vocab, dtype=torch.int32, device='cuda'),
    sampled_tokens=torch.zeros(2, 4, dtype=torch.int64, device='cuda'),
    num_sampled=torch.tensor([4, 4], dtype=torch.int32, device='cuda'),
    num_rejected=torch.tensor([1, 0], dtype=torch.int32, device='cuda'),
    query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32, device='cuda'),
    all_token_ids=torch.zeros(max_reqs, mml, dtype=torch.int64, device='cuda'),
    total_len=torch.tensor([10, 9], dtype=torch.int64, device='cuda'),
)
t0 = time.time()
fn(**tensors)
torch.cuda.synchronize()
print(f'FIRST call (compile+load+launch): {time.time()-t0:.2f}s', flush=True)
t0 = time.time()
fn(**tensors)
torch.cuda.synchronize()
print(f'SECOND call: {time.time()-t0:.4f}s', flush=True)
print('PROBE PASSED — this kernel loads fine on THIS gpu in isolation', flush=True)
# Wrap in host `timeout NNN docker run …` — a hang here means the card/driver
# pair is the trigger, not engine context.
