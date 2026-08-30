#!/usr/bin/env python3
"""GGUF tensor inventory — parse headers of (possibly split) GGUF shards and
report tensor names/sizes by group. Used 2026-08-28 to discover that
per_layer_token_embd.weight (24.8 GiB) is the n-gram/PLE lookup table.

Usage: python3 gguf-tensor-inventory.py <shard1> [shard2 ...]
Notes:
- GGUF v3 KV types: 8=STRING (not u64!), 9=u64, 10=i64, 11=f64, 12=array, 13=f16, 14=bf16
- gguf-py's GGUFReader only indexes tensors local to each shard — for split
  GGUFs parse each shard's raw header and aggregate (Unsloth UD splits keep
  shard 1 KV-only; tensor infos start in shard 2).
"""
import struct, sys, re
from collections import defaultdict

SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,9:8,10:8,11:8,13:2,14:2}
def rd(f, t):
    if t in SZ: f.read(SZ[t])
    elif t == 8:
        ln, = struct.unpack('<Q', f.read(8)); f.read(ln)
    elif t == 12:
        bt, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
        for _ in range(n): rd(f, bt)
    else: raise ValueError(f'kv type {t}')

# approximate bytes-per-param by ggml type id
BPP = {0:4,1:4,2:2,3:2,4:0.5625,5:0.625,6:0.6875,7:0.75,8:1.03125,9:0.414,10:0.523,
       11:0.566,12:0.66,13:0.758,14:0.3125,15:0.348,16:0.417,17:0.25,18:0.3125,
       19:0.5625,20:0.521,21:0.29,22:0.375,23:0.375,24:0.406,25:0.479,26:0.51,
       27:2,28:4,29:2,30:2}

tensors = []
for path in sys.argv[1:]:
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF', f'{path}: not GGUF'
    f.read(4)  # version
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    for _ in range(n_kv):
        ln, = struct.unpack('<Q', f.read(8)); f.read(ln)
        vt, = struct.unpack('<I', f.read(4)); rd(f, vt)
    for _ in range(n_tensors):
        ln, = struct.unpack('<Q', f.read(8)); name = f.read(ln).decode()
        nd, = struct.unpack('<I', f.read(4)); dims = struct.unpack(f'<{nd}Q', f.read(8*nd))
        ttype, = struct.unpack('<I', f.read(4)); off, = struct.unpack('<Q', f.read(8))
        ne = 1
        for d in dims: ne *= d
        tensors.append((name, ne*BPP.get(ttype, 0.5), ttype, dims))

groups = defaultdict(lambda: [0.0, 0])
for name, b, _, _ in tensors:
    key = ('PLE-ngram' if 'ple' in name.lower() else
           'per_layer_token_embd' if 'per_layer_token_embd' in name else
           'token_embd' if 'token_embd' in name else
           'experts' if '_exps' in name else 'backbone')
    groups[key][0] += b; groups[key][1] += 1

total = sum(v[0] for v in groups.values())
for k, (b, c) in sorted(groups.items(), key=lambda x: -x[1][0]):
    print(f"{k:24} {b/2**30:7.1f} GiB ({c:5} tensors)")
print(f"{'TOTAL':24} {total/2**30:7.1f} GiB ({len(tensors)} tensors)")
print("\nTOP 12:")
for name, b, tt, dims in sorted(tensors, key=lambda x: -x[1])[:12]:
    print(f"  {name:50} {b/2**30:6.2f} GiB dims={dims}")
