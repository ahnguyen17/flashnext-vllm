#!/usr/bin/env python3
"""Instrumented stall probe for a serving engine that boots but dies/stalls on request #N.

Pattern (proven 8/29 on the vLLM PP3 saga — caught the `post_update` triton
wedge live after two patches to the WRONG sibling kernel):
  1. Per-token timestamps on the failing request  — "exactly 1 token then hang"
     means the trigger is the SECOND decode step; uniform slowness means
     something else entirely.
  2. Background stack sampler — dumps every worker rank's python stack every
     30s DURING the stall. The frame sitting in the same triton/recv spot
     across consecutive dumps is the culprit, caught in the act.
  3. Same-shape request ladder — if request #2 fails identically to a
     same-shape #1, the trigger is engine state from #1, not the new shape.

Also note the duration tell: near-identical "slow" durations across separate
runs (301.8 / 300.7 / 305.3s) = a timer/internal-recovery constant, NOT
variance. Stop calling it JIT.

Usage: python3 stall-probe.py [--url http://127.0.0.1:8003] [--key change-me]
       [--container qwen38-pp3-bench] (container+rank-grep only wired for the
       VLLM::Worker_* naming; adjust for other engines).
"""
import argparse, json, subprocess, threading, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:8003/v1/completions')
p.add_argument('--key', default='change-me')
p.add_argument('--model', default='flash-next')
p.add_argument('--container', default='qwen38-pp3-bench')
args = p.parse_args()
H = {'Content-Type': 'application/json', 'Authorization': f'Bearer {args.key}'}

DUMP = ('pip install -q py-spy 2>/dev/null; '
        'for R in PP0 PP1 PP2; do P=$(ps -eo pid,args | grep "Worker_$R" | grep -v grep '
        '| awk "{print \\$1}"); echo "--- $R:"; py-spy dump --pid $P 2>&1 | sed -n "4,9p"; done')

def dump_ranks(tag):
    try:
        out = subprocess.run(['docker', 'exec', args.container, 'bash', '-c', DUMP],
                             capture_output=True, text=True, timeout=60).stdout
        print(f'=== STACKS {tag} ===\n{out}', flush=True)
    except Exception as e:
        print(f'stack dump failed: {e}', flush=True)

def stream_timed(prompt, n, label, eos=True):
    body = json.dumps({'model': args.model, 'prompt': f'[nonce:{time.time()}] {prompt}',
                       'max_tokens': n, 'temperature': 0, 'stream': True,
                       'ignore_eos': not eos}).encode()
    t0 = time.time(); stamps = []
    try:
        with urllib.request.urlopen(urllib.request.Request(args.url, data=body, headers=H), timeout=900) as r:
            for line in r:
                if not line.startswith(b'data: '): continue
                d = line[6:].strip()
                if d == b'[DONE]': break
                ch = json.loads(d).get('choices', [{}])[0]
                if ch.get('text'): stamps.append(time.time() - t0)
        if stamps:
            gen = stamps[-1] - stamps[0]
            print(f'{label}: {len(stamps)} tok | ttft {stamps[0]:.1f}s | gen {gen:.1f}s | '
                  f'{len(stamps)/gen if gen > 0 else 0:.1f} t/s', flush=True)
            print(f'  token times: {[round(s,1) for s in stamps[:12]]}', flush=True)
            return True
        print(f'{label}: EMPTY response', flush=True); return False
    except Exception as e:
        print(f'{label}: FAILED {type(e).__name__} {e}', flush=True); return False

stop = threading.Event()
def sampler():
    time.sleep(45)
    for k in range(8):
        if stop.is_set(): return
        dump_ranks(f'req1+{45 + k*30}s')
        time.sleep(30)
threading.Thread(target=sampler, daemon=True).start()

ok1 = stream_timed('Count from 1 to 5.', 64, 'REQ1_warmup', eos=True)
stop.set()
ok2 = stream_timed('Count from 6 to 10.', 64, 'REQ2_same_shape', eos=True)
ok3 = stream_timed('List three colors.', 64, 'REQ3_same_shape', eos=True)
if ok3:
    stream_timed('The theory of quantum electrodynamics explains', 256, 'REQ4_long', eos=False)
print('=== PROBE DONE ===', flush=True)
