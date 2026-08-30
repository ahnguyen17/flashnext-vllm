# PP3 Round 2, act 3 — v18 falsifies the clean fix; mixed-arch triton cache poisoning; site-13 / v19

Date: 2026-08-30 night. Continues `pp3-round2-v17-verdict.md` (v17 livelock, sites 11/12 retracted as regressions).

## v18 — the clean combination, FALSIFIED

Config: image `pp3fix4` (sites 8-10 only, NO kernel ports) + `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`, otherwise v14-verbatim (devices 1,2,3; `VLLM_PP_LAYER_PARTITION=8,28,12` @ 0.85; 32K; seqs 2; MTP-3; both distributed timeouts 3600). Launcher `~/pp18-launch.sh`, watcher `~/pp18-watch.sh` → `~/pp18-watch.log` (tee'd to file — see ops lessons).

Result: boot clean (~13 min). `warm-64` burned the FULL 1800s client ceiling. Watcher auto-dumped all ranks:

- Worker_PP0: `post_update (input_batch.py:627)` → `triton _init_handles (compiler.py:469)` — the ORIGINAL kernel, no ports involved
- Worker_PP1/PP2: idle in `irecv_tensor_dict`/`recv`
- EngineCore: idle in `dequeue`/`get_response`
- dmesg: NO Xid/nvrm faults; PP0 R-state ~91% CPU, 100% SM

Meaning: with 30 minutes of grace the module-load still never completed. **Timeout-env is necessary but NOT sufficient.** The "~300s slow init" model only held for v10. And v10 — same image, same kernel, same request shape — COMPLETED (64 tok, 301.8s). The only meaningful delta between v10 and v18: **compile-cache state.**

## Leading theory: shared TRITON_CACHE_DIR cross-arch poisoning

- The rig is MIXED-ARCH: RTX 3090 = sm_86; unlocked CMP 170HX (GA100) = sm_80.
- All ranks share one `TRITON_CACHE_DIR` (same container FS: `/root/.cache/vllm/triton` under the `vllm-cache` volume).
- If an sm_80 rank wins the compile race for a kernel, the sm_86 rank loads a wrong-arch cubin → `cuModuleLoadData` spins forever: 100% CPU, 100% SM, no error, no dmesg fault.
- v10 = the lucky race (the 3090's own compile landed first → its cubin loaded in ~300s).
- v12–v18 = poisoned cache (or, on pp3fix6, the ports' own livelock masking it).
- v14's "fresh cache control" does NOT contradict the theory: pp3fix6's broken ports dominated that run; a fresh-cache test of the CLEAN image had never run until v19.

## Site-13: per-rank TRITON_CACHE_DIR (structural fix)

`/opt/vllm/vllm/v1/executor/multiproc_executor.py`, in `worker_main`, anchor `rank = kwargs.get("rank", 0)` (verify count==1 before replacing):

```python
rank = kwargs.get("rank", 0)
# site-13: per-rank TRITON_CACHE_DIR. On mixed-arch rigs (sm_86 +
# sm_80) a shared cache lets one rank load another rank's cubin,
# and cuModuleLoadData spins forever on the wrong-arch binary.
try:
    _tc = os.environ.get("TRITON_CACHE_DIR", "/root/.cache/vllm/triton")
    os.environ["TRITON_CACHE_DIR"] = f"{_tc}-rank{rank}"
except Exception:
    pass
```

Build recipe: run container from base image → `docker cp` patch script in → `docker exec` by path → grep-verify → `docker commit --change 'ENTRYPOINT ["vllm"]' <ctr> <tag>`. Image: `qwen38-flash-next:pp3fix7` (pp3fix4 + site-13).

⚠️ **Heredoc into `docker exec ... python3 - <<'EOF'` was SILENTLY SWALLOWED** (exit 0, no output, patch never applied, discovered only on grep). Patch scripts must be docker cp'd and exec'd by path; ALWAYS grep-verify after.

## v19 (verdict pending at this write)

`~/pp19-launch.sh`: image pp3fix7 + FRESH volume `vllm-cache-v19` + env 7200 + v14-verbatim config. Launched 20:44:52 8/30; watcher proc logs to `~/pp19-watch.log` (ladder: warm-64 → warm-repeat → decode-256 → decode-512; abort + auto stack-dump on any failure).

Why both levers: the fresh volume reproduces v10's compile-then-load path (no inherited poison); site-13 makes cross-arch poisoning structurally impossible for every future boot regardless of race outcomes.

Decision tree:
- **v19 serves** → fix = sites 8-10 + 13 + `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`; upstream-reportable (mixed-arch multi-GPU rigs). Ladder gives the decode number vs the ≥50 t/s bar.
- **v19 still hangs** → last diagnostic = no-MTP control run (isolates the spec-decode kernel class as sole casualty), then close-out + upstream report with the full evidence chain.

## Ops & comms lessons (this window)

- Watcher blindness fixed: `bash watch.sh > file.log 2>&1` with `python3 -u` inside; tail the file for live progress. During live runs ground truth remains: docker logs + `/metrics` counters + py-spy.
- **Build-budget comms**: the operator asked "how many more builds do we have to do?" mid-campaign. Answer with a hard number + what each build decides ("1 decisive, 2 at most"), then hold to it — that framing satisfied him. He also asked for a mid-campaign summary + roadmap; a standing table (mission / current state / decision tree / parked decisions) worked cleanly.
- **Interrupt semantics**: the operator cancelled a mid-teardown `write_file` once — it was a question-injection point, not a stop order. On interrupt: summarize state crisply, offer go/no-go, don't barrel on silently. He re-authorized with "just do whatever you can to get this working."
