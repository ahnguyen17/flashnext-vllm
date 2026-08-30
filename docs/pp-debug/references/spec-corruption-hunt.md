# Spec-decode corruption hunt (v26 → v29, 2026-08-30 night)

Post-deadlock phase. Engine serves (site-18 fixed the ring) but text corrupts.

## Observed signatures (all temperature=0)
| Build | Sites | Output |
|---|---|---|
| v26 (pp3fix10) | +15 (pure-torch port) | `" Paris.\n\n# \n\n## \n\n\n\n!#"` — right first tokens, then degeneration |
| v27 (pp3fix11) | +19 (qsl=None → apply −num_rejected) | `"The capital capital! of of! France France! is is!! Paris Paris!"` — token doubling |
| v28 (pp3fix12) | +20 (reference triton kernel + dummy histogram on non-last ranks) | `"Siliconicon hearts… ParallelParallel streams… a a storm… NoteNote: :"` — same doubling |

Rates were fine (~30–42 t/s); generations ended early despite `ignore_eos` (209/256, 443/512) — consistent with num_computed/total_len drift making the scheduler believe sequences finished.

## Interpretation
- v28 is the decisive cell: reference kernel ⇒ same doubling ⇒ the ring's **consume-side integration** double-appends tokens on non-last ranks regardless of kernel implementation.
- Two suspects for the double-append: (a) ring consume appends tokens that another non-last-rank path also appends; (b) queue cadence after site-18 made every step's entry real (mask now data-only), interacting badly with steps where the scheduler also replays/appends.
- v29 (pp3fix13, site-21): `update_pp_decode_requests` early-returns — if text is clean, non-last ranks didn't need the sync at all in PP3+MTP+PLE config (scheduler is centralized; non-last ranks' all_token_ids only feed local bookkeeping). Ship that, note the cadence bug upstream. If still doubled → the collision is on the LAST rank: diff `_update_states_after_model_execute` (old-file path, appends sampled ids) vs the triton post_update append on the last rank.

## site-19 detail (real port bug found by signature change)
`_post_update_kernel` computes `computed_delta = query_len − num_rejected` and applies it whenever nonzero — with `query_start_loc=None` (the ring path) that is `−num_rejected`. The pure-torch port wrapped the whole adjustment in `if query_start_loc is not None:` → non-last ranks' num_computed inflated by the rejection count every spec step → desync. Fix:
```python
qlen = (qsl[1:] - qsl[:-1]).long() if qsl is not None else torch.zeros(num_reqs, dtype=torch.long, device=...)
delta = qlen[valid] - num_rejected[valid].long()
```
The v26→v27 corruption MODE change (degeneration → doubling) was the tell that the fix moved the arithmetic but another error remained.

## Quality-check protocol for spec-decode patches
1. Short factual completion — first tokens must be right AND stay right for 20+ tokens.
2. Long gen (48+) — look for doubled substrings (`"word word"`, `"Siliconicon"`).
3. Full-length check — `ignore_eos: true`, `max_tokens: 256` must return ~256.
4. `/metrics` spec counters — drafts/accepted > 0 (drafter alive) independent of text quality.

## ROOT CAUSE FOUND (v30 trace build, 2026-08-30 ~02:30)

The consume-side double-append theory was WRONG. pp3fix14 (pp3fix12 + 6 TRACE
prints at scheduler AFT/SPEC-OUT/OUT + RING/CONSUME) nailed it in one boot:

**Phantom first-decode draft leak.** At the FIRST decode step after prefill,
async-scheduling placeholder spec ids (`[-1]*num_spec`, set by
`AsyncScheduler._update_after_schedule` L44) count into
`num_tokens_with_spec`, so `num_new_tokens = nts + ph − nct` inflates from 1
to 4 (MTP-3). But the scheduler-side PLE/n-gram drafter has NO output history
yet → L706's `num_scheduled_spec_tokens` formula evaluates 0 → zero drafts
enter `scheduled_spec_decode_tokens` → the worker sees `num_draft_tokens=0`
→ plain sampler, 1 token out. All 4 scheduled slots still advance
`num_computed_tokens` (+4 with only 1 real token) and NO rejection rollback
ever lands for the 3 phantom slots (rollback requires a non-empty
scheduled_spec entry at output time). The +3 gap is PERMANENT; every later
input gather reads 3 positions past the real token stream → prompt tokens
re-fed into context → `"The capital capital! of of! France France!..."`.
From the 2nd decode step on, everything is healthy (advance 4, rollback rej,
net = gen on scheduler/last-rank/ring-consumer — all three consistent).
Also explains the early endings despite ignore_eos (drifted accounting).

Trace excerpt (the whole bug in 4 lines):
```
AFT step=1  sched=5 nct=5 nts=5   # prefill ok
AFT step=4  sched=4 nct=9 nts=5   # 4 scheduled, only 1 can be real — no SPEC-OUT this step
OUT step=4  gen=1 appended=1      # plain sampling (0 drafts) — +3 leak born
SPEC-OUT step=7 draft=3 rej=2     # from here on machinery is self-consistent
```

**Fix iterations (post-root-cause):**
- **v31/pp3fix15** ("first decode only" guard): step 4 clean (sched=1) but the leak
  moved to step 7 — the drafter stays empty on generic prompts, so the condition
  is *placeholder-ness*, not *first-decode-ness*.
- **v32/pp3fix16** (site-22, guard on `spec_token_ids[0] == -1`): **CLEAN TEXT on
  every prompt**, accounting airtight (`nct = prompt + outlen` at every step).
  BUT: zero SPEC-OUT events all run — dropping placeholder slots starves the
  last rank's MTP speculator (`speculator.propose` needs scheduled draft slots
  to extend), so drafting never engages → MTP inert at ~36 t/s (1 tok per 3-step
  PP cadence) vs no-MTP's 57. Correct but slow.
- **v33/pp3fix17 (engagement design)**: schedule placeholders freely (feeds the
  drafter), roll back unverified slots on ALL THREE accountants so every step
  nets `len(generated)`:
  - **site-23 (scheduler)**: `request.num_phantom_spec_slots` set in `schedule()`
    when placeholder ids inflate num_new but don't enter the dict (formula ≤ 0);
    rolled back in `update_from_output` (+ `num_output_placeholders`).
  - **site-24 (last rank, `sample()` plain branch)**: report
    `num_rejected = query_len − num_sampled` for non-prefilling rows where the
    gap ≤ num_spec — last rank's own post_update (+qsl − rej) and the ring
    consumers (+qlen optimistic at receive, − rej at consume) both net to
    num_sampled. Prefill rows masked via `is_prefilling_np`; gap-signature check
    is the second safety net.
  - Gotcha: V2 runner attr is `self.num_speculative_steps` (NOT num_spec_tokens);
    a wrong getattr silently zeros the guard and disables the correction.
  - **v33/pp3fix17**: engagement restored (SPEC-OUTs, 62–64 t/s) but text
    doubled again — the phantom flag lived on the REQUEST and async
    scheduling runs schedule(N+1) (zeroing it) before update_from_output(N).
    Race, not logic.
  - **v34/pp3fix18**: race fixed (flag moved to SchedulerOutput.phantom_spec_slots
    — travels with its own step) — STILL doubled, no PHANTOM fired: the
    `if num_scheduled_spec_tokens > 0` branch ran before the elif and put the
    **placeholder ids [-1,-1,-1] into scheduled_spec_decode_tokens as if they
    were real drafts** → worker's rejection sampler consumed token-id -1 as
    model input → garbage tokens in context ("10$1$! 1$!$$!"). Two distinct
    corruption mechanisms found in total: (1) accounting leak (phantom slots
    never rolled back), (2) placeholder ids fed as drafts.
  - **v35/pp3fix19**: placeholder ids withheld from the dict → engine CRASH:
    `assert num_output_placeholders >= 0` (async_scheduler:63). The phantom
    rollback must NOT decrement ph — phantom slots entered via
    num_tokens_with_spec, never via the async reservation. Also learned: S24
    fired with rej0=0 — adaptive verification compacts the worker's schedule
    to 1+drafts, so with 0 drafts the worker computes ONE query token and
    never leaks worker-side on phantom steps.
  - **v36/pp3fix20**: crash fixed, but text = TOTAL garbage (mojibake) and
    ZERO engagement — sanitizing drafts starves the in-forward drafter.
  - **v37/pp3fix21 (ground truth)**: slot-markers-in-dict + race-free ledger
    + ids trace → **`ids=[-1,-1,-1]` on EVERY SPEC-OUT**. The worker does NOT
    fill the -1 markers with real proposals before verification — the
    rejection sampler verifies the raw markers as draft tokens and "accepts"
    garbage (-1-ish ids → "!", "$" tokens) into the stream. Engagement is
    real (62-68 t/s) but every accepted draft is garbage.

**FORK-LEVEL VERDICT (04:00, 2026-08-30):** the qwen4_exp fork's MTP path has
a *design* bug: scheduler-side draft ids are never replaced by the
speculator's real proposals anywhere in the pipeline (verified-draft ids are
always the -1 markers), and the speculator won't propose without scheduled
slots. Fixing it = rewriting where the input gather sources draft-slot token
ids (model_state.draft_tokens vs scheduler markers) — a fresh-eyes deep dive,
NOT a one-site accounting patch. Four real bugs were fixed getting here
(ring deadlock sites 17/18, accounting leak site-23, ph underflow, plus the
race), and the corruption is now fully understood mechanistically.

**Shipped state (UPDATED 8/30 ~08:00):** prod 8012 = the vLLM AWQ no-MTP lane
ITSELF, promoted per the operator's call — `~/scripts/serve-flashnext-vllm-8012.sh`,
container qwen38-prod-8012, GPUs 1,2,3, aliases flash-next + qwen3.8-27b,
`--restart unless-stopped`, 48–54 t/s, VISION LIVE (checkpoint ships the ViT —
HF-style inline, unlike GGUF's mmproj; verified with a live image test), and
context raised to 262144 (256K): hybrid-arch KV math (12 full-attn layers ×
2 KV heads × 256 bf16 → 4/14/6 KB/token/rank vs pools holding 588K+ tokens on
the binding rank). The earlier 64K ceiling claim was a pool÷max-len
misestimation; the `--kv-cache-memory` flag is a GLOBAL cap and must NOT be
set from one rank's suggestion. GGUF llama.cpp lane = rollback
(`~/scripts/serve-flashnext-8012.sh`). MTP chain parked at pp3fix21 with
this ledger; swap-verification ladder lives in the prod-lane-swap skill.

## v33 verdict (~02:50): engagement WON, race LOST

pp3fix17 ran the full design. Results:
- **Spec ENGAGED**: SPEC-OUT lines from step 7 on, drafts verified every step.
- **62–64 t/s copy-heavy, 61.7 generic — beats no-MTP's 57.** Free placeholder
  scheduling is what feeds the speculator; speed validates the design.
- **Text doubled again** (`"capital capital! of of!"`): the step-4 phantom
  rollback NEVER FIRED — no `TRACE PHANTOM` line despite correct logic.
- **Root cause of the miss: async-scheduling state race.** The phantom count
  lived on `request.num_phantom_spec_slots`, reset to 0 at the top of the spec
  branch on EVERY schedule() pass. Async scheduling runs schedule(N+1) BEFORE
  update_from_output(N) — so the flag set by step-4's schedule was zeroed by
  step-7's schedule before step-4's output processing could read it. Single-
  slot request state cannot survive a 1-deep scheduling pipeline.

## v34 fix (pp3fix18, race-free)

- New `SchedulerOutput` dataclass field (output.py):
  `phantom_spec_slots: dict[str, int] | None = None`
- schedule() populates a local dict (entry only when placeholders inflate
  num_new but the formula leaves them out of scheduled_spec_decode_tokens)
  and passes it at construction — the OUTPUT object travels with its own
  step, so update_from_output reads `(scheduler_output.phantom_spec_slots
  or {}).get(req_id, 0)`. No request-side flag, no race.
- Site-24 gained a `TRACE S24` print (row count + first rej) so the verdict
  log names whether the last-rank correction fires.
- Success criteria: clean text AND SPEC-OUT events AND ≥60 t/s. Fallback
  stays banked: no-MTP via `~/pp20-launch.sh` (57–59 t/s, clean).

## Artefacts on disk
- Launchers: `~/pp26..37-launch.sh` · watchers: `~/pp30..37-watch.sh` + logs · raw traces: `~/pp30-trace.txt`, `~/pp3[1-7]-watch.log`
- Images: `pp3fix14` (trace prints) → `15` (v31) → `16` (v32, CLEAN TEXT) → `17` (v33, engaged, request-flag race) → `18` (v34, phantom dict race-free) → `19` (v35, ph-underflow crash) → `20` (v36, garbage/no-engagement) → `21` (v37, ids ground truth — PARKED)
- Trace-patched source tree: `~/pp13-src/v1tree/` (extracted from pp3fix12 via docker create+cp; carries sites 23/24 + all TRACE prints — the running source for everything pp3fix14+)
- Two-prompt bench for spec configs: `/tmp/bench_v32.py` pattern (copy-heavy passage vs generic, `usage.completion_tokens` for rate, `grep -c 'TRACE SPEC-OUT'` for engagement)
- Patch scripts: `~/site19.py`, `~/site20.py`, `~/site21.py` (docker cp + exec pattern, ast.parse-gated)
- Equivalence test (branch-coverage lesson): `~/equiv15.py` — its original flaw: only tested non-None `query_start_loc`
