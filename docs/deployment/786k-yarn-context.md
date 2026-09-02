# 786K Context via YaRN 3.0 — Promotion Notes (2026-09-01)

The proven 256K PP3 recipe was extended to a 786,432-token window
(3.0× the checkpoint's native 262,144) with **three config deltas and zero
kernel/patch changes**. Validated on the live 3-rank pipeline
(24 GB + 64 GB + 40 GB, partition 8/28/12).

## The three deltas

1. `--max-model-len 786432`
2. YaRN override (this vLLM lineage reads the Transformers-v5 key
   `rope_parameters`, not `rope_scaling`):

   ```
   --hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":3.0,"original_max_position_embeddings":262144}}}'
   ```

   `--hf-overrides` deep-merges nested `PretrainedConfig` (`_apply_dict_overrides`
   → `_update_nested`), so the `text_config` form does not clobber sibling keys.
   If the override is silently ignored, the engine fails fast: it rejects
   `max_model_len` greater than the derived 262,144 **before weight load**.

3. `--gpu-memory-utilization 0.85 → 0.92`. The binding rank (the 64 GB card,
   28 layers, ~14 KB KV/token) needs 10.99 GiB for 786K tokens:
   - util 0.90 → 10.6 GiB available → clean boot failure:
     *"estimated maximum model length is 758912"* (a useful error — it names
     the largest window that *does* fit at a given util).
   - util 0.92 → 11.9 GiB → boots; full pool = **846,453 tokens (1.08× window)**.

## Measured results (all on the live lane, single request)

| Probe | Result |
|---|---|
| Needle retrieval, 93% depth | **HIT at 314K, 538K, and 737K tokens** (exact quotes; 2.8× native) |
| Cold prefill throughput | 4,468 t/s @314K → 2,460 @538K → 1,833 @737K |
| Cold TTFT | ~70 s @314K, ~219 s @538K, ~402 s @737K |
| Warm short decode (fresh ctx) | 49.7 t/s (256K-lane baseline same morning: 46.2) |
| Prefill @41K | 5,313 t/s (baseline 5,496; −3%, within run variance) |
| Turn-2 on identical 538K prompt | **16 s vs 219 s cold** — prefix caching works at depth |
| Decode at 538K depth | ~10 t/s (attention/indexer cost; fine for archive query, not interactive chat at full window) |
| Vision / tool calls / auth | all unchanged and passing |

Prefix-cache eviction: the 846K pool holds one 700K prefix **or** one 538K —
two 538K prefixes do not coexist. A "slow hot" reading immediately after a
larger request is eviction, not a cache failure.

## Gotchas re-confirmed

- The first request after boot measures ~4× slow (per-shape CUDA-graph
  capture inside the measurement window). Re-measure warm before publishing.
- Long synthetic prompts: calibrate tokens/record on the actual generator
  (a 2.5× miscalibration silently turned a "300K" probe into a 786,417-token
  prompt — which usefully proved the boundary rejection is a clean 400).
- Promotion of a running experiment: `docker update --restart unless-stopped
  <container>` applies durability without a re-boot.

Rollback: the 256K config is one command (`scripts/serve-vllm-pp3-256k.sh`).

## Follow-up (2026-09-01): MTP un-parked — PP draft-table sync (site-26)

The parked MTP blocker is fixed and validated at 32K bench config:
single-stream **96.6 t/s vs 57-59 no-MTP (1.7x)**, copy-heavy 64.5, clean
text (zero doubled patterns across a 512-token audit), exact-string
preservation, 52% token acceptance (707/1356 over 452 draft steps).

Root cause was PP-specific: the in-forward speculator (`propose()`) runs only
on the last pipeline rank, but every rank's input gather reads its own
`req_states.draft_tokens` - non-last ranks fed never-written zero rows at
draft positions, so verification ran on garbage inputs. The scheduler-side
`[-1]` markers were layout counts only, never the value source.

Fix (`scripts/site26-pp-draft-table-sync.py`, image layer on top of the
existing PP ring patches): a third broadcast on the same symmetric ring -
last rank sends the full `[max_num_reqs, num_spec]` draft table immediately
after `propose()`; non-last ranks receive it in-place at their existing ring
receive. Full-table semantics match the `pp_size`-spaced decode cadence.
No PP+MTP recipe appears to exist in the wild (official recipes validate
MTP on TP only); this is upstream-reportable.
