# Hybrid-arch context math + multimodal verification (8/30, the 256K push)

Class-level techniques from raising the FlashNext AWQ lane from 32K → 256K
context, and the vision-claim reversal. (Ops procedure lives in prod-lane-swap.)

## Hybrid GDN context budget — the technique

1. Read geometry from the checkpoint's `text_config`: `num_hidden_layers`,
   `full_attention_interval`, `num_key_value_heads`, `head_dim`, KV dtype.
2. **Only full-attention layers carry per-token KV** (interval 4 → 1 in 4
   layers; the GDN/linear layers hold a CONSTANT per-sequence state — no
   per-token cost). Count attn layers PER PIPELINE RANK from the partition:
   8/28/12 with attn at every 4th layer (3,7,...,47) → 2/7/3 per rank.
3. Per-rank bytes/token = attn_layers_on_rank × 2 × kv_heads × head_dim ×
   dtype_bytes. FlashNext: 2×2×256×2 = 4 KB (PP0), 14 KB (PP1), 6 KB (PP2).
4. Read each rank's granted pool from its `Available KV cache memory:` boot
   line. Capacity per rank = pool ÷ per-rank-bpt; binding rank = min.
   FlashNext at util 0.85: 603K / 588K / 2.04M tokens → 256K fits, 2.2× margin.

## Two estimation traps (both bit in one session)

- **pool ÷ max_model_len ≠ bytes/token.** vLLM sizes the pool from FREE
  memory at the util target, independent of max_model_len (only constraint:
  pool ≥ max_model_len × bpt or boot refuses). Dividing the two assumes
  sized-to-fit and understates capacity by the entire slack — a 10× error:
  "96K doesn't fit" was claimed; 256K fit with margin.
- **`--kv-cache-memory` is a GLOBAL cap.** Every PP worker echoes its own
  "Replace gpu_memory_utilization config with --kv-cache-memory=X (fully
  utilize)" suggestion, but any single X caps ALL ranks. Taking the thin
  rank's 5 GiB strangles a fat rank (16 GiB free) below target. For max
  context on PP: omit the flag entirely.

## Multimodal capability — verify the checkpoint, not the log line

- `Supported tasks: ['generate']` in vLLM boot logs is the TASK registry —
  VL models still serve task 'generate'. It says nothing about vision.
- HF/AWQ checkpoints ship the vision tower INSIDE the weights (unlike GGUF's
  separate mmproj file). "No mmproj file" ≠ "no vision".
- Checklist: `architectures` ends `ForConditionalGeneration`; `vision_config`
  present; `image_token_id`/`vision_*_token_id` keys; `model.visual.*`
  tensors in `model.safetensors.index.json`; fork registers the arch under
  `_MULTIMODAL_MODELS` (registry.py).
- Decisive test (~10s): base64 a small PNG, send through
  /v1/chat/completions, ask "what color?". Reversed a wrong "vision loss"
  claim the moment the user pushed back on it.
