# Serving GLM-5.3-Flash (321B) on a CMP 170HX pair with Project Maya

**Status: PROD lane since 2026-10-20** (replaces the vLLM Flash-Next lane on the same port;
that recipe is kept in this repo as the one-command rollback).

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA CMP 170HX — one 64 GB, one 40 GB (GA100, BAR1-unlocked, PCIe **Gen2 x4**) |
| CPU | Xeon E5-2690 v4 (AVX2) |
| RAM | 185 GB |
| Disk | NVMe SSD, ~120 GB free needed for the model |

The same host also carries 2× RTX 3090 — which are **deliberately NOT used** (see the
4-GPU trap below).

## Model

GLM-5.3-Flash (zai-org, MIT) — 321B-parameter MoE, ~18B active/token, native 1M context.
Served with [Project Maya](https://github.com/mw00/project-maya) v1.1.19 and its
**Maya-M** quant (116 GB GGUF): routed experts IQ2_S/IQ3_S with GPTQ-style error-feedback
rounding against per-expert FP8 statistics, attention/shared experts Q6_K. Keeps 97.9% of
the FP8 model's zero-shot accuracy, 86% same-next-token.

## Install (non-interactive)

```sh
git clone --depth 1 https://github.com/mw00/project-maya.git
cd project-maya
# one GPU visible at first (coexist eval); pin by UUID — CUDA order ≠ nvidia-smi order
CUDA_VISIBLE_DEVICES=GPU-<uuid> ./maya.sh --setup --yes --plain --model Maya-M \
  --gpu 0 --port <port> --host-compiler /usr/bin/g++-13 \
  --no-start --download-model --no-speed-calibration
```

Gotchas hit on this host:

- **Host compiler:** system gcc 15 is too new for CUDA 12.4's nvcc. `--host-compiler
  /usr/bin/g++-13` (gcc-13 installed side by side) fixes it.
- **Port:** pick one that isn't your reverse proxy's.
- **`--no-start`:** first start pins almost all free RAM for the expert tier — launch it
  yourself when the machine is ready (`MAYA_GLM_RAM_HEADROOM_GB` raises the floor left
  for the system; default 6 GB).
- Long builds/downloads must be spawned detached (`subprocess.Popen(..., start_new_session=True)`)
  — anything tied to a terminal session's lifetime can be killed mid-ninja.

## Topology: the 170HX pair ONLY — the 4-GPU trap

Maya splits by **layers** and sizes a per-GPU expert pool from free VRAM. On a mixed
64+40+24+24 GB pool the auto-split assigned each 3090 **one thin layer with ZERO expert
slots** — pure pipeline hops over Gen2 x4 — and packed every routed expert onto the 170HX
pair. Result: decode 20-36 t/s and prefill collapse (~15 t/s) from prompt buffers
borrowing a full card's expert pool.

With `--gpus` pointed at the 170HX pair only:

- Layers [0,28) on the 64 GB card (282 slots/layer), [28,45) on the 40 GB (232 slots/layer)
- **100% of routed experts in VRAM**, boot ~90 s, both 3090s free for other lanes
- The engine build is per-arch: `CMAKE_CUDA_ARCHITECTURES="80;86"` if you ever move it.

## Production config

```jsonc
// maya-maya-m.json (args)
["--glm-pack", "<models>/Maya-M/pack", "--max-context", "1048576",
 "--expert-cache", "auto", "--prefill", "32768", "--kv-streaming", "on",
 "--parallel", "4"]
// port 8012, host 0.0.0.0, api_key set — drop-in for the vLLM lane
```

- **`--parallel 4`, not 8, at 1M context:** every concurrent conversation keeps its own
  ~11 GB latent-KV host copy; 8× would crowd system RAM.
- **`--prefill 32768`:** ~2× prompt speed vs the auto 8192 on a two-GPU split.
- **`--calibrate`** (10-15 min): +31% decode on this host (38 → 50 t/s).

## Measured (this host, warm, temp 0.7)

| Metric | Value |
|---|---|
| Decode, single stream (512 tok) | 38.2 t/s → **48-50 t/s after calibration** |
| Prefill 8K | ~503 t/s |
| Prefill 821K (1M mode) | **594 t/s sustained** (950 shallow → slower as the sparse index grows); TTFT 23 min |
| Needle retrieval @ 821K tokens, 93% depth | **HIT** — exact sentence found and quoted (expert cache 93.2% at depth) |
| Vision (rendered-text card) | PASS, ~7 s |
| Tool calls (`tool_choice: auto`) | PASS, clean JSON args |
| Wrong API key | 401; `/health` unauthenticated 200 |

## Ops notes

- The chat path **ignores the request's `model` field** (llama.cpp-style) — clients
  sending the previous lane's model alias keep working untouched; `/v1/models` lists the
  real name.
- Both OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic
  (`/v1/messages`) APIs are served — agent harnesses can point either way.
- Thinking model: small `max_tokens` gets eaten by reasoning before `content` — probe
  with ≥300.
- **Cold 1M prompts cost ~20-25 min each** (load-once / interrogate-many is the right
  pattern; for repeated cold huge ingests a batched vLLM lane is still the tool).
- Rollback to the vLLM lane: `docker start <vllm-container>` after stopping Maya by the
  exact PID holding the port. Pause any auto-restart watchdog for the vLLM container
  **before** the swap or it will relaunch vLLM into GPUs Maya now owns.
