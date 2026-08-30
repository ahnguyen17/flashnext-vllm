# Prod-Lane Swap Playbook (engine A → engine B, same port, zero harness changes)

Generalized from the 8/28 swap: Qwen3.8-27B (vLLM) → Qwen3.8-Flash-Next 176B
(llama.cpp) on `:8012`, with downstream consumers workers, reachy-brain, ops scripts, a
caddy public funnel, and two tailscale paths all live against the port.
Rig-agnostic — the lane specifics live in the parent SKILL.md.

## 1. Enumerate consumers BEFORE touching anything

- Ripgrep the port across the home tree — three URL forms:
  `localhost:PORT | 127\.0\.0\.1:PORT | <tailscale-ip>:PORT`
- Places that hide consumers: `~/scripts/*boot-test*.sh` (auth assertions!),
  worker code (`LLM_URL`-style env defaults), probes, `crontab -l`,
  `~/.hermes/cron/jobs.json` (check `enabled` — disabled jobs still matter as
  future combatants), systemd timers.
- For each hit record: URL form, auth header, model string sent, endpoints
  used (`/v1/chat/completions`, `/v1/models`, `/health`).

## 2. Replicate the FULL contract on the new engine

The port is the least of it. In observed order of "would have broken things":

1. **Auth key AND rejection behavior.** Ops boot-tests often assert that a
   WRONG key gets 401 (ours: `W=$(curl -H "Authorization: Bearer wrong" …)`).
   vLLM `--api-key K` ↔ llama-server `--api-key K` both enforce; serving
   keyless "because clients already send the header" silently breaks the
   rejection assert and opens the port.
2. **`/health` stays unauthenticated** — watchdogs curl it bare; 401 there
   reads as "server down" (curl succeeds on any HTTP response, so a plain
   `if ! curl` watchdog survives, but `-sf` ones don't).
3. **Model string leniency.** llama-server accepts ANY `model` field when one
   model is loaded — legacy vLLM-era names (even different casing) hit the new
   lane unchanged. `--alias` only changes the `/v1/models` listing (picked up
   by informational tooling like gpu-mode scripts — set it to something true).
4. **Vision path WITH auth** if the old lane served images — test the exact
   client payload shape (base64 `image_url` + Authorization together).

Verification battery (one python script, run all — see `scripts/verify_dropin.py`):
health ±key → 200/200; models wrong-key → 401; models right-key → 200;
chat with a LEGACY model string → answers; vision+auth → reads a rendered-text
card. Test cards beat arbitrary images: "read the name/MRN" is unambiguous
ground truth (a model accurately describing a boring photo looks like failure).

## 3. Audit startup combatants

- `crontab -l` `@reboot` lines: any script that (a) kills processes by GPU
  memory size, (b) launches another server needing the same GPUs/port.
- Disable with a DATED comment (`# DISABLED 8/28 <reason>:`) — never delete;
  the old scripts ARE the rollback levers. Back up: `crontab -l > ~/scripts/crontab.bak-<date>`.
- Remember kill-by-size restarters are LETHAL to the replacement: ours kills
  any >30GB GPU process — document "never run while replacement is up".

## 4. Network exposure (tailscale stack)

- **serve (tailnet-only, zero downtime):** `tailscale serve --bg --https=<p> http://127.0.0.1:<port>`
  — ALWAYS `--bg` (foreground mode dies under tool timeouts, exit 124).
  Survives backend restarts (points at the port, not the pid). Repoint:
  `tailscale serve --https=<p> off` then re-add with the new target.
- **direct tailnet:** server `--host 0.0.0.0` → `http://<ts-ip>:<port>/v1`
  (plain HTTP but inside WireGuard — encrypted in transit).
- **funnel (public):** if caddy fronts the port with a public-key→internal-key
  `header_up Authorization` swap, the PUBLIC side needs zero changes on an
  engine swap — the swap pattern decouples outside harnesses from the lane's
  own auth. Never aim `tailscale funnel` directly at an unkeyed endpoint.

## 5. Restart mechanics

- The background-terminal session pid is often a wrapper (`chmod … && script`
  spawns a child that later `exec`s). Real listener pid: `ss -tlnp`. Kill
  child → confirm port free → relaunch → poll `/health` (~2 min for ~100GB
  warm page cache).
- Bench tooling must carry the lane auth once `--api-key` is on (`FN_KEY`-style
  env in bench scripts; `/props` and `/completion` both need it).

## 6. Post-swap config A/Bs (e.g. ubatch)

- Nonce'd A/B bench (unique per-case prefix defeats llama.cpp slot prefix
  reuse, which otherwise flatters short-prefill numbers).
- ubatch scaling law: gains ∝ 1/(existing GPU saturation). CPU-bound rigs +2×,
  saturated rigs +20% shallow / 0 deep (indexer-bound), decode never moves.
- Keep only on strict win; log BOTH runs + config to the baseline repo and
  commit — the repo, not the skill, is the frozen yardstick.
