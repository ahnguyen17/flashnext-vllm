# Staging Lessons — Flash-Next 111G onto NVMe (2026-08-28)

How the model got from a dying USB enclosure to NVMe, and every trap hit on the way.

## The enemy: WD portable enclosure progressive degradation

- WD Elements/ easystore (WD50NDZW, USB-native, NOT shuckable) degrade under
  SUSTAINED reads: healthy 97-100 MB/s → crawl (3 reads/s at 650 ms each ≈
  3.8 MB/s) after tens of GB. Each USB reset recovers less than the last
  (97 → <11 → stall). Short/ paced I/O stays fine indefinitely.
- **Reset recipe (no replug):** `echo 0 > /sys/bus/usb/devices/<bus-port>/authorized;
  sleep 3-4; echo 1 > .../authorized; sleep 7;` then remount by PARTUUID
  (device names shuffle on replug — never trust sdb).
- **Verdict: never plan a >20G sustained transfer off these.** Pace (256-512MB
  chunks + 2-3s sleep) extends life but tonight it still decayed 33→6 MB/s over an hour.

## The dual-writer race (lost 39G of verified copy)

Two pipelines writing ONE target file = data loss:
- A hash-verified chunk copier wrote the target directly (unlinked-inode ghost after…)
- An hf-download wrapper script's RETRY path did `rm -f "$target"` on mismatch —
  deleting the concurrent writer's file out from under it. The writer kept
  writing 39G into the orphaned inode, invisible at the path.
- **Rules: exactly ONE writer per target file. Download scripts that retry with
  `rm` must never share a target with anything. Space-gate checks must account
  for ALL concurrent consumers of the filesystem.**

## Deleted-inode rescue (the technique that ALMOST saved it)

Process still holds an unlinked file open → data alive via `/proc/PID/fd/N`:
- `ln /proc/PID/fd/N /target` → **EXDEV "Invalid cross-device link" on this
  kernel** — procfs magic-symlink link() is refused.
- **`cp /proc/PID/fd/N /target` works** (reads through the magic symlink).
  Cost: full-copy time (NVMe→NVMe ~1 min for 46G). Rescue BEFORE killing the
  process — killing first frees the inode and the data is gone. (Lost the 39G
  this way: link failed, kill was in the same command block anyway.)

## Kill discipline (recurred 3× tonight despite existing rules)

- `pkill/pgrep -f '<pattern>'` self-matches the running shell's own command
  line (exit -15/-9, mystery interrupts). Kill by exact PID, or bracket the
  pattern (`vll[m]`), or match by PORT (`ss -tlnp | grep <port>` → owning PID).
- Killing a wrapper (`sudo bash -c ...`/`bash script.sh`) does NOT kill the
  child (hf/python keeps downloading into directories you may have deleted).
  Kill the CHILD first, verify with `pgrep -af` — expect one orphan.

## What worked

- hf download (HF_HUB_DISABLE_XET=1) @ ~10 MB/s per-IP: reliable, resumable,
  hash-verifiable — beat the sick enclosure. Parallel range streams split the
  same per-IP cap (measured), hf_transfer deprecated in hub 1.27, Xet hangs.
- Chunked copier pattern (512MB chunk → fsync → read-back verify → state file
  with offset) survives kills/resets; combine with the USB authorized-toggle
  watchdog for self-healing paced copies.
- Opening files `w+b` not `wb` when the code read-backs its own writes
  (`io.UnsupportedOperation: read` — its str() is just "read").
