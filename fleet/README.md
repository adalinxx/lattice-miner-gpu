# Fleet watchdog

Keeps a small fleet of **unreliable / ephemeral** vast.ai GPU miners alive. Ephemeral
cloud GPUs are cattle, not pets — hosts go offline and instances get reclaimed — so the
watchdog **reconciles** the desired number of healthy labeled instances every ~10 min:
counts running/booting instances, destroys dead ones (so they stop billing), and re-rents
the cheapest matching replacement. The miner container is stateless (its node re-syncs
from built-in seeds on boot), so recovery is just "rent an equivalent box and run it."

It runs as a **GitHub Actions scheduled workflow** (`.github/workflows/watchdog.yml`) —
off-fleet, free, secrets managed, version-controlled. It can't die with the GPUs it
supervises.

## One-time setup

1. **Secret — `VAST_API_KEY`** (repo → Settings → Secrets and variables → Actions):
   your vast.ai API key. Without it the workflow is a no-op. Never commit this.
2. **Optional secret — `FLEET_HEARTBEAT_URL`**: a [healthchecks.io](https://healthchecks.io)
   (free) ping URL. The watchdog pings it each successful run, so you get **alerted if the
   watchdog itself stops running** (a dead-man's switch on your resilience layer).
3. **Optional variables** (repo → Variables) to tune without editing code:
   `FLEET_LABEL` (default `nexus-3060-prod`), `FLEET_DESIRED` (default `1`),
   `FLEET_IMAGE`, `FLEET_MAX_DPH` (default `0.10` $/hr), `FLEET_QUERY`, `FLEET_DISK`.

Run it manually any time via **Actions → Fleet Watchdog → Run workflow** (optionally
overriding the desired count).

## What it does / doesn't touch

- Acts **only** on instances carrying `FLEET_LABEL` — never your other rentals.
- Fails safe: if the vast API is unreachable, it makes **no changes** that cycle (and
  skips the heartbeat, so a stuck API surfaces as a missed heartbeat rather than
  silent "healthy").
- Counts *booting* instances toward desired, so a slow ~7 min image pull doesn't cause
  double-renting.

## Complementary: per-miner heartbeat (recommended)

The watchdog detects a **dead host** (instance gone/offline). It does **not** detect a
box that's up but whose miner is *stuck* (rare, but possible). For that, have the miner
itself ping a heartbeat while it's hashing — a short addition to `deploy/gpu-entrypoint.sh`
(loop pinging a per-instance healthchecks URL when the coordinator is running + the GPU
is drawing power). Ask and it can be added.

## Scaling up later

For multi-provider cheapest-GPU auto-selection or a larger fleet, graduate to
[SkyPilot managed jobs](https://docs.skypilot.co/en/stable/examples/managed-jobs.html)
(`sky jobs launch`), which provisions + auto-recovers across ~25 backends including
vast.ai and RunPod. This watchdog is the minimal, zero-infra tier for a handful of boxes.
