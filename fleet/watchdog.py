#!/usr/bin/env python3
"""Fleet watchdog — keep N labeled vast.ai GPU miners alive; re-rent any that drop.

Ephemeral cloud-rental GPUs are cattle, not pets: hosts go offline and instances get
reclaimed. This runs periodically (GitHub Actions cron) and RECONCILES the desired
number of healthy instances carrying a label:

  * counts running/loading instances (loading = still booting, don't double-rent),
  * destroys clearly-dead ones (offline/exited) so they stop billing,
  * rents the cheapest matching replacement up to the desired count,
  * pings a dead-man's-switch so the operator is alerted if the WATCHDOG itself stops.

Stateless + idempotent: safe to run every N minutes. It only ever acts on instances
carrying FLEET_LABEL, so it never touches unrelated rentals.

The miner container is self-contained and stateless (its node re-syncs from built-in
seeds on boot), so "recovery" is simply "rent an equivalent box and run the image" —
no checkpoint/restore needed.

Config (env; all optional except the API key, set on the vastai CLI beforehand):
  FLEET_LABEL          instance label to manage           (default nexus-3060-prod)
  FLEET_DESIRED        how many healthy instances to hold  (default 1)
  FLEET_IMAGE          miner image                         (default ghcr.io/adalinxx/lattice-miner-gpu:main)
  FLEET_ONSTART        onstart command                     (default /usr/local/bin/gpu-entrypoint)
  FLEET_DISK           disk GB                             (default 32)
  FLEET_QUERY          vastai offer search query           (default cheapest reliable RTX 3060, CUDA>=12.6)
  FLEET_MAX_DPH        do not rent above this $/hr          (default 0.10)
  FLEET_HEARTBEAT_URL  dead-man's-switch ping (e.g. healthchecks.io)  (default none)
  VASTAI_BIN           vastai CLI path                     (default vastai)
"""
import json
import os
import subprocess
import sys
import urllib.request

VAST = os.environ.get("VASTAI_BIN", "vastai")
LABEL = os.environ.get("FLEET_LABEL", "nexus-3060-prod")
DESIRED = int(os.environ.get("FLEET_DESIRED", "1"))
IMAGE = os.environ.get("FLEET_IMAGE", "ghcr.io/adalinxx/lattice-miner-gpu:main")
ONSTART = os.environ.get("FLEET_ONSTART", "/usr/local/bin/gpu-entrypoint")
DISK = os.environ.get("FLEET_DISK", "32")
QUERY = os.environ.get(
    "FLEET_QUERY",
    "rentable=true verified=true num_gpus=1 gpu_name=RTX_3060 cuda_vers>=12.6 "
    f"disk_space>={DISK} reliability>0.98",
)
MAX_DPH = float(os.environ.get("FLEET_MAX_DPH", "0.10"))
HEARTBEAT_URL = os.environ.get("FLEET_HEARTBEAT_URL", "").strip()

RUNNING = {"running"}
BOOTING = {"loading", "created", "scheduling"}  # counts toward desired; don't destroy


def vast(args):
    """Run a vastai CLI command, returning (stdout, ok)."""
    r = subprocess.run([VAST, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! vastai {' '.join(args[:2])} exit={r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout, r.returncode == 0


def fleet_instances():
    out, ok = vast(["show", "instances", "--raw"])
    if not ok:
        return None  # None = API failure: do NOTHING this cycle (fail safe, don't churn)
    try:
        allinst = json.loads(out)
    except json.JSONDecodeError:
        print("  ! could not parse instances JSON")
        return None
    return [i for i in allinst if i.get("label") == LABEL]


def status_of(inst):
    return (inst.get("actual_status") or inst.get("cur_state") or "unknown").lower()


def rent_replacement():
    out, ok = vast(["search", "offers", QUERY, "-o", "dph+", "--raw"])
    if not ok:
        return False
    try:
        offers = [o for o in json.loads(out) if o.get("dph_total", 1e9) <= MAX_DPH]
    except json.JSONDecodeError:
        print("  ! could not parse offers JSON")
        return False
    if not offers:
        print(f"  no offer matching query under ${MAX_DPH:.4f}/hr")
        return False
    o = offers[0]
    print(f"  renting offer {o['id']} ({o.get('gpu_name')}) @ ${o['dph_total']:.4f}/hr {o.get('geolocation','')}")
    out, ok = vast([
        "create", "instance", str(o["id"]),
        "--image", IMAGE, "--disk", DISK,
        "--onstart-cmd", ONSTART, "--label", LABEL, "--raw",
    ])
    success = ok and '"success":true' in out.replace(" ", "")
    print("  -> rented" if success else f"  -> rent failed: {out.strip()[:200]}")
    return success


def ping_heartbeat():
    if not HEARTBEAT_URL:
        return
    try:
        urllib.request.urlopen(HEARTBEAT_URL, timeout=10)
        print("  heartbeat pinged")
    except Exception as e:  # noqa: BLE001 - heartbeat is best-effort
        print(f"  ! heartbeat ping failed: {e}")


def main():
    inst = fleet_instances()
    if inst is None:
        print("vast API unavailable — skipping this cycle (no changes)")
        # Do NOT ping heartbeat on API failure: a stuck vast API should surface as a
        # missed heartbeat rather than silently look healthy.
        return 0

    running = [i for i in inst if status_of(i) in RUNNING]
    booting = [i for i in inst if status_of(i) in BOOTING]
    dead = [i for i in inst if status_of(i) not in RUNNING | BOOTING]
    print(f"label={LABEL}: running={len(running)} booting={len(booting)} "
          f"dead={len(dead)} desired={DESIRED}")

    for i in dead:
        print(f"  destroying dead instance {i['id']} (status={status_of(i)})")
        vast(["destroy", "instance", str(i["id"])])

    # booting instances count toward desired so a slow ~7min image pull doesn't cause
    # over-provisioning on the next cycle.
    have = len(running) + len(booting)
    need = max(0, DESIRED - have)
    if need:
        print(f"  need {need} more to reach desired={DESIRED}")
        for _ in range(need):
            rent_replacement()
    else:
        print("  fleet at desired capacity")

    ping_heartbeat()
    return 0


if __name__ == "__main__":
    sys.exit(main())
