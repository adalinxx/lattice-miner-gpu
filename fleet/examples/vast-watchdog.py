#!/usr/bin/env python3
"""Vast.ai watchdog (OPTIONAL, PROVIDER-SPECIFIC example).

The generic/recommended resilience layer is provider-agnostic SkyPilot managed jobs
(see ../miner.sky.yaml + ../README.md). This is a self-contained convenience for
operators who specifically use vast.ai and want a zero-infra alternative to SkyPilot.

Keep N labeled vast.ai GPU miners alive; re-rent any that drop.

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
  FLEET_LABEL          instance label to manage           (default nexus-miner)
  FLEET_DESIRED        how many healthy instances to hold  (default 1)
  FLEET_IMAGE          miner image                         (default ghcr.io/adalinxx/lattice-miner-gpu:main)
  FLEET_ONSTART        onstart command                     (default /usr/local/bin/gpu-entrypoint)
  FLEET_DISK           disk GB                             (default 32)
  FLEET_QUERY          vastai offer search query           (default: cheapest reliable single
                                                            CUDA>=12.6 GPU of ANY model)
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
LABEL = os.environ.get("FLEET_LABEL", "nexus-miner")
DESIRED = int(os.environ.get("FLEET_DESIRED", "1"))
IMAGE = os.environ.get("FLEET_IMAGE", "ghcr.io/adalinxx/lattice-miner-gpu:main")
ONSTART = os.environ.get("FLEET_ONSTART", "/usr/local/bin/gpu-entrypoint")
DISK = os.environ.get("FLEET_DISK", "32")
# GPU-model-AGNOSTIC by design: the miner runs on any CUDA GPU, so rent the cheapest
# suitable card of ANY model. cuda_vers>=12.6 = the host driver can run the image's
# CUDA 12.6 kernels; a modest inet floor avoids a box that takes forever to pull the
# image. rent_replacement() sorts by price and picks the cheapest. Narrow it (e.g. a
# specific gpu_name, more VRAM) via the FLEET_QUERY env var if you ever want to.
QUERY = os.environ.get(
    "FLEET_QUERY",
    "rentable=true verified=true num_gpus=1 cuda_vers>=12.6 "
    f"disk_space>={DISK} reliability>0.98 inet_down>=100",
)
MAX_DPH = float(os.environ.get("FLEET_MAX_DPH", "0.10"))
HEARTBEAT_URL = os.environ.get("FLEET_HEARTBEAT_URL", "").strip()

RUNNING = {"running"}
# Destruction is an ALLOWLIST, not a denylist: only tear down instances in a KNOWN
# terminal state. Everything else — running, still-booting, or a status string we don't
# recognize — is left alone and counts toward capacity. This is deliberate: classifying
# "anything not explicitly healthy" as dead would destroy a healthy/booting box (or one
# reporting a transient/unknown status like a Docker restart or an empty status right
# after create) and rent a paid replacement — the exact runaway spend this guards against.
# A genuinely-missing box surfaces as a heartbeat gap, not as churn. Extend TERMINAL only
# with statuses confirmed terminal against `vastai show instances --raw`.
TERMINAL = {"exited", "offline", "inactive", "error"}


def vast(args):
    """Run a vastai CLI command, returning (stdout, ok). Fails safe on any launch error."""
    try:
        r = subprocess.run([VAST, *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ! could not run vastai {' '.join(args[:2])}: {e}")
        return "", False
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


def rent_replacement(exclude=()):
    """Rent the cheapest matching offer not in `exclude`. Returns the offer id on
    success, else None. `exclude` lets a multi-rent cycle skip offers it already took,
    so renting need>1 doesn't retarget the same single cheapest offer every time."""
    out, ok = vast(["search", "offers", QUERY, "-o", "dph+", "--raw"])
    if not ok:
        return None
    try:
        offers = [o for o in json.loads(out)
                  if o.get("dph_total", 1e9) <= MAX_DPH and o.get("id") not in exclude]
    except json.JSONDecodeError:
        print("  ! could not parse offers JSON")
        return None
    if not offers:
        print(f"  no offer matching query under ${MAX_DPH:.4f}/hr")
        return None
    o = offers[0]
    print(f"  renting offer {o['id']} ({o.get('gpu_name')}) @ ${o['dph_total']:.4f}/hr {o.get('geolocation','')}")
    out, ok = vast([
        "create", "instance", str(o["id"]),
        "--image", IMAGE, "--disk", DISK,
        "--onstart-cmd", ONSTART, "--label", LABEL, "--raw",
    ])
    success = ok and '"success":true' in out.replace(" ", "")
    print("  -> rented" if success else f"  -> rent failed: {out.strip()[:200]}")
    return o["id"] if success else None


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
    dead = [i for i in inst if status_of(i) in TERMINAL]
    # kept = running + still-booting + any UNKNOWN status: everything we don't destroy,
    # and everything that counts toward capacity (so a slow ~7min image pull, reported
    # as a booting/unknown status, doesn't trigger a duplicate rent next cycle).
    kept = [i for i in inst if status_of(i) not in TERMINAL]
    print(f"label={LABEL}: running={len(running)} kept(incl. booting)={len(kept)} "
          f"dead={len(dead)} desired={DESIRED}")

    for i in dead:
        print(f"  destroying dead instance {i['id']} (status={status_of(i)})")
        vast(["destroy", "instance", str(i["id"])])

    have = len(kept)
    need = max(0, DESIRED - have)
    rented = 0
    if need:
        print(f"  need {need} more to reach desired={DESIRED}")
        tried = set()
        for _ in range(need):
            oid = rent_replacement(exclude=tried)
            if oid is None:
                break  # no offer available / create failed — retry next cycle
            tried.add(oid)
            rented += 1
    else:
        print("  fleet at desired capacity")

    # Ping the dead-man's switch only if capacity is met or we made progress toward it.
    # If we needed boxes and rented NONE (no affordable offer / vast rejecting creates),
    # skip the ping so a fleet stuck below desired surfaces as a missed heartbeat rather
    # than silently looking healthy.
    if need == 0 or rented > 0:
        ping_heartbeat()
    else:
        print("  needed capacity but rented none — skipping heartbeat so the shortfall alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
