# Operating the miner (provider-agnostic)

The miner is a **generic, self-contained container**: it bundles a full node + mining
coordinator + CUDA worker, joins the network via **built-in seeds**, and mines
unattended. It is **stateless** (the node re-syncs from seeds on boot) and **runs on
any CUDA GPU on any provider** — nothing here is tied to a specific cloud or GPU model.

## 1. Run it anywhere (the baseline)

On any box with an NVIDIA GPU + the NVIDIA container runtime:

```bash
docker run --gpus all -v lattice-data:/data ghcr.io/adalinxx/lattice-miner-gpu:main
```

That's the whole miner. `libcuda` comes from the host driver; the node syncs and the
GPU mines. This works on **any** provider (vast.ai, RunPod, Lambda, a local rig, …) —
it's just a container.

## 2. Resilient spot mining (recommended): SkyPilot managed jobs

Cheap GPUs are **interruptible** (spot/marketplace hosts drop and get reclaimed), so
the miner should be treated as cattle: replace-and-restart automatically. The
**provider-agnostic** way to do that is [SkyPilot](https://docs.skypilot.co) **managed
jobs** — it provisions the cheapest available matching GPU across ~25 clouds/marketplaces
and **auto-recovers** preemptions by re-provisioning elsewhere and restarting. Because
the miner is stateless, that's all the resilience it needs (no checkpointing).

```bash
pip install "skypilot[aws,gcp,lambda,runpod,vast]"   # enable the providers you use
sky check                                             # verify credentials
sky jobs launch -n lattice-miner --use-spot fleet/miner.sky.yaml
sky jobs queue                # status
sky jobs logs lattice-miner   # live mining logs
sky jobs cancel lattice-miner # stop
```

The task spec is [`fleet/miner.sky.yaml`](miner.sky.yaml) — a set of acceptable GPUs
(cheapest-available wins), `use_spot`, and the miner's entrypoint. Run several by
launching more managed jobs, and spread them across regions/providers to decorrelate
preemptions.

> **Marketplace caveat.** SkyPilot's docker-image-as-task support is first-class on
> AWS/GCP/Azure/Lambda but newer/limited on **Vast.ai and RunPod** (SkyPilot's own docs
> note docker-container tasks aren't fully supported on RunPod). Validate `sky jobs
> launch` against your chosen marketplace before relying on it. If it doesn't work
> there, fall back to option 1 (`docker run`) on that provider, or — on Vast.ai
> specifically — the optional watchdog below.

## 3. Optional: Vast.ai-only watchdog (zero-infra alternative)

If you specifically use **vast.ai** and want a zero-setup alternative to SkyPilot,
[`examples/vast-watchdog.py`](examples/vast-watchdog.py) + the
[`Vast.ai Watchdog`](../.github/workflows/vast-watchdog.yml) GitHub Action reconcile a
desired number of healthy labeled instances (re-renting any that drop) on a cron,
off-fleet. It is **provider-specific** and **inert unless you set the `VAST_API_KEY`
secret** — a convenience, not the generic path. See the file header for config.

## Observability (any of the above)

Ephemeral, NAT'd boxes can't be pull-scraped, so use a **push/heartbeat** model:
- **Dead-man's switch:** have the miner ping a [healthchecks.io](https://healthchecks.io)
  URL while it's hashing; alert when the pings stop (GPU stopped / instance gone). A
  small loop in `deploy/gpu-entrypoint.sh` can do this — ask and it can be added.
- **Chain health:** poll the always-reachable node RPCs of your stable backbone for
  height/peers (a static status page is enough).
- If you outgrow that: agents `remote_write` to a central VictoriaMetrics (push from
  ephemeral miners, pull-scrape the stable backbone in one system). Avoid the Prometheus
  Pushgateway as a general bus (it loses `up` health and never expires dead series).
