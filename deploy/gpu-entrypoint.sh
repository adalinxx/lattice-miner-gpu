#!/bin/sh
# Self-starting merged miner (current lattice-node architecture): one GPU box
# runs a Nexus node plus an adopted child chain under `lattice up --foreground`
# (the supervisor restarts either process), and drives mining through the
# reference mine-supervisor once the parent is synced. ONE CUDA solution
# advances Nexus AND the child (merged mining): the child process prepares
# candidates and hands them to its parent over the authenticated hierarchy
# link, and the parent's mining template carries them.
#
# The child joins PERMISSIONLESSLY: a topology entry with no genesis seed boots
# the child process in `awaitingGenesis`, and it re-derives the genesis through
# its authenticated parent link (never from "a node that tracks it"). Child
# peers for catch-up sync come from the parent rendezvous (getChildPeers).
#
# libcuda is injected from the host driver by the NVIDIA container runtime
# (vast.ai / RunPod / Lambda --gpus all); NVRTC compiles the kernel at run time.
#
# Config (env):
#   NEXUS_PEERS   space-separated publicKey@host:port overlay peers
#                 (default: the mainnet backbones + the testnet follower)
#   CHILD_PATH    absolute child path to adopt (default Nexus/testnet);
#                 set empty to mine Nexus only
#   WORKERS       GPU worker processes (default 1; one drives the whole GPU)
#   BATCH_SIZE    nonce span per worker per iteration (default 2_000_000_000)
#   REWARD_BATCH  pre-signed `lattice-rewards emit-batch` file; mining waits
#                 for it to exist (default /data/reward-batch.jsonl) — ship it
#                 with scp after boot. The reward KEY never touches this host.
#   MINER_BACKEND cuda|opencl|cpu for the worker shim (default cuda)
set -eu

ROOT="${DATA_DIR:-/data}"
CHILD_PATH="${CHILD_PATH-Nexus/testnet}"
NEXUS_PEERS="${NEXUS_PEERS:-139b8f3639e7c515417c63bd3a652a5c6fd4a1a2d0baed8e33ea63047995fe64@lattice-mainnet-iad.fly.dev:4001 35edf67bfe3d612aeb1f0e25da9d3f0ced44dbf79d34f00c548cf9005be6eb7d@lattice-mainnet-ams.fly.dev:4001 9cace839489acb30385a9f20025cb9d6365283c81dce14cadab26507065acd4e@lattice-mainnet-sjc.fly.dev:4001 57f80deb3b00da1b14b630638a4d0307be98126ec1d550476e4889087bb22d0f@lattice-mainnet-testnet.fly.dev:4001}"
WORKERS="${WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2000000000}"
REWARD_BATCH="${REWARD_BATCH:-$ROOT/reward-batch.jsonl}"
NEXUS_RPC="http://127.0.0.1:8080"

mkdir -p "$ROOT"

peers_json=""
for peer in $NEXUS_PEERS; do
    peers_json="$peers_json\"$peer\","
done
peers_json="${peers_json%,}"

child_json=""
if [ -n "$CHILD_PATH" ]; then
    child_json=",
    \"$CHILD_PATH\": {
      \"listen\": 4101,
      \"fact\": 4102,
      \"rpc\": 8180
    }"
fi

# Declarative topology, rewritten every boot; identities and chain state
# persist under $ROOT. The child entry has no genesis seed on purpose (adopt).
cat > "$ROOT/lattice.json" <<EOF
{
  "chains": {
    "Nexus": {
      "listen": 4001,
      "fact": 4002,
      "rpc": 8080,
      "peers": [$peers_json]
    }$child_json
  }
}
EOF

# Bring-up runs beside the foreground supervisor: wait for the parent to
# catch up to the network, then for the reward batch, then hand over to the
# reference mining supervisor. The child catches up in the
# background and starts contributing candidates when ready; it never blocks
# Nexus mining.
(
    # Synced means CAUGHT UP TO THE NETWORK, not locally stable: a just-booted
    # node sits at height 0 with connected peers for longer than any local
    # stability window while cold sync spins up, and mining then extends a
    # private fork from genesis. Gate on the height a public reference node
    # reports (any backbone; tried in order) — never on local quiescence.
    echo "mining bring-up: waiting for Nexus to catch up to the network…"
    sleep 10
    while :; do
        network_height=""
        for ref in ${REFERENCE_RPCS:-https://lattice-mainnet-iad.fly.dev https://lattice-mainnet-ams.fly.dev https://lattice-mainnet-sjc.fly.dev}; do
            network_height="$(curl -fsS --max-time 8 "$ref/api/chain/info" 2>/dev/null | jq -r '.height // empty')" && [ -n "$network_height" ] && break
        done
        [ -n "$network_height" ] || { sleep 10; continue; }
        height="$(curl -fsS "$NEXUS_RPC/api/chain/info" 2>/dev/null | jq -r '.height // -1')"
        peers="$(curl -fsS "$NEXUS_RPC/api/peers" 2>/dev/null | jq -r '.count // 0' 2>/dev/null)"
        if [ "${peers:-0}" -ge 1 ] && [ "${height:--1}" -ge "$network_height" ]; then
            break
        fi
        echo "mining bring-up: local $height / network $network_height…"
        sleep 10
    done
    echo "mining bring-up: Nexus caught up (height $height, network $network_height)."

    while [ ! -s "$REWARD_BATCH" ]; do
        echo "mining bring-up: waiting for reward batch at $REWARD_BATCH (scp it in)…"
        sleep 15
    done
    echo "mining bring-up: starting the mining supervisor."
    NODE_URL="$NEXUS_RPC" \
    COORDINATOR=/usr/local/bin/lattice-mining-coordinator \
    WORKER=/usr/local/bin/lattice-cuda-worker \
    WORKERS="$WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    REWARD_BATCH="$REWARD_BATCH" \
    CURSOR_FILE="$ROOT/reward-cursor" \
    LOG_FILE="$ROOT/mining.log" \
    exec python3 /usr/local/bin/mine-supervisor.py
) &

exec lattice up --root "$ROOT" --foreground
