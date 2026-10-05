#!/bin/sh
# Self-starting merged miner (current lattice-node architecture): ONE
# lattice-node process hosts Nexus plus the configured child levels under
# `lattice up --foreground`, and mining runs through the reference
# mine-supervisor once Nexus is synced. ONE CUDA solution advances Nexus AND
# the hosted children (merged mining): the template from the Nexus RPC
# carries every hosted level.
#
# Rewards: each block names its own reward recipient (`rewardRecipient`,
# covered by the proof of work). The host holds no key and no pre-signed
# reward file; a recipient is only an address.
#
# libcuda is injected from the host driver by the NVIDIA container runtime
# (vast.ai / RunPod / Lambda --gpus all); NVRTC compiles the kernel at run time.
#
# Config (env):
#   RECIPIENTS     REQUIRED. comma-separated <chain path>=<address> entries,
#                  e.g. "Nexus=bafy...,Nexus/testnet=bafy..."; a hosted chain
#                  with no entry burns its reward and fees
#   NEXUS_PEERS    space-separated publicKey@host:port overlay peers
#                  (default: the mainnet backbones + the testnet follower)
#   HOSTED_CHAINS  space-separated Nexus-rooted child paths to host and
#                  merge-mine, parent first (default Nexus/testnet); set empty
#                  to mine Nexus only
#   WORKERS        GPU worker processes (default 1; one drives the whole GPU)
#   BATCH_SIZE     nonce span per coordinator round (default 2_000_000_000)
#   MINER_BACKEND  cuda|opencl|cpu for the worker shim (default cuda)
set -eu

ROOT="${DATA_DIR:-/data}"
RECIPIENTS="${RECIPIENTS:?set RECIPIENTS=<chain path>=<address>[,...] (e.g. Nexus=bafy...); without it every reward burns}"
HOSTED_CHAINS="${HOSTED_CHAINS-Nexus/testnet}"
NEXUS_PEERS="${NEXUS_PEERS:-139b8f3639e7c515417c63bd3a652a5c6fd4a1a2d0baed8e33ea63047995fe64@lattice-mainnet-iad.fly.dev:4001 35edf67bfe3d612aeb1f0e25da9d3f0ced44dbf79d34f00c548cf9005be6eb7d@lattice-mainnet-ams.fly.dev:4001 9cace839489acb30385a9f20025cb9d6365283c81dce14cadab26507065acd4e@lattice-mainnet-sjc.fly.dev:4001 57f80deb3b00da1b14b630638a4d0307be98126ec1d550476e4889087bb22d0f@lattice-mainnet-testnet.fly.dev:4001}"
WORKERS="${WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2000000000}"
NEXUS_RPC="http://127.0.0.1:8080"
# The node's loopback operator port requires the cookie it writes at every
# start (`lattice up --root $ROOT` gives it $ROOT/chains/Nexus); re-read on
# every use, since a node restart rotates it.
COOKIE_FILE="$ROOT/chains/Nexus/.cookie"
rpc() { curl -fsS --user "$(cat "$COOKIE_FILE" 2>/dev/null)" "$NEXUS_RPC$1"; }

mkdir -p "$ROOT"

peers_json=""
for peer in $NEXUS_PEERS; do
    peers_json="$peers_json\"$peer\","
done
peers_json="${peers_json%,}"

chains_json=""
for chain in $HOSTED_CHAINS; do
    chains_json="$chains_json\"$chain\","
done
chains_json="${chains_json%,}"

# Declarative topology, rewritten every boot; identity and chain state
# persist under $ROOT.
cat > "$ROOT/lattice.json" <<EOF
{
  "hostedChains": [$chains_json],
  "listen": 4001,
  "peers": [$peers_json],
  "rpc": 8080
}
EOF

# Bring-up runs beside the foreground supervisor: wait for the parent to
# catch up to the network, then hand over to the reference mining supervisor.
# Hosted children catch up in the background; they never block Nexus mining.
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
        height="$(rpc /api/chain/info 2>/dev/null | jq -r '.height // -1')"
        peers="$(rpc /api/peers 2>/dev/null | jq -r '.count // 0' 2>/dev/null)"
        if [ "${peers:-0}" -ge 1 ] && [ "${height:--1}" -ge "$network_height" ]; then
            break
        fi
        echo "mining bring-up: local $height / network $network_height…"
        sleep 10
    done
    echo "mining bring-up: Nexus caught up (height $height, network $network_height)."

    echo "mining bring-up: starting the mining supervisor."
    NODE_URL="$NEXUS_RPC" \
    COOKIE_FILE="$COOKIE_FILE" \
    COORDINATOR=/usr/local/bin/lattice-mining-coordinator \
    WORKER=/usr/local/bin/lattice-cuda-worker \
    WORKERS="$WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    RECIPIENTS="$RECIPIENTS" \
    LOG_FILE="$ROOT/mining.log" \
    exec python3 /usr/local/bin/mine-supervisor.py
) &

exec lattice up --root "$ROOT" --foreground
