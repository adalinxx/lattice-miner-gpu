#!/usr/bin/env bash
# Self-contained GPU miner entrypoint.
#
# The CUDA worker (lattice-miner-gpu) does PoW only; it cannot reach the network on
# its own. So this image bundles a Lattice node and the mining coordinator:
#
#   1. lattice-node joins via bootstrap seeds, then syncs the chain source-agnostically
#      (any peer, PoW + content-addressed), exposing RPC on localhost.
#   2. lattice-mining-coordinator pulls templates from that local node and drives the
#      CUDA worker for the actual proof-of-work, then gossips sealed blocks back.
#
# Bootstrap seeds are built into the node binary (BootstrapPeers), and the default
# --min-peer-key-bits (16) already matches the live network, so a bare node joins and
# syncs source-agnostically on its own — no --peer or key-bits flag to configure.
# (Verified: a fresh bare node reaches the mainnet's mineable tip in ~2 min.)
#
# libcuda is injected from the host driver by the NVIDIA container runtime (vast.ai /
# RunPod / Lambda --gpus all); NVRTC compiles the kernel at run time.
#
# MERGED (MULTI-CHAIN) MINING — set CHILD_CHAINS to also advance child chain(s) of
# Nexus on THIS box (single-box model). The node runs with --supervise-children, so a
# deploy auto-spawns and supervises one child node process (deterministic loopback
# ports); the coordinator then merge-mines Nexus + each child in one PoW search (easiest
# target wins, sealed blocks anchor the child via ChildBlockProof). NOTE: each deploy
# builds a FRESH genesis (timestamped now()), so this is for ONE box mining its own
# child — it is NOT a way for many boxes to share one child chain (they'd diverge).
#
# Tunables (env):
#   DATA_DIR          node data dir                         (default /data)
#   RPC_PORT          local node RPC port                   (default 8080)
#   MINER_WORKERS     concurrent worker invocations         (default 1)
#   MINER_BACKEND     cuda | opencl | cpu                   (default cuda)
#   MINER_BATCH_SIZE  nonces per GPU dispatch (default 2e9; large = efficient on GPU)
#   EXTRA_NODE_ARGS   extra lattice-node flags (e.g. --coinbase-address <addr>, or
#                     --min-peer-key-bits N to join a network running a different value)
#   EXTRA_MINER_ARGS  extra coordinator flags
#   CHILD_CHAINS      space/comma-separated child directory name(s) to deploy under
#                     Nexus and merge-mine (e.g. "toy"). Empty = single-chain (default).
#   CHILD_* deploy params (applied to every child in CHILD_CHAINS):
#     CHILD_TARGET_BLOCK_TIME (1000)  CHILD_INITIAL_REWARD (1024)
#     CHILD_HALVING_INTERVAL (210000) CHILD_RETARGET_WINDOW (120)
#     CHILD_PREMINE (0)               CHILD_MAX_TX (100)
#     CHILD_MAX_STATE_GROWTH (100000) CHILD_MAX_BLOCK_SIZE (1000000)
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RPC_PORT="${RPC_PORT:-8080}"
NODE_API="http://127.0.0.1:${RPC_PORT}/api"
mkdir -p "$DATA_DIR"

# Normalize CHILD_CHAINS (allow comma or whitespace separators) into an array.
CHILD_CHAINS="${CHILD_CHAINS:-}"
read -r -a CHILDREN <<< "${CHILD_CHAINS//,/ }"

# Child deploy defaults (mirror SmokeTests deployChild), overridable via env.
CHILD_TARGET_BLOCK_TIME="${CHILD_TARGET_BLOCK_TIME:-1000}"
CHILD_INITIAL_REWARD="${CHILD_INITIAL_REWARD:-1024}"
CHILD_HALVING_INTERVAL="${CHILD_HALVING_INTERVAL:-210000}"
CHILD_RETARGET_WINDOW="${CHILD_RETARGET_WINDOW:-120}"
CHILD_PREMINE="${CHILD_PREMINE:-0}"
CHILD_MAX_TX="${CHILD_MAX_TX:-100}"
CHILD_MAX_STATE_GROWTH="${CHILD_MAX_STATE_GROWTH:-100000}"
CHILD_MAX_BLOCK_SIZE="${CHILD_MAX_BLOCK_SIZE:-1000000}"

# Mirror of the node's deterministicPort(basePort:directory:) (BackgroundLoops.swift):
# FNV-1a-32 over the ASCII directory name, low 14 bits, offset onto the base port. The
# supervised child's RPC port is deterministic from the parent RPC port + directory but
# is NOT returned by /chain/deploy, so we recompute it — then VERIFY by probing the
# child (below), which fails loud if this ever drifts from the node's formula. Child
# directory names are ASCII, matching the node's `directory.utf8` byte iteration.
child_rpc_port() {  # $1=dir  $2=base_rpc_port
  local dir=$1 base=$2 h=2166136261 i c
  for (( i=0; i<${#dir}; i++ )); do
    printf -v c '%d' "'${dir:i:1}"
    h=$(( (h ^ c) & 0xFFFFFFFF ))
    h=$(( (h * 16777619) & 0xFFFFFFFF ))
  done
  printf '%d' $(( base + 1 + (h & 0x3FFF) ))
}

# Block until a node (parent or child) is MINEABLE, not merely up: gate on
# POST /chain/template == 200 (503 while syncing). Uses the chain's own cookie, so a
# wrong port (no/mismatched cookie) can never spuriously pass.
wait_mineable() {  # $1=rpc_port  $2=cookie_file  $3=label
  local port=$1 cookie=$2 label=$3 api="http://127.0.0.1:${port}/api"
  echo "[gpu-miner] waiting for '${label}' to be mineable (chain/template 200 on :${port})…"
  until [ -s "$cookie" ] && \
        curl -fsS -o /dev/null -X POST "${api}/chain/template" \
          -H "Authorization: Bearer $(cat "$cookie" 2>/dev/null)" \
          -H 'content-type: application/json' -d '{}' 2>/dev/null; do
    kill -0 "$NODE_PID" 2>/dev/null || { echo "[gpu-miner] node exited before '${label}' became mineable" >&2; exit 1; }
    sleep 5
  done
}

# Deploy one child under Nexus. The parent (running --supervise-children) auto-spawns
# the supervised child process. Idempotent across restarts: a child persisted from a
# prior run is re-supervised on startup, so a re-deploy returns a 409 "already has data"
# — treated as success.
deploy_child() {  # $1=dir
  local dir=$1 code attempt
  local body
  body=$(printf '{"directory":"%s","parentDirectory":"Nexus","targetBlockTime":%s,"initialReward":%s,"halvingInterval":%s,"premine":%s,"maxTransactionsPerBlock":%s,"maxStateGrowth":%s,"maxBlockSize":%s,"retargetWindow":%s,"wasmPolicies":[],"startMining":true}' \
    "$dir" "$CHILD_TARGET_BLOCK_TIME" "$CHILD_INITIAL_REWARD" "$CHILD_HALVING_INTERVAL" \
    "$CHILD_PREMINE" "$CHILD_MAX_TX" "$CHILD_MAX_STATE_GROWTH" "$CHILD_MAX_BLOCK_SIZE" "$CHILD_RETARGET_WINDOW")
  for attempt in 1 2 3 4 5; do
    code=$(curl -sS -o /tmp/deploy.out -w '%{http_code}' -X POST "${NODE_API}/chain/deploy" \
      -H "Authorization: Bearer $(cat "${DATA_DIR}/.cookie")" \
      -H 'content-type: application/json' -d "$body" 2>/dev/null) || code=000
    if [ "$code" = "200" ]; then echo "[gpu-miner]   deployed child '$dir'"; return 0; fi
    if grep -q "already has data on disk" /tmp/deploy.out 2>/dev/null; then
      echo "[gpu-miner]   child '$dir' already deployed (restored) — continuing"; return 0
    fi
    echo "[gpu-miner]   deploy '$dir' attempt ${attempt}/5 failed (HTTP ${code}): $(head -c 200 /tmp/deploy.out 2>/dev/null)" >&2
    sleep 3
  done
  echo "[gpu-miner] FATAL: could not deploy child '$dir'" >&2
  return 1
}

SUPERVISE_ARG=()
[ "${#CHILDREN[@]}" -gt 0 ] && SUPERVISE_ARG=(--supervise-children)

echo "[gpu-miner] starting node (built-in seeds, default key-bits → source-agnostic sync)…"
[ "${#CHILDREN[@]}" -gt 0 ] && echo "[gpu-miner] merged mining enabled for child chain(s): ${CHILDREN[*]}"
# shellcheck disable=SC2086
lattice-node --autosize --data-dir "$DATA_DIR" --rpc-port "$RPC_PORT" \
  "${SUPERVISE_ARG[@]}" ${EXTRA_NODE_ARGS:-} &
NODE_PID=$!
trap 'kill "$NODE_PID" 2>/dev/null || true' EXIT INT TERM

# Parent (Nexus) must be synced before it can deploy/build a child genesis.
wait_mineable "$RPC_PORT" "${DATA_DIR}/.cookie" "Nexus"

# Deploy + await each child, and accumulate the coordinator's merged-mining flags
# (--child-node <loopback url> + --child-rpc-cookie-file), mirroring LatticeMiner.
CHILD_COORD_ARGS=()
for dir in "${CHILDREN[@]}"; do
  [ -z "$dir" ] && continue
  echo "[gpu-miner] deploying child chain '$dir' under Nexus…"
  deploy_child "$dir"
  crpc=$(child_rpc_port "$dir" "$RPC_PORT")
  ccookie="${DATA_DIR}/children/${dir}/.cookie"
  wait_mineable "$crpc" "$ccookie" "Nexus/${dir}"
  # Fail loud if the derived port doesn't actually serve this child (formula drift).
  if ! curl -fsS "http://127.0.0.1:${crpc}/api/chain/info" \
        -H "Authorization: Bearer $(cat "$ccookie")" 2>/dev/null \
      | jq -e --arg d "$dir" '.chains[] | select(.directory==$d)' >/dev/null; then
    echo "[gpu-miner] FATAL: derived child RPC port ${crpc} does not serve chain '$dir'" >&2
    exit 1
  fi
  echo "[gpu-miner]   child '$dir' mineable on :${crpc}"
  CHILD_COORD_ARGS+=(--child-node "http://127.0.0.1:${crpc}/api" --child-rpc-cookie-file "$ccookie")
done

# GPU batch size: the coordinator default (10k nonces/batch) is tuned for CPU workers.
# A GPU worker is spawned per batch, so a tiny batch means the run is dominated by
# CUDA init/teardown instead of hashing (GPU reads ~idle). Use a large dispatch so each
# kernel launch does real work — matches the docs ("raise --batch-size so each GPU
# dispatch is large") and the Mac/vast configs.
MINER_BATCH_SIZE="${MINER_BATCH_SIZE:-2000000000}"

echo "[gpu-miner] node mineable — starting coordinator (${MINER_BACKEND:-cuda}, ${MINER_WORKERS:-1} worker(s), batch ${MINER_BATCH_SIZE}${CHILD_CHAINS:+, children: ${CHILDREN[*]}})"
# The coordinator has no --backend flag and never passes one to the worker, so the GPU
# backend is forced by the cuda-worker shim (which reads MINER_BACKEND from the env).
# Coinbase = the node's identity (a fresh non-premine key unless EXTRA_NODE_ARGS sets
# --coinbase-address). stdbuf -oL keeps coordinator output line-buffered so container
# logs show mining progress live instead of block-buffering it.
# shellcheck disable=SC2086
exec stdbuf -oL -eL lattice-mining-coordinator \
  --node "$NODE_API" \
  --rpc-cookie-file "${DATA_DIR}/.cookie" \
  --worker-executable /usr/local/bin/lattice-cuda-worker \
  --workers "${MINER_WORKERS:-1}" \
  --batch-size "$MINER_BATCH_SIZE" \
  "${CHILD_COORD_ARGS[@]}" \
  ${EXTRA_MINER_ARGS:-}
