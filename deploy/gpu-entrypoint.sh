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
# Nexus on THIS box. Child chains run as SEPARATE PROCESSES that subscribe to the local
# Nexus node for blocks, exactly as documented in deploy/README.md ("Per-process child
# chains") and exercised by the SmokeTests (lib/lattice.mjs spawnChild,
# scenarios/persistence/restart-with-children.mjs). For each child this entrypoint runs
# the documented runbook — deploy -> genesis-hex -> spawn -> register-rpc — then the
# coordinator merge-mines Nexus + each child in one PoW search (a solution advances
# whatever chain's difficulty it clears; the easiest target is always cleared). Sealed
# child blocks anchor to Nexus via ChildBlockProof.
#   * The deploy response (genesis-hex + the child's parent-side P2P address) is saved
#     under the child's data dir, so an in-place restart RE-SPAWNS each child from the
#     same genesis against its persisted data (heights/balances preserved) — matching
#     the restart-with-children smoke, no re-deploy.
#   * SINGLE-BOX ONLY: a first deploy builds a FRESH genesis (timestamped now()), so
#     this is ONE box running its own child — NOT a way for many boxes to share one
#     child (a fresh box would deploy a divergent genesis).
#   * Best-effort: if a child can't be established it is SKIPPED and the box keeps
#     mining the rest (down to Nexus-only). A child never blocks Nexus mining.
#
# Tunables (env):
#   DATA_DIR          node data dir                         (default /data)
#   RPC_PORT          local Nexus node RPC port             (default 8080)
#   MINER_WORKERS     concurrent worker invocations         (default 1)
#   MINER_BACKEND     cuda | opencl | cpu                   (default cuda)
#   MINER_BATCH_SIZE  nonces per GPU dispatch (default 2e9; large = efficient on GPU)
#   EXTRA_NODE_ARGS   extra lattice-node flags (e.g. --coinbase-address <addr>)
#   EXTRA_MINER_ARGS  extra coordinator flags
#   CHILD_CHAINS      space/comma-separated child directory name(s) to deploy under
#                     Nexus and merge-mine (e.g. "toy"). Empty = single-chain (default).
#   CHILD_BOOT_TIMEOUT  seconds to wait for a child to become mineable before skipping
#                       it and mining without it                       (default 180)
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
CHILD_BOOT_TIMEOUT="${CHILD_BOOT_TIMEOUT:-180}"

# Child deploy defaults (mirror SmokeTests spawnChild), overridable via env.
CHILD_TARGET_BLOCK_TIME="${CHILD_TARGET_BLOCK_TIME:-1000}"
CHILD_INITIAL_REWARD="${CHILD_INITIAL_REWARD:-1024}"
CHILD_HALVING_INTERVAL="${CHILD_HALVING_INTERVAL:-210000}"
CHILD_RETARGET_WINDOW="${CHILD_RETARGET_WINDOW:-120}"
CHILD_PREMINE="${CHILD_PREMINE:-0}"
CHILD_MAX_TX="${CHILD_MAX_TX:-100}"
CHILD_MAX_STATE_GROWTH="${CHILD_MAX_STATE_GROWTH:-100000}"
CHILD_MAX_BLOCK_SIZE="${CHILD_MAX_BLOCK_SIZE:-1000000}"

if [ "${#CHILDREN[@]}" -gt 0 ] && [ -n "${SKYPILOT_TASK_ID:-}" ]; then
  echo "[gpu-miner] WARNING: CHILD_CHAINS is set on a managed SkyPilot job (SKYPILOT_TASK_ID present)." >&2
  echo "[gpu-miner]          Merged mining is SINGLE-BOX ONLY — a fresh recovery box re-forks a new child genesis." >&2
fi

CHILD_PIDS=()

# authed helpers ------------------------------------------------------------------
rpc_get()  { curl -fsS "${NODE_API}$1" -H "Authorization: Bearer $(cat "${DATA_DIR}/.cookie")" 2>/dev/null; }
rpc_post() { curl -sS -o "$2" -w '%{http_code}' -X POST "${NODE_API}$1" \
               -H "Authorization: Bearer $(cat "${DATA_DIR}/.cookie")" \
               -H 'content-type: application/json' -d "$3" 2>/dev/null; }

# Block until a chain is MINEABLE (POST /chain/template==200 with its own cookie, 503
# while syncing). $4=timeout secs; 0 (parent) waits indefinitely, only bailing if the
# parent process dies; >0 (child) returns 1 after the timeout so the caller degrades to
# fewer chains instead of hanging forever.
wait_mineable() {  # $1=rpc_port  $2=cookie_file  $3=label  $4=timeout_secs
  local port=$1 cookie=$2 label=$3 timeout=${4:-0}
  local api="http://127.0.0.1:${port}/api"
  local waited=0
  echo "[gpu-miner] waiting for '${label}' to be mineable (chain/template 200 on :${port}$([ "$timeout" -gt 0 ] && echo ", timeout ${timeout}s"))…"
  until [ -s "$cookie" ] && \
        curl -fsS -o /dev/null -X POST "${api}/chain/template" \
          -H "Authorization: Bearer $(cat "$cookie" 2>/dev/null)" \
          -H 'content-type: application/json' -d '{}' 2>/dev/null; do
    kill -0 "$NODE_PID" 2>/dev/null || { echo "[gpu-miner] Nexus node exited before '${label}' became mineable" >&2; exit 1; }
    if [ "$timeout" -gt 0 ] && [ "$waited" -ge "$timeout" ]; then
      echo "[gpu-miner] '${label}' not mineable after ${timeout}s" >&2
      return 1
    fi
    sleep 5
    waited=$(( waited + 5 ))
  done
  return 0
}

# Establish one per-process child end-to-end, mirroring deploy/README.md + spawnChild:
#   1. deploy (first boot) OR load the saved deploy info (restart) for its genesis-hex
#      and parent-side P2P address;
#   2. spawn the child as its own lattice-node subscribed to the local Nexus P2P;
#   3. wait until it is mineable and verify identity;
#   4. register its RPC with Nexus and append the coordinator's --child-node flags.
# Best-effort: any failure logs a warning, returns 1, and the child is skipped.
ensure_child() {  # $1=dir  $2=index  (uses globals: NEXUS_DIR, PARENT_P2P)
  local dir=$1 idx=$2
  local childDir="${DATA_DIR}/children/${dir}"
  local deployFile="${childDir}/deploy.json"
  local crpc=$(( RPC_PORT + 10 + idx ))
  local cp2p=$(( ${PARENT_P2P##*:} + 100 + idx ))   # avoid the parent's own P2P port
  local ghex chainP2P code
  mkdir -p "$childDir"

  echo "[gpu-miner] establishing child chain '${dir}' (rpc :${crpc}, p2p :${cp2p})…"
  # Obtain the child's genesis-hex (+ its parent-side P2P address), saved to deploy.json:
  #   - restart with persisted /data  -> reuse the saved deploy.json (re-spawn, no deploy);
  #   - fresh                         -> POST /chain/deploy;
  #   - already deployed (e.g. 409, or deploy.json lost but parent still tracks it)
  #                                    -> GET /chain/genesis to reuse the existing genesis.
  if [ -s "$deployFile" ]; then
    echo "[gpu-miner]   reusing saved deploy for '${dir}' (restart) — re-spawning from persisted genesis"
  else
    local body
    body=$(printf '{"directory":"%s","parentDirectory":"%s","chainPath":["%s","%s"],"targetBlockTime":%s,"initialReward":%s,"halvingInterval":%s,"premine":%s,"maxTransactionsPerBlock":%s,"maxStateGrowth":%s,"maxBlockSize":%s,"retargetWindow":%s,"wasmPolicies":[],"startMining":false}' \
      "$dir" "$NEXUS_DIR" "$NEXUS_DIR" "$dir" "$CHILD_TARGET_BLOCK_TIME" "$CHILD_INITIAL_REWARD" \
      "$CHILD_HALVING_INTERVAL" "$CHILD_PREMINE" "$CHILD_MAX_TX" "$CHILD_MAX_STATE_GROWTH" \
      "$CHILD_MAX_BLOCK_SIZE" "$CHILD_RETARGET_WINDOW")
    code=$(rpc_post "/chain/deploy" /tmp/deploy.out "$body") || code=000
    if [ "$code" = "200" ]; then
      cp /tmp/deploy.out "$deployFile"
    elif rpc_get "/chain/genesis?chainPath=${NEXUS_DIR}/${dir}" > /tmp/gen.out 2>/dev/null \
         && [ -n "$(jq -r '.genesisHex // empty' /tmp/gen.out 2>/dev/null)" ]; then
      echo "[gpu-miner]   child '${dir}' already deployed — reusing existing genesis"
      cp /tmp/gen.out "$deployFile"
    else
      echo "[gpu-miner] WARNING: could not deploy or fetch genesis for child '${dir}' (deploy HTTP ${code}): $(head -c 160 /tmp/deploy.out 2>/dev/null) — mining without it" >&2
      return 1
    fi
  fi
  ghex=$(jq -r '.genesisHex' "$deployFile")
  chainP2P=$(jq -r '.chainP2PAddress // empty' "$deployFile")
  if [ -z "$ghex" ] || [ "$ghex" = "null" ]; then
    echo "[gpu-miner] WARNING: no genesis-hex for child '${dir}' — skipping it" >&2
    return 1
  fi

  # Spawn the child as its own process, subscribed to the local Nexus P2P (per the
  # runbook: boots from embedded genesis, extracts blocks from the parent). --no-dns-seeds
  # because a child never joins mainnet gossip; it gets everything from the parent.
  # shellcheck disable=SC2086
  lattice-node \
    --genesis-hex "$ghex" \
    --chain-directory "$dir" \
    --chain-path "${NEXUS_DIR}/${dir}" \
    --subscribe-p2p "$PARENT_P2P" \
    --peer "${chainP2P:-$PARENT_P2P}" \
    --port "$cp2p" --rpc-port "$crpc" --data-dir "$childDir" \
    --no-dns-seeds &
  CHILD_PIDS+=( $! )

  local ccookie="${childDir}/.cookie"
  if ! wait_mineable "$crpc" "$ccookie" "${NEXUS_DIR}/${dir}" "$CHILD_BOOT_TIMEOUT"; then
    echo "[gpu-miner] WARNING: child '${dir}' did not become mineable on :${crpc} — skipping it" >&2
    return 1
  fi
  if ! curl -fsS "http://127.0.0.1:${crpc}/api/chain/info" \
        -H "Authorization: Bearer $(cat "$ccookie")" 2>/dev/null \
      | jq -e --arg d "$dir" '.chains[] | select(.directory==$d)' >/dev/null; then
    echo "[gpu-miner] WARNING: child '${dir}' on :${crpc} did not report its chain — skipping it" >&2
    return 1
  fi

  # Register the child's RPC with Nexus so chain/map can route (idempotent each boot).
  rpc_post "/chain/register-rpc" /tmp/reg.out \
    "$(printf '{"chainPath":["%s","%s"],"endpoint":"http://127.0.0.1:%s","authToken":"%s"}' \
       "$NEXUS_DIR" "$dir" "$crpc" "$(cat "$ccookie")")" >/dev/null || true

  echo "[gpu-miner]   child '${dir}' mineable on :${crpc}"
  CHILD_COORD_ARGS+=(--child-node "http://127.0.0.1:${crpc}/api" --child-rpc-cookie-file "$ccookie")
  return 0
}

echo "[gpu-miner] starting Nexus node (built-in seeds, default key-bits → source-agnostic sync)…"
[ "${#CHILDREN[@]}" -gt 0 ] && echo "[gpu-miner] merged mining requested for child chain(s): ${CHILDREN[*]}"
# shellcheck disable=SC2086
lattice-node --autosize --data-dir "$DATA_DIR" --rpc-port "$RPC_PORT" ${EXTRA_NODE_ARGS:-} &
NODE_PID=$!
# On teardown, stop the coordinator's node tree. (In a container, exiting the main
# process tears everything down anyway; this covers the pre-coordinator setup phase.)
trap 'kill "$NODE_PID" ${CHILD_PIDS[@]+"${CHILD_PIDS[@]}"} 2>/dev/null || true' EXIT INT TERM

# Nexus must be synced before it can deploy/build a child genesis or mine.
wait_mineable "$RPC_PORT" "${DATA_DIR}/.cookie" "Nexus" 0

CHILD_COORD_ARGS=()
if [ "${#CHILDREN[@]}" -gt 0 ]; then
  NEXUS_DIR=$(rpc_get "/chain/info" | jq -r '.nexus')
  PARENT_P2P=$(rpc_get "/chain/info" | jq -r '.p2pAddress')
  if [ -z "$NEXUS_DIR" ] || [ "$NEXUS_DIR" = "null" ] || [ -z "$PARENT_P2P" ] || [ "$PARENT_P2P" = "null" ]; then
    echo "[gpu-miner] WARNING: could not read Nexus dir / P2P address — mining Nexus only" >&2
  else
    established=0
    for i in "${!CHILDREN[@]}"; do
      dir="${CHILDREN[$i]}"
      [ -z "$dir" ] && continue
      if ensure_child "$dir" "$i"; then established=$(( established + 1 )); fi
    done
    [ "$established" -eq 0 ] && echo "[gpu-miner] NOTE: no requested child chains are available — mining Nexus only" >&2
  fi
fi

# GPU batch size: the coordinator default (10k nonces/batch) is tuned for CPU workers.
# A GPU worker is spawned per batch, so a tiny batch means the run is dominated by
# CUDA init/teardown instead of hashing (GPU reads ~idle). Use a large dispatch so each
# kernel launch does real work — matches the docs and the Mac/vast configs.
MINER_BATCH_SIZE="${MINER_BATCH_SIZE:-2000000000}"

nchild=$(( ${#CHILD_COORD_ARGS[@]} / 2 ))
echo "[gpu-miner] starting coordinator (${MINER_BACKEND:-cuda}, ${MINER_WORKERS:-1} worker(s), batch ${MINER_BATCH_SIZE}$([ "$nchild" -gt 0 ] && echo ", +${nchild} child chain(s)"))"
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
  ${CHILD_COORD_ARGS[@]+"${CHILD_COORD_ARGS[@]}"} \
  ${EXTRA_MINER_ARGS:-}
