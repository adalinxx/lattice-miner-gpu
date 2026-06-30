#!/usr/bin/env bash
# Self-contained GPU miner entrypoint.
#
# The CUDA worker (lattice-miner-gpu) does PoW only; it cannot reach the network on
# its own. So this image bundles a Lattice node and the mining coordinator:
#
#   1. lattice-node syncs the chain from the backbone (DNS-seed discovery — no peer
#      to configure), exposing RPC on localhost.
#   2. lattice-mining-coordinator pulls templates from that local node and drives the
#      CUDA worker for the actual proof-of-work, then gossips sealed blocks back.
#
# libcuda is injected from the host driver by the NVIDIA container runtime (vast.ai /
# RunPod / Lambda --gpus all); NVRTC compiles the kernel at run time.
#
# Tunables (env):
#   DATA_DIR          node data dir                         (default /data)
#   RPC_PORT          local node RPC port                   (default 8080)
#   MINER_WORKERS     concurrent worker invocations         (default 1)
#   MINER_BACKEND     cuda | opencl | cpu                   (default cuda)
#   EXTRA_NODE_ARGS   extra lattice-node flags (e.g. --coinbase-address <addr>,
#                     --external-address <ip:port>)
#   EXTRA_MINER_ARGS  extra coordinator flags (e.g. --child-node <url>)
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RPC_PORT="${RPC_PORT:-8080}"
NODE_API="http://127.0.0.1:${RPC_PORT}/api"
mkdir -p "$DATA_DIR"

echo "[gpu-miner] starting node (DNS-seed discovery → backbone)…"
# shellcheck disable=SC2086
lattice-node --autosize --data-dir "$DATA_DIR" --rpc-port "$RPC_PORT" ${EXTRA_NODE_ARGS:-} &
NODE_PID=$!
trap 'kill "$NODE_PID" 2>/dev/null || true' EXIT INT TERM

echo "[gpu-miner] waiting for node RPC + chain sync…"
until curl -fsS "${NODE_API}/chain/info" >/dev/null 2>&1; do
  kill -0 "$NODE_PID" 2>/dev/null || { echo "[gpu-miner] node exited before RPC came up" >&2; exit 1; }
  sleep 5
done

echo "[gpu-miner] node up — starting coordinator (${MINER_BACKEND:-cuda}, ${MINER_WORKERS:-1} worker(s))"
# The coordinator polls templates until the node is synced/mineable, so it tolerates
# a still-catching-up node; it credits the node's coinbase identity (a fresh,
# non-premine key unless EXTRA_NODE_ARGS sets --coinbase-address).
# shellcheck disable=SC2086
exec lattice-mining-coordinator \
  --node "$NODE_API" \
  --rpc-cookie-file "${DATA_DIR}/.cookie" \
  --worker-executable /usr/local/bin/lattice-miner-gpu \
  --backend "${MINER_BACKEND:-cuda}" \
  --workers "${MINER_WORKERS:-1}" \
  ${EXTRA_MINER_ARGS:-}
