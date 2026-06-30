#!/usr/bin/env sh
# Backend shim for the mining coordinator's --worker-executable.
#
# The coordinator invokes the worker with the nonce-range flags (--work-id,
# --block-hex, --target, --start-nonce, --count, --prefix-hex) but has no way to
# pass a --backend — so a bare worker would default to `metal` and exit on Linux.
# This shim forces the GPU backend, then hands the coordinator's flags through.
# MINER_BACKEND (inherited from the container env) lets an operator pick
# cuda|opencl|cpu without a rebuild.
exec /usr/local/bin/lattice-miner-gpu --backend "${MINER_BACKEND:-cuda}" "$@"
