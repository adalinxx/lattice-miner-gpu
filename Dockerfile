# syntax=docker/dockerfile:1
#
# Self-contained CUDA GPU miner image for cloud rental (vast.ai / RunPod / Lambda).
# Composes the CUDA PoW worker built here with the Lattice node + mining coordinator
# from the published node image. See deploy/gpu-entrypoint.sh for the run model.
#
# Run:  docker run --gpus all ghcr.io/adalinxx/lattice-miner-gpu:main
# libcuda is provided by the host driver via the NVIDIA container runtime.

# ── Stage 1: build the CUDA PoW worker (Rust) ─────────────────────────────────
# cudarc dynamically loads libcuda at run time and NVRTC-compiles the kernel, so the
# build needs only Rust — no CUDA toolkit (mirrors the linux-gpu-features CI job).
FROM rust:1-bookworm AS worker
WORKDIR /src
COPY . .
RUN cargo build --release --features cuda

# ── The Lattice node + coordinator (already built, static-swift-stdlib) ───────
FROM ghcr.io/adalinxx/lattice-node:sha-8637025 AS node

# ── Stage 2: the self-contained GPU miner ─────────────────────────────────────
# RUNTIME base (not devel): the only CUDA piece needed at run time is libnvrtc (cudarc
# NVRTC-compiles the kernel); libcuda.so.1 is injected from the host driver by the NVIDIA
# container runtime. The devel base drags in the whole toolkit (~7GB unpacked → slow cold
# pulls on every fresh rental host); runtime + the single cuda-nvrtc package is ~half that.
# The Swift binaries are static-swift-stdlib but still link the node's shared apt deps, so
# install the same set the node image uses.
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cuda-nvrtc-12-6 \
    curl \
    dnsutils \
    jq \
    libatomic1 \
    libcurl4 \
    libjavascriptcoregtk-4.1-0 \
    libsqlite3-0 \
    libxml2 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/lattice-node            /usr/local/bin/lattice-node
COPY --from=node /usr/local/bin/lattice                 /usr/local/bin/lattice
COPY --from=node /usr/local/bin/lattice-mining-coordinator /usr/local/bin/lattice-mining-coordinator
# The reference mining supervisor (one coordinator batch per block, one
# pre-signed reward per block, in nonce order), pinned to the same node release.
ADD https://raw.githubusercontent.com/adalinxx/lattice-node/8637025d/deploy/mine-supervisor.py /usr/local/bin/mine-supervisor.py
RUN chmod +x /usr/local/bin/mine-supervisor.py
COPY --from=worker /src/target/release/lattice-miner-gpu /usr/local/bin/lattice-miner-gpu
# Backend shim: the coordinator can't pass --backend to the worker, so force it here.
COPY deploy/cuda-worker.sh /usr/local/bin/lattice-cuda-worker
COPY deploy/gpu-entrypoint.sh /usr/local/bin/gpu-entrypoint
RUN chmod +x /usr/local/bin/lattice-cuda-worker /usr/local/bin/gpu-entrypoint

VOLUME /data
ENTRYPOINT ["/usr/local/bin/gpu-entrypoint"]
