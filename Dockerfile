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
FROM ghcr.io/adalinxx/lattice-node:main AS node

# ── Stage 2: the self-contained GPU miner ─────────────────────────────────────
# The devel base guarantees libnvrtc (cudarc NVRTC-compiles the kernel at run time);
# libcuda.so.1 is injected from the host driver by the NVIDIA container runtime. The
# Swift binaries are static-swift-stdlib but still link the node's shared apt deps,
# so install the same set the node image uses.
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dnsutils \
    jq \
    libatomic1 \
    libcurl4 \
    libjavascriptcoregtk-4.1-0 \
    libsqlite3-0 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/lattice-node            /usr/local/bin/lattice-node
COPY --from=node /usr/local/bin/lattice-mining-coordinator /usr/local/bin/lattice-mining-coordinator
COPY --from=worker /src/target/release/lattice-miner-gpu /usr/local/bin/lattice-miner-gpu
COPY deploy/gpu-entrypoint.sh /usr/local/bin/gpu-entrypoint
RUN chmod +x /usr/local/bin/gpu-entrypoint

VOLUME /data
ENTRYPOINT ["/usr/local/bin/gpu-entrypoint"]
