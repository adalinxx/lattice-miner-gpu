# lattice-miner-gpu

A GPU proof-of-work mining worker for the [Lattice](https://github.com/adalinxx/lattice-node)
network. It implements the **Mining Worker Protocol** — a coordinator
(`lattice-mining-coordinator`) hands it a nonce range and the nonce-independent
PoW prefix; it searches for a nonce where

```
SHA256( prefix || nonce_be64 )  ≤  target
```

and prints one `WorkerResult` JSON object. A worker needs only SHA-256 and the
protocol — no Lattice/block-parsing code. See
`lattice-node/docs/mining-worker-protocol.md` for the normative contract.

## Status

Built in phases, each gated on reproducing the previous one bit-for-bit:

- [x] **Phase 1 — CPU golden reference** (`--backend cpu`). The oracle every
      GPU kernel is tested against.
- [x] **Phase 2 — Metal kernel** (`--backend metal`, Apple Silicon). Resumes
      SHA-256 from a host-computed prefix midstate over the final block(s).
      Verified bit-for-bit against the CPU oracle. **~230 MH/s** un-optimized on
      M-series (vs ~10 MH/s single CPU thread).
- [x] **Phase 3 — kernel optimization** — full unroll, op-reduced Ch/Maj,
      host-precomputed message words. **~834 MH/s** (3.6× the first kernel).
      Remaining headroom (message-schedule precompute, high-word early-exit)
      toward ~1 GH/s.
- [x] **Phase 4 — CUDA** (`--backend cuda`, NVIDIA). `cudarc` + NVRTC compiles
      the kernel at runtime; `libcuda` is loaded dynamically, so the host binary
      builds without the CUDA toolkit and runs on any NVIDIA driver (incl. cloud
      rental: vast.ai / RunPod / Lambda). Same midstate-resume kernel as Metal.
- [x] **Phase 5 — OpenCL** (`--backend opencl`, AMD / NVIDIA / Intel). One
      backend for every non-Apple GPU via the platform OpenCL ICD.

Every backend reproduces the CPU oracle bit-for-bit, and the host
**re-verifies every hit** against `sha256::finalize_from_midstate` before
reporting it — a kernel bug surfaces as a hard `assert!`, never an invalid share.

Backends are a thin Rust host calling per-vendor native kernels (not one portable
shader): SHA-256's hot path is rotates and 3-input booleans, which native ISAs do
in one instruction. **Metal runs only on macOS** (Apple GPU isn't visible inside
Linux containers); **CUDA/OpenCL** are how the network is mined on commodity and
cloud GPUs — essential for hashpower beyond Apple Silicon.

## Build & run

```bash
cargo build --release           # macOS: Metal + CPU
./target/release/lattice-miner-gpu \
  --work-id w1 --prefix-hex deadbeef \
  --target ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --start-nonce 0 --count 1000000 --backend metal
```

Non-Apple GPUs (opt-in features; the default build pulls no GPU toolkit):

```bash
# NVIDIA — builds without the CUDA toolkit (libcuda loaded at runtime).
cargo build --release --features cuda
lattice-miner-gpu ... --backend cuda

# AMD / NVIDIA / Intel — needs an OpenCL loader at build (Linux: ocl-icd-opencl-dev).
cargo build --release --features opencl
lattice-miner-gpu ... --backend opencl
```

```bash
cargo test   # reference PoW vectors + midstate/target oracle (all platforms)
```

## Use with a node

Point the coordinator at this binary:

```bash
lattice-mining-coordinator \
  --node http://127.0.0.1:8080/api --rpc-cookie-file ~/.lattice/.cookie \
  --worker-executable ./target/release/lattice-miner-gpu --workers 1
```

## Docker (cloud GPU rental)

A self-contained CUDA image (`ghcr.io/adalinxx/lattice-miner-gpu:main`, built by CI)
bundles a Lattice node + the mining coordinator + this worker. The node syncs the
backbone via DNS seeds and the coordinator drives the GPU — no peer or wiring to
configure:

```bash
docker run --gpus all -v lattice-data:/data ghcr.io/adalinxx/lattice-miner-gpu:main
```

`libcuda` is provided by the host driver through the NVIDIA container runtime
(`--gpus all`). Tunables via `-e`: `MINER_WORKERS`, `MINER_BACKEND` (cuda|opencl|cpu),
`EXTRA_NODE_ARGS` (e.g. `--coinbase-address <addr>`), `EXTRA_MINER_ARGS`. See
`deploy/gpu-entrypoint.sh`.

**Merged (multi-chain) mining.** Set `-e CHILD_CHAINS="toy"` to also deploy and
merge-mine a child chain of Nexus on this box — one PoW search advances both (easiest
target wins; sealed blocks anchor the child). This is **single-box only**: each deploy
builds a fresh genesis, so it's for one box mining its own child, not many boxes sharing
one child. Child deploy params are overridable (`CHILD_TARGET_BLOCK_TIME`,
`CHILD_INITIAL_REWARD`, …); see `deploy/gpu-entrypoint.sh`.
