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
- [ ] **Phase 3 — kernel optimization** (precomputed message words, high-word
      early-exit, full unroll, constant-memory K) — headroom to ~1 GH/s.
- [ ] **Phase 4 — CUDA** (`cudarc` + PTX), for NVIDIA (needs NVIDIA hardware).

Currently on the `metal` crate; objc2-metal is the eventual migration. **Metal
runs on the host** (Apple GPU is not visible inside Linux containers), so the
coordinator + this worker run natively on macOS against a node's RPC.

The architecture is a thin Rust host calling per-vendor native kernels (not a
single portable shader): SHA-256's hot path is rotates and 3-input booleans,
which native ISAs do in one instruction and portable shader languages don't.

## Build & run

```bash
cargo build --release
./target/release/lattice-miner-gpu \
  --work-id w1 --prefix-hex deadbeef \
  --target ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --start-nonce 0 --count 1000000
```

```bash
cargo test   # includes reference PoW vectors
```

## Use with a node

Point the coordinator at this binary:

```bash
lattice-mining-coordinator \
  --node http://127.0.0.1:8080/api --rpc-cookie-file ~/.lattice/.cookie \
  --worker-executable ./target/release/lattice-miner-gpu --workers 1
```
