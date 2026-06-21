//! `lattice-miner-gpu` — a proof-of-work mining worker for the Lattice network.
//!
//! Implements the Mining Worker Protocol (see `lattice-node/docs/
//! mining-worker-protocol.md`): search one immutable nonce range for a nonce
//! such that `SHA256(prefix || nonce_be64) <= target`, and print one
//! `WorkerResult` JSON object.
//!
//! Phase 1 is this pure-Rust CPU search — the **golden reference**. The Metal
//! (and later CUDA) backends must reproduce it bit-for-bit; the CPU path stays
//! as the oracle the GPU kernels are tested against.

mod sha256;

#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(feature = "cuda")]
mod cuda_backend;

#[cfg(feature = "opencl")]
mod opencl_backend;

use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Search one nonce range for a PoW solution.
#[derive(Parser, Debug)]
#[command(name = "lattice-miner-gpu", version, about)]
struct Args {
    /// Opaque work identifier; echoed back in the result.
    #[arg(long)]
    work_id: String,

    /// Nonce-independent PoW preimage prefix (hex). Preferred input.
    #[arg(long)]
    prefix_hex: Option<String>,

    /// Serialized nonce-0 block (hex). Accepted for contract compatibility but
    /// unused: deriving the prefix from a block needs Lattice, which this worker
    /// deliberately avoids. Supply `--prefix-hex`.
    #[arg(long)]
    block_hex: Option<String>,

    /// PoW target as 256-bit big-endian hex. A nonce wins iff `digest <= target`.
    #[arg(long)]
    target: String,

    /// First nonce in this assignment.
    #[arg(long, default_value_t = 0)]
    start_nonce: u64,

    /// Number of nonces to search: `[start_nonce, start_nonce + count)`.
    #[arg(long)]
    count: u64,

    /// Search backend: `metal` (Apple GPU, default), `cuda` (NVIDIA),
    /// `opencl` (AMD/NVIDIA/Intel), or `cpu` (the reference search). cuda/opencl
    /// require the matching build feature.
    #[arg(long, default_value = "metal")]
    backend: String,
}

/// Mirror of the worker `WorkerResult` JSON contract.
#[derive(Serialize)]
struct WorkerResult {
    #[serde(rename = "workId")]
    work_id: String,
    status: &'static str,
    nonce: Option<u64>,
    hash: Option<String>,
    #[serde(rename = "rangeStart")]
    range_start: u64,
    #[serde(rename = "rangeCount")]
    range_count: u64,
}

/// Parse a 256-bit big-endian target from hex (optional `0x`), right-aligned.
fn parse_target(s: &str) -> Option<[u8; 32]> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).ok()?;
    if bytes.len() > 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    Some(out)
}

/// `digest <= target` as unsigned 256-bit big-endian integers.
#[inline]
pub(crate) fn meets_target(digest: &[u8; 32], target: &[u8; 32]) -> bool {
    for i in 0..32 {
        if digest[i] != target[i] {
            return digest[i] < target[i];
        }
    }
    true
}

/// One PoW hash: `SHA256(prefix || nonce_be64)`, resumed from the prefix midstate.
#[inline]
fn pow_hash(midstate: &Sha256, nonce: u64) -> [u8; 32] {
    let mut h = midstate.clone();
    h.update(nonce.to_be_bytes());
    h.finalize().into()
}

/// Golden-reference CPU search over `[start, start+count)`. Returns the first
/// winning `(nonce, digest)`, or `None` if the range is exhausted.
fn search_cpu(prefix: &[u8], target: &[u8; 32], start: u64, count: u64) -> Option<(u64, [u8; 32])> {
    let mut midstate = Sha256::new();
    midstate.update(prefix);

    let mut nonce = start;
    let mut remaining = count;
    while remaining > 0 {
        let digest = pow_hash(&midstate, nonce);
        if meets_target(&digest, target) {
            return Some((nonce, digest));
        }
        nonce = nonce.wrapping_add(1);
        remaining -= 1;
    }
    None
}

fn main() {
    let args = Args::parse();

    let prefix = args
        .prefix_hex
        .as_deref()
        .map(|h| h.strip_prefix("0x").unwrap_or(h))
        .and_then(|h| hex::decode(h).ok())
        .unwrap_or_else(|| {
            eprintln!("error: --prefix-hex is required (this worker does not parse blocks)");
            std::process::exit(2);
        });

    let target = parse_target(&args.target).unwrap_or_else(|| {
        eprintln!("error: invalid --target");
        std::process::exit(2);
    });

    let found = match args.backend.as_str() {
        "cpu" => search_cpu(&prefix, &target, args.start_nonce, args.count),
        "metal" => {
            #[cfg(target_os = "macos")]
            {
                metal_backend::search_metal(&prefix, &target, args.start_nonce, args.count)
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("error: metal backend requires macOS");
                std::process::exit(2);
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                cuda_backend::search_cuda(&prefix, &target, args.start_nonce, args.count)
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("error: cuda backend not compiled in (rebuild with --features cuda)");
                std::process::exit(2);
            }
        }
        "opencl" => {
            #[cfg(feature = "opencl")]
            {
                opencl_backend::search_opencl(&prefix, &target, args.start_nonce, args.count)
            }
            #[cfg(not(feature = "opencl"))]
            {
                eprintln!("error: opencl backend not compiled in (rebuild with --features opencl)");
                std::process::exit(2);
            }
        }
        other => {
            eprintln!("error: unknown backend '{other}' (use metal|cuda|opencl|cpu)");
            std::process::exit(2);
        }
    };

    let result = match found {
        Some((nonce, digest)) => WorkerResult {
            work_id: args.work_id,
            status: "found",
            nonce: Some(nonce),
            hash: Some(hex::encode(digest)),
            range_start: args.start_nonce,
            range_count: args.count,
        },
        None => WorkerResult {
            work_id: args.work_id,
            status: "exhausted",
            nonce: None,
            hash: None,
            range_start: args.start_nonce,
            range_count: args.count,
        },
    };

    println!("{}", serde_json::to_string(&result).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors cross-checked against the Swift reference worker and an
    // independent SHA-256: SHA256("deadbeef" || nonce_be64).
    #[test]
    fn pow_hash_matches_reference_vectors() {
        let mut ms = Sha256::new();
        ms.update(hex::decode("deadbeef").unwrap());

        assert_eq!(
            hex::encode(pow_hash(&ms, 0)),
            "40657e5cc8e75162df7ea33a8fd55daa6e1a46d96502febf77af0883cdf365a4"
        );
        assert_eq!(
            hex::encode(pow_hash(&ms, 7)),
            "c425e8763ce8ea8cf50dbe9bee6505e6af66b13e85012bb29e5a07c747e0001f"
        );
    }

    #[test]
    fn max_target_accepts_first_nonce() {
        let target = [0xff_u8; 32];
        let prefix = hex::decode("deadbeef").unwrap();
        let (nonce, _) = search_cpu(&prefix, &target, 0, 1).unwrap();
        assert_eq!(nonce, 0);
    }

    #[test]
    fn meets_target_is_big_endian() {
        let mut t = [0u8; 32];
        t[0] = 0x01; // target = 2^248
        let mut below = [0u8; 32];
        below[1] = 0xff; // < target
        let mut above = [0u8; 32];
        above[0] = 0x02; // > target
        assert!(meets_target(&below, &t));
        assert!(!meets_target(&above, &t));
        assert!(meets_target(&t, &t)); // equal counts
    }
}
