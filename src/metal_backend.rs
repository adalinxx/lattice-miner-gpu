//! Metal (Apple Silicon) backend. Each GPU thread resumes SHA-256 from the
//! host-computed prefix midstate over the 1–2 final blocks that hold the
//! 8-byte nonce, compares the digest to the target (big-endian), and reports the
//! smallest winning thread index. Mirrors `sha256::finalize_from_midstate`,
//! which is the oracle this is verified against.

use crate::sha256::{self, Midstate};
use metal::{
    CompileOptions, ComputePipelineDescriptor, Device, MTLResourceOptions, MTLSize,
};
use std::mem::size_of;

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

inline uint rotr(uint x, uint n) { return rotate(x, 32u - n); }

struct Params {
    uint num_blocks;
    uint nonce_off;
    ulong base_nonce;
    uint target[8];
};

inline void compress(thread uint* state, thread const uint* win) {
    uint w[64];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < 16; t++) w[t] = win[t];
    #pragma clang loop unroll(full)
    for (uint t = 16; t < 64; t++) {
        uint s0 = rotr(w[t-15],7) ^ rotr(w[t-15],18) ^ (w[t-15] >> 3);
        uint s1 = rotr(w[t-2],17) ^ rotr(w[t-2],19) ^ (w[t-2] >> 10);
        w[t] = w[t-16] + s0 + w[t-7] + s1;
    }
    uint a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < 64; t++) {
        uint S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint ch = g ^ (e & (f ^ g));               // (e & f) ^ (~e & g)
        uint t1 = h + S1 + ch + K[t] + w[t];
        uint S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint maj = (a & b) ^ (c & (a ^ b));        // (a&b) ^ (a&c) ^ (b&c)
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

kernel void search(
    constant uint*  midstate      [[buffer(0)]],
    constant uint*  region_words  [[buffer(1)]],
    constant Params& p            [[buffer(2)]],
    device atomic_uint* found     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong nonce = p.base_nonce + (ulong)gid;
    uint state[8];
    for (uint i = 0; i < 8; i++) state[i] = midstate[i];

    // Precomputed big-endian message words (nonce region zeroed); OR in the 8
    // nonce bytes, then compress the 1-2 final blocks.
    uint msg[32];
    uint nwords = p.num_blocks * 16;
    for (uint i = 0; i < nwords; i++) msg[i] = region_words[i];
    for (uint k = 0; k < 8; k++) {
        uint gb = p.nonce_off + k;
        uint byte = (uint)((nonce >> (8 * (7 - k))) & 0xff);
        msg[gb >> 2] |= byte << ((3u - (gb & 3u)) * 8u);
    }
    for (uint b = 0; b < p.num_blocks; b++) {
        compress(state, &msg[b * 16]);
    }

    // big-endian digest <= target ?
    bool le = true;
    for (uint i = 0; i < 8; i++) {
        if (state[i] != p.target[i]) { le = state[i] < p.target[i]; break; }
    }
    if (le) {
        atomic_fetch_min_explicit(found, gid, memory_order_relaxed);
    }
}
"#;

#[repr(C)]
struct Params {
    num_blocks: u32,
    nonce_off: u32,
    base_nonce: u64,
    target: [u32; 8],
}

/// Build the final region (nonce zeroed) + return (region, num_blocks, nonce_off).
fn region_for(ms: &Midstate) -> (Vec<u8>, u32, u32) {
    let mut region = ms.tail.clone();
    region.extend_from_slice(&[0u8; 8]); // nonce placeholder (kernel fills it)
    region.push(0x80);
    while (region.len() + 8) % 64 != 0 {
        region.push(0);
    }
    region.extend_from_slice(&((ms.total_len as u64) * 8).to_be_bytes());
    let num_blocks = (region.len() / 64) as u32;
    (region, num_blocks, ms.tail.len() as u32)
}

/// GPU search over `[start, start+count)`. Returns the first winning
/// `(nonce, digest)` or `None`. The digest is recomputed (and re-verified) on
/// the host, so a kernel bug surfaces as a failed `meets_target` assertion.
pub fn search_metal(
    prefix: &[u8],
    target: &[u8; 32],
    start: u64,
    count: u64,
) -> Option<(u64, [u8; 32])> {
    let ms = sha256::midstate_for_prefix(prefix);
    let (region, num_blocks, nonce_off) = region_for(&ms);
    let region_words: Vec<u32> = region
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut tgt = [0u32; 8];
    for i in 0..8 {
        tgt[i] = u32::from_be_bytes([
            target[4 * i], target[4 * i + 1], target[4 * i + 2], target[4 * i + 3],
        ]);
    }

    let device = Device::system_default().expect("no Metal device");
    let library = device
        .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
        .expect("kernel compile failed");
    let function = library.get_function("search", None).expect("no kernel fn");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    let pipeline = device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline failed");
    let queue = device.new_command_queue();

    let opt = MTLResourceOptions::StorageModeShared;
    let ms_buf = device.new_buffer_with_data(ms.state.as_ptr() as *const _, 32, opt);
    let region_buf = device.new_buffer_with_data(
        region_words.as_ptr() as *const _,
        (region_words.len() * 4) as u64,
        opt,
    );
    let found_buf = device.new_buffer(4, opt);

    let tg = pipeline.max_total_threads_per_threadgroup().min(256);
    let chunk: u64 = 16_000_000;
    let mut base = start;
    let mut remaining = count;

    while remaining > 0 {
        let n = remaining.min(chunk);
        unsafe { *(found_buf.contents() as *mut u32) = u32::MAX; }

        let params = Params {
            num_blocks,
            nonce_off,
            base_nonce: base,
            target: tgt,
        };
        let params_buf = device.new_buffer_with_data(
            &params as *const _ as *const _,
            size_of::<Params>() as u64,
            opt,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&ms_buf), 0);
        enc.set_buffer(1, Some(&region_buf), 0);
        enc.set_buffer(2, Some(&params_buf), 0);
        enc.set_buffer(3, Some(&found_buf), 0);
        enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gid = unsafe { *(found_buf.contents() as *const u32) };
        if gid != u32::MAX {
            let nonce = base + gid as u64;
            let hash = sha256::finalize_from_midstate(&ms, nonce);
            assert!(
                crate::meets_target(&hash, target),
                "GPU reported a nonce whose hash exceeds target — kernel bug"
            );
            return Some((nonce, hash));
        }
        base += n;
        remaining -= n;
    }
    None
}
