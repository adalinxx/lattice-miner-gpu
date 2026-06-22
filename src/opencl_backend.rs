//! OpenCL backend — covers AMD, NVIDIA, and Intel GPUs (any OpenCL 1.2+ ICD).
//!
//! Each work-item resumes SHA-256 from the host-computed prefix midstate over the
//! 1–2 final blocks that hold the 8-byte nonce, compares the digest to the target
//! (big-endian), and reports the smallest winning global id via an atomic min.
//! This mirrors `metal_backend::search` and `sha256::finalize_from_midstate`
//! bit-for-bit; the host re-verifies every hit, so a kernel bug fails loudly.
//!
//! Feature-gated (`--features opencl`): keeps the default macOS/Metal build free
//! of the OpenCL link dependency.

use crate::sha256::{self, Midstate};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_uint, cl_ulong, CL_BLOCKING};
use std::ptr;

/// OpenCL C kernel. `rotate(x, n)` is a built-in left-rotate, so `rotr(x,n) =
/// rotate(x, 32-n)` — identical to the Metal shader.
const KERNEL_SRC: &str = r#"
__constant uint K[64] = {
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
inline uint ssig0(uint x){ return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
inline uint ssig1(uint x){ return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

// 16-word SLIDING message schedule (w[16] not w[64]) so it stays in registers
// instead of spilling to private/local memory. Bit-for-bit identical output to
// the reference; the host re-verifies every hit.
inline void compress(uint *state, const uint *win) {
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = win[i];
    uint a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (int t = 0; t < 64; t++) {
        uint wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = w[t & 15] + ssig0(w[(t + 1) & 15]) + w[(t + 9) & 15] + ssig1(w[(t + 14) & 15]);
            w[t & 15] = wt;
        }
        uint S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint ch = g ^ (e & (f ^ g));
        uint t1 = h + S1 + ch + K[t] + wt;
        uint S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint maj = (a & b) ^ (c & (a ^ b));
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__kernel void search(
    __global const uint *midstate,       // 8 words
    __global const uint *region_words,   // num_blocks*16 words
    const uint num_blocks,
    const uint nonce_off,
    const ulong base_nonce,
    __global const uint *target,         // 8 words (big-endian)
    __global volatile uint *found)       // smallest winning gid, init to 0xffffffff
{
    ulong gid = get_global_id(0);
    ulong nonce = base_nonce + gid;

    uint state[8];
    for (uint i = 0; i < 8; i++) state[i] = midstate[i];

    uint msg[32];
    uint nwords = num_blocks * 16u;
    for (uint i = 0; i < nwords; i++) msg[i] = region_words[i];
    for (uint k = 0; k < 8u; k++) {
        uint gb = nonce_off + k;
        uint byte = (uint)((nonce >> (8u * (7u - k))) & 0xffUL);
        msg[gb >> 2] |= byte << ((3u - (gb & 3u)) * 8u);
    }
    for (uint b = 0; b < num_blocks; b++) compress(state, &msg[b * 16u]);

    bool le = true;
    for (uint i = 0; i < 8u; i++) {
        if (state[i] != target[i]) { le = state[i] < target[i]; break; }
    }
    if (le) atomic_min(found, (uint)gid);
}
"#;

/// Build the final SHA-256 region (nonce zeroed) as big-endian words, plus
/// num_blocks and the nonce byte offset. Identical layout to the Metal backend.
fn region_words_for(ms: &Midstate) -> (Vec<u32>, u32, u32) {
    let mut region = ms.tail.clone();
    region.extend_from_slice(&[0u8; 8]); // nonce placeholder
    region.push(0x80);
    while (region.len() + 8) % 64 != 0 {
        region.push(0);
    }
    region.extend_from_slice(&((ms.total_len as u64) * 8).to_be_bytes());
    let num_blocks = (region.len() / 64) as u32;
    let nonce_off = ms.tail.len() as u32;
    let words = region
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (words, num_blocks, nonce_off)
}

/// OpenCL search over `[start, start+count)`. Returns the first winning
/// `(nonce, digest)` or `None`. Errors (no device / build failure) surface as a
/// printed message and `None`, so the coordinator simply re-assigns the range.
/// True if at least one OpenCL GPU device is present (for `--backend auto`).
pub fn is_available() -> bool {
    get_all_devices(CL_DEVICE_TYPE_GPU)
        .map(|d| !d.is_empty())
        .unwrap_or(false)
}

pub fn search_opencl(
    prefix: &[u8],
    target: &[u8; 32],
    start: u64,
    count: u64,
) -> Option<(u64, [u8; 32])> {
    match run(prefix, target, start, count) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("error: opencl backend failed: {e}");
            None
        }
    }
}

fn run(
    prefix: &[u8],
    target: &[u8; 32],
    start: u64,
    count: u64,
) -> Result<Option<(u64, [u8; 32])>, String> {
    let ms = sha256::midstate_for_prefix(prefix);
    let (region_words, num_blocks, nonce_off) = region_words_for(&ms);

    let mut tgt = [0u32; 8];
    for i in 0..8 {
        tgt[i] = u32::from_be_bytes([
            target[4 * i], target[4 * i + 1], target[4 * i + 2], target[4 * i + 3],
        ]);
    }

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|e| format!("get_all_devices: {e}"))?
        .first()
        .ok_or("no OpenCL GPU device found")?;
    let device = Device::new(device_id);
    let context = Context::from_device(&device).map_err(|e| format!("context: {e}"))?;
    // OpenCL 1.2 `clCreateCommandQueue` (works on every ICD incl. macOS 1.2 and
    // older AMD; the 2.0 `…WithProperties` variant isn't present on macOS).
    let queue = unsafe { CommandQueue::create(&context, device_id, 0) }
        .map_err(|e| format!("queue: {e}"))?;
    let program = Program::create_and_build_from_source(&context, KERNEL_SRC, "")
        .map_err(|e| format!("kernel build: {e}"))?;
    let kernel = Kernel::create(&program, "search").map_err(|e| format!("kernel: {e}"))?;

    // Immutable inputs.
    let mut ms_buf = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 8, ptr::null_mut())
            .map_err(|e| format!("ms_buf: {e}"))?
    };
    let mut region_buf = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, region_words.len(), ptr::null_mut())
            .map_err(|e| format!("region_buf: {e}"))?
    };
    let mut tgt_buf = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_ONLY, 8, ptr::null_mut())
            .map_err(|e| format!("tgt_buf: {e}"))?
    };
    let mut found_buf = unsafe {
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut())
            .map_err(|e| format!("found_buf: {e}"))?
    };
    unsafe {
        queue.enqueue_write_buffer(&mut ms_buf, CL_BLOCKING, 0, &ms.state, &[])
            .map_err(|e| format!("write ms: {e}"))?;
        queue.enqueue_write_buffer(&mut region_buf, CL_BLOCKING, 0, &region_words, &[])
            .map_err(|e| format!("write region: {e}"))?;
        queue.enqueue_write_buffer(&mut tgt_buf, CL_BLOCKING, 0, &tgt, &[])
            .map_err(|e| format!("write tgt: {e}"))?;
    }

    let chunk: u64 = 16_000_000;
    let mut base = start;
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(chunk);
        unsafe {
            queue.enqueue_write_buffer(&mut found_buf, CL_BLOCKING, 0, &[u32::MAX], &[])
                .map_err(|e| format!("reset found: {e}"))?;
            ExecuteKernel::new(&kernel)
                .set_arg(&ms_buf)
                .set_arg(&region_buf)
                .set_arg(&(num_blocks as cl_uint))
                .set_arg(&(nonce_off as cl_uint))
                .set_arg(&(base as cl_ulong))
                .set_arg(&tgt_buf)
                .set_arg(&found_buf)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&queue)
                .map_err(|e| format!("enqueue: {e}"))?;
            queue.finish().map_err(|e| format!("finish: {e}"))?;
        }

        let mut gid = [u32::MAX];
        unsafe {
            queue.enqueue_read_buffer(&found_buf, CL_BLOCKING, 0, &mut gid, &[])
                .map_err(|e| format!("read found: {e}"))?;
        }
        if gid[0] != u32::MAX {
            let nonce = base + gid[0] as u64;
            let hash = sha256::finalize_from_midstate(&ms, nonce);
            assert!(
                crate::meets_target(&hash, target),
                "OpenCL reported a nonce whose hash exceeds target — kernel bug"
            );
            return Ok(Some((nonce, hash)));
        }
        base += n;
        remaining -= n;
    }
    Ok(None)
}
