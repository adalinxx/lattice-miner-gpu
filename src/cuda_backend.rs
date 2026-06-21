//! CUDA backend — native performance on NVIDIA GPUs (incl. cloud rental:
//! vast.ai / RunPod / Lambda). Uses `cudarc` with dynamic loading (libcuda is
//! resolved at runtime, so the host binary builds without the CUDA toolkit) and
//! NVRTC (the kernel is compiled at runtime, mirroring the Metal/OpenCL paths).
//!
//! Each thread resumes SHA-256 from the host-computed prefix midstate over the
//! 1–2 final blocks holding the 8-byte nonce, compares the digest to the target
//! (big-endian), and atomic-mins the smallest winning global id. Mirrors
//! `metal_backend::search` / `sha256::finalize_from_midstate` bit-for-bit; the
//! host re-verifies every hit. Feature-gated (`--features cuda`).

use crate::sha256::{self, Midstate};
use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

/// CUDA-C kernel (compiled by NVRTC at runtime). `count` guards threads launched
/// beyond the requested range (grid is rounded up to the block size).
const KERNEL_SRC: &str = r#"
__constant__ unsigned int K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32u - n));
}

__device__ void compress(unsigned int *state, const unsigned int *win) {
    unsigned int w[64];
    #pragma unroll
    for (unsigned int t = 0; t < 16; t++) w[t] = win[t];
    #pragma unroll
    for (unsigned int t = 16; t < 64; t++) {
        unsigned int s0 = rotr(w[t-15],7) ^ rotr(w[t-15],18) ^ (w[t-15] >> 3);
        unsigned int s1 = rotr(w[t-2],17) ^ rotr(w[t-2],19) ^ (w[t-2] >> 10);
        w[t] = w[t-16] + s0 + w[t-7] + s1;
    }
    unsigned int a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    #pragma unroll
    for (unsigned int t = 0; t < 64; t++) {
        unsigned int S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        unsigned int ch = g ^ (e & (f ^ g));
        unsigned int t1 = h + S1 + ch + K[t] + w[t];
        unsigned int S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        unsigned int maj = (a & b) ^ (c & (a ^ b));
        unsigned int t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

extern "C" __global__ void search(
    const unsigned int *midstate,
    const unsigned int *region_words,
    unsigned int num_blocks,
    unsigned int nonce_off,
    unsigned long long base_nonce,
    unsigned long long count,
    const unsigned int *target,
    unsigned int *found)
{
    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    unsigned long long nonce = base_nonce + gid;

    unsigned int state[8];
    for (unsigned int i = 0; i < 8; i++) state[i] = midstate[i];

    unsigned int msg[32];
    unsigned int nwords = num_blocks * 16u;
    for (unsigned int i = 0; i < nwords; i++) msg[i] = region_words[i];
    for (unsigned int k = 0; k < 8u; k++) {
        unsigned int gb = nonce_off + k;
        unsigned int byte = (unsigned int)((nonce >> (8u * (7u - k))) & 0xffULL);
        msg[gb >> 2] |= byte << ((3u - (gb & 3u)) * 8u);
    }
    for (unsigned int b = 0; b < num_blocks; b++) compress(state, &msg[b * 16u]);

    bool le = true;
    for (unsigned int i = 0; i < 8u; i++) {
        if (state[i] != target[i]) { le = state[i] < target[i]; break; }
    }
    if (le) atomicMin(found, (unsigned int)gid);
}
"#;

/// Build the final SHA-256 region (nonce zeroed) as big-endian words; identical
/// layout to the Metal/OpenCL backends.
fn region_words_for(ms: &Midstate) -> (Vec<u32>, u32, u32) {
    let mut region = ms.tail.clone();
    region.extend_from_slice(&[0u8; 8]);
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

pub fn search_cuda(
    prefix: &[u8],
    target: &[u8; 32],
    start: u64,
    count: u64,
) -> Option<(u64, [u8; 32])> {
    match run(prefix, target, start, count) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("error: cuda backend failed: {e}");
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

    let dev = CudaDevice::new(0).map_err(|e| format!("CudaDevice::new: {e}"))?;
    let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc compile: {e}"))?;
    dev.load_ptx(ptx, "pow", &["search"])
        .map_err(|e| format!("load_ptx: {e}"))?;
    let func = dev.get_func("pow", "search").ok_or("kernel fn not found")?;

    let ms_dev = dev.htod_copy(ms.state.to_vec()).map_err(|e| format!("htod ms: {e}"))?;
    let region_dev = dev.htod_copy(region_words.clone()).map_err(|e| format!("htod region: {e}"))?;
    let tgt_dev = dev.htod_copy(tgt.to_vec()).map_err(|e| format!("htod tgt: {e}"))?;

    let block = 256u32;
    let chunk: u64 = 16_000_000;
    let mut base = start;
    let mut remaining = count;
    while remaining > 0 {
        let n = remaining.min(chunk);
        let found_dev = dev.htod_copy(vec![u32::MAX]).map_err(|e| format!("htod found: {e}"))?;
        let grid = ((n + block as u64 - 1) / block as u64) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.clone()
                .launch(
                    cfg,
                    (
                        &ms_dev,
                        &region_dev,
                        num_blocks,
                        nonce_off,
                        base,
                        n,
                        &tgt_dev,
                        &found_dev,
                    ),
                )
                .map_err(|e| format!("launch: {e}"))?;
        }
        let gid = dev.dtoh_sync_copy(&found_dev).map_err(|e| format!("dtoh found: {e}"))?[0];
        if gid != u32::MAX {
            let nonce = base + gid as u64;
            let hash = sha256::finalize_from_midstate(&ms, nonce);
            assert!(
                crate::meets_target(&hash, target),
                "CUDA reported a nonce whose hash exceeds target — kernel bug"
            );
            return Ok(Some((nonce, hash)));
        }
        base += n;
        remaining -= n;
    }
    Ok(None)
}

// Scalar kernel args are plain Copy types; cudarc derives DeviceRepr for the
// primitive u32/u64 we pass, so no extra impls are needed.
#[allow(dead_code)]
fn _assert_devicerepr<T: DeviceRepr>() {}
