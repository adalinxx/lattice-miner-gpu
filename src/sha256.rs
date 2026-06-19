//! A minimal SHA-256 that exposes the **midstate** (the 8-word state after the
//! complete 64-byte blocks of a constant prefix). The host computes the midstate
//! once; each GPU thread resumes from it over only the 1–2 final blocks that
//! contain the 8-byte nonce. `finalize_from_midstate` is the exact computation
//! the Metal/CUDA kernels mirror — it is the per-nonce oracle.

pub const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// One SHA-256 compression of a 64-byte block into `state`.
pub fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for t in 0..16 {
        w[t] = u32::from_be_bytes([
            block[4 * t], block[4 * t + 1], block[4 * t + 2], block[4 * t + 3],
        ]);
    }
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16]
            .wrapping_add(s0)
            .wrapping_add(w[t - 7])
            .wrapping_add(s1);
    }
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];
    for t in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[t])
            .wrapping_add(w[t]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Constant inputs the GPU kernel needs to resume hashing per nonce.
#[derive(Clone)]
pub struct Midstate {
    /// State after the complete 64-byte blocks of the prefix.
    pub state: [u32; 8],
    /// Remaining prefix bytes (prefix.len() % 64), the head of the final block(s).
    pub tail: Vec<u8>,
    /// Total preimage length in bytes = prefix.len() + 8 (the nonce).
    pub total_len: usize,
}

/// Decompose `prefix` into the constant midstate + tail (preimage = prefix‖nonce8).
pub fn midstate_for_prefix(prefix: &[u8]) -> Midstate {
    let mut state = H0;
    let full = prefix.len() / 64;
    for i in 0..full {
        let mut block = [0u8; 64];
        block.copy_from_slice(&prefix[i * 64..i * 64 + 64]);
        compress(&mut state, &block);
    }
    Midstate {
        state,
        tail: prefix[full * 64..].to_vec(),
        total_len: prefix.len() + 8,
    }
}

/// `SHA256(prefix ‖ nonce_be64)` computed by resuming from the midstate over the
/// final block(s) — the exact per-nonce computation the GPU kernel performs.
pub fn finalize_from_midstate(ms: &Midstate, nonce: u64) -> [u8; 32] {
    // Final region = tail ‖ nonce_be64 ‖ 0x80 ‖ zeros ‖ (total_len*8 as u64 BE),
    // padded to a multiple of 64.
    let mut region = ms.tail.clone();
    region.extend_from_slice(&nonce.to_be_bytes());
    region.push(0x80);
    while (region.len() + 8) % 64 != 0 {
        region.push(0);
    }
    let bit_len = (ms.total_len as u64) * 8;
    region.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = ms.state;
    for chunk in region.chunks_exact(64) {
        let mut block = [0u8; 64];
        block.copy_from_slice(chunk);
        compress(&mut state, &block);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&state[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn reference(prefix: &[u8], nonce: u64) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(prefix);
        h.update(nonce.to_be_bytes());
        h.finalize().into()
    }

    #[test]
    fn midstate_matches_full_sha256_across_prefix_lengths() {
        // Cover every tail length 0..=130 so the 1-block vs 2-block final region
        // boundary (tail_len >= 48) is exercised.
        for plen in 0..=130usize {
            let prefix: Vec<u8> = (0..plen).map(|i| (i * 31 + 7) as u8).collect();
            let ms = midstate_for_prefix(&prefix);
            for nonce in [0u64, 1, 7, 0xdead_beef, u64::MAX] {
                assert_eq!(
                    finalize_from_midstate(&ms, nonce),
                    reference(&prefix, nonce),
                    "mismatch at prefix_len={plen} nonce={nonce}"
                );
            }
        }
    }

    #[test]
    fn reproduces_deadbeef_vectors() {
        let ms = midstate_for_prefix(&hex::decode("deadbeef").unwrap());
        assert_eq!(
            hex::encode(finalize_from_midstate(&ms, 0)),
            "40657e5cc8e75162df7ea33a8fd55daa6e1a46d96502febf77af0883cdf365a4"
        );
        assert_eq!(
            hex::encode(finalize_from_midstate(&ms, 7)),
            "c425e8763ce8ea8cf50dbe9bee6505e6af66b13e85012bb29e5a07c747e0001f"
        );
    }
}
