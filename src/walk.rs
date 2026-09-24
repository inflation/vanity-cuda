use crate::field::Fe;
use crate::pattern::Matcher;
use color_eyre::eyre::{Result, ensure};
use curve25519_dalek::{MontgomeryPoint, Scalar, constants::X25519_BASEPOINT};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use zeroize::Zeroizing;

/// Random clamped base scalar; key `i` of a lane is `s0 + 8i`.
pub struct Seed(Zeroizing<[u8; 32]>);

impl Seed {
    pub fn random() -> Seed {
        let mut s = Zeroizing::new([0u8; 32]);
        getrandom::fill(s.as_mut()).expect("OS random number generator failed");
        s[0] &= 248;
        s[31] = s[31] & 127 | 64;
        Seed(s)
    }

    /// `s0 + 8 * off`, or `None` if the sum is no longer a clamped scalar.
    pub fn key(&self, off: u64) -> Option<Zeroizing<[u8; 32]>> {
        let mut k = Zeroizing::new(*self.0);
        let mut carry = off as u128 * 8;
        for c in k.as_chunks_mut().0 {
            let sum = u64::from_le_bytes(*c) as u128 + (carry as u64) as u128;
            *c = (sum as u64).to_le_bytes();
            carry = (carry >> 64) + (sum >> 64);
        }
        (k[31] & 0xc0 == 0x40).then_some(k)
    }

    /// Affine u of the public key at `off`.
    pub fn u(&self, off: u64) -> Fe {
        let k = self.key(off).expect("seed offset overflow");
        Fe::from_bytes(&MontgomeryPoint::mul_base_clamped(*k).to_bytes())
    }
}

/// Affine u coordinate of `8n * B`.
pub fn step_point(n: u64) -> Fe {
    Fe::from_bytes(&(X25519_BASEPOINT * Scalar::from(8 * n)).to_bytes())
}

pub struct Hit {
    pub private: Zeroizing<[u8; 32]>,
    pub public: [u8; 32],
}

pub struct Ctx {
    pub matcher: Matcher,
    pub keys: AtomicU64,
    pub stop: AtomicBool,
    pub tx: Sender<Hit>,
}

impl Ctx {
    pub fn done(&self) -> bool {
        self.stop.load(Relaxed)
    }

    pub fn count(&self, n: u64) {
        self.keys.fetch_add(n, Relaxed);
    }

    /// Recompute the key at `off` with dalek, verify it, and report it.
    pub fn report(&self, seed: &Seed, off: u64, prefix: u64) -> Result<()> {
        let Some(private) = seed.key(off) else {
            return Ok(());
        };
        let public = MontgomeryPoint::mul_base_clamped(*private).to_bytes();
        let p = u64::from_be_bytes(public[..8].try_into().unwrap());
        ensure!(
            p == prefix && self.matcher.matches(p),
            "internal error: GPU hit failed verification"
        );
        let _ = self.tx.send(Hit { private, public });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_stays_clamped() {
        let seed = Seed::random();
        let k = seed.key(u64::MAX).unwrap();
        assert_eq!(k[0] & 7, 0);
        assert_eq!(k[31] & 0xc0, 0x40);
    }
}
