use crate::field::Fe;
use crate::pattern::Matcher;
use curve25519_dalek::{MontgomeryPoint, Scalar, constants::X25519_BASEPOINT};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::Sender;
use zeroize::Zeroizing;

pub type Point = (Fe, Fe);

/// Random clamped base scalar; key `i` of a walk is `s0 + 8i`.
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

    fn point(&self, off: u64) -> Point {
        let k = self.key(off).expect("seed offset overflow");
        (
            Fe::from_bytes(&MontgomeryPoint::mul_base_clamped(*k).to_bytes()),
            Fe::ONE,
        )
    }
}

/// Affine u coordinate of `8n * B`.
pub fn step_point(n: u64) -> Fe {
    Fe::from_bytes(&(X25519_BASEPOINT * Scalar::from(8 * n)).to_bytes())
}

/// Differential addition `p + q` given `d = p - q` and `qp = u(q) + 1`, `qm = u(q) - 1`.
#[inline(always)]
pub fn xadd(p: Point, d: Point, qp: Fe, qm: Fe) -> Point {
    let u = p.0.sub(p.1).mul(qp);
    let v = p.0.add(p.1).mul(qm);
    (d.1.mul(u.add(v).sqr()), d.0.mul(u.sub(v).sqr()))
}

/// Projective points `(s0 + 8m) * B` for `m in start..start + n`.
pub fn chain(seed: &Seed, start: usize, n: usize) -> Vec<Point> {
    let g = step_point(1);
    let (gp, gm) = (g.add(Fe::ONE), g.sub(Fe::ONE));
    let mut c = vec![seed.point(start as u64), seed.point(start as u64 + 1)];
    while c.len() < n {
        let m = c.len();
        c.push(xadd(c[m - 1], c[m - 2], gp, gm));
    }
    c.truncate(n);
    c
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
    pub fn report(&self, seed: &Seed, off: u64, prefix: u64) {
        let Some(private) = seed.key(off) else { return };
        let public = MontgomeryPoint::mul_base_clamped(*private).to_bytes();
        let p = u64::from_be_bytes(public[..8].try_into().unwrap());
        assert!(
            p == prefix && self.matcher.matches(p),
            "internal error: hit failed verification"
        );
        let _ = self.tx.send(Hit { private, public });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_matches_dalek() {
        let seed = Seed::random();
        for (m, p) in chain(&seed, 7, 20).into_iter().enumerate() {
            let expect = MontgomeryPoint::mul_base_clamped(*seed.key(m as u64 + 7).unwrap());
            assert_eq!(p.0.mul(p.1.invert()).to_bytes(), expect.to_bytes());
        }
    }

    #[test]
    fn key_stays_clamped() {
        let seed = Seed::random();
        let k = seed.key(u64::MAX).unwrap();
        assert_eq!(k[0] & 7, 0);
        assert_eq!(k[31] & 0xc0, 0x40);
    }
}
