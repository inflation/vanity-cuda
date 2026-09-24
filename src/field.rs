const MASK: u64 = (1 << 51) - 1;

/// GF(2^255 - 19) element in radix 2^51.
#[derive(Clone, Copy, Debug)]
pub struct Fe(pub [u64; 5]);

#[inline(always)]
fn m(a: u64, b: u64) -> u128 {
    a as u128 * b as u128
}

impl Fe {
    pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    pub fn from_bytes(b: &[u8; 32]) -> Fe {
        let w = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Fe([
            w(0) & MASK,
            (w(6) >> 3) & MASK,
            (w(12) >> 6) & MASK,
            (w(19) >> 1) & MASK,
            (w(24) >> 12) & MASK,
        ])
    }

    #[inline(always)]
    pub fn add(self, b: Fe) -> Fe {
        let (a, b) = (self.0, b.0);
        Fe([
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
        ])
    }

    /// `b` must be weakly reduced (limbs < 2^52).
    #[inline(always)]
    pub fn sub(self, b: Fe) -> Fe {
        let (a, b) = (self.0, b.0);
        Fe([
            a[0] + 0xFFFFFFFFFFFDA - b[0],
            a[1] + 0xFFFFFFFFFFFFE - b[1],
            a[2] + 0xFFFFFFFFFFFFE - b[2],
            a[3] + 0xFFFFFFFFFFFFE - b[3],
            a[4] + 0xFFFFFFFFFFFFE - b[4],
        ])
        .carry()
    }

    #[inline(always)]
    fn carry(self) -> Fe {
        let mut l = self.0;
        for i in 0..4 {
            l[i + 1] += l[i] >> 51;
            l[i] &= MASK;
        }
        l[0] += (l[4] >> 51) * 19;
        l[4] &= MASK;
        Fe(l)
    }

    #[inline(always)]
    fn reduce_wide(mut c: [u128; 5]) -> Fe {
        let mut l = [0u64; 5];
        for i in 0..4 {
            c[i + 1] += c[i] >> 51;
            l[i] = c[i] as u64 & MASK;
        }
        l[4] = c[4] as u64 & MASK;
        l[0] += (c[4] >> 51) as u64 * 19;
        l[1] += l[0] >> 51;
        l[0] &= MASK;
        Fe(l)
    }

    #[inline(always)]
    pub fn mul(self, b: Fe) -> Fe {
        let [a0, a1, a2, a3, a4] = self.0;
        let [b0, b1, b2, b3, b4] = b.0;
        let (b1_19, b2_19, b3_19, b4_19) = (b1 * 19, b2 * 19, b3 * 19, b4 * 19);
        Fe::reduce_wide([
            m(a0, b0) + m(a4, b1_19) + m(a3, b2_19) + m(a2, b3_19) + m(a1, b4_19),
            m(a1, b0) + m(a0, b1) + m(a4, b2_19) + m(a3, b3_19) + m(a2, b4_19),
            m(a2, b0) + m(a1, b1) + m(a0, b2) + m(a4, b3_19) + m(a3, b4_19),
            m(a3, b0) + m(a2, b1) + m(a1, b2) + m(a0, b3) + m(a4, b4_19),
            m(a4, b0) + m(a3, b1) + m(a2, b2) + m(a1, b3) + m(a0, b4),
        ])
    }

    #[inline(always)]
    pub fn sqr(self) -> Fe {
        let [a0, a1, a2, a3, a4] = self.0;
        let (a3_19, a4_19) = (a3 * 19, a4 * 19);
        let (d0, d1, d2) = (a0 * 2, a1 * 2, a2 * 2);
        Fe::reduce_wide([
            m(a0, a0) + m(d1, a4_19) + m(d2, a3_19),
            m(a3, a3_19) + m(d0, a1) + m(d2, a4_19),
            m(a1, a1) + m(d0, a2) + m(a3 * 2, a4_19),
            m(a4, a4_19) + m(d0, a3) + m(d1, a2),
            m(a2, a2) + m(d0, a4) + m(d1, a3),
        ])
    }

    fn pow2k(self, k: u32) -> Fe {
        (0..k).fold(self, |x, _| x.sqr())
    }

    pub fn invert(self) -> Fe {
        let z2 = self.sqr();
        let z9 = z2.pow2k(2).mul(self);
        let z11 = z9.mul(z2);
        let z5 = z11.sqr().mul(z9);
        let z10 = z5.pow2k(5).mul(z5);
        let z20 = z10.pow2k(10).mul(z10);
        let z40 = z20.pow2k(20).mul(z20);
        let z50 = z40.pow2k(10).mul(z10);
        let z100 = z50.pow2k(50).mul(z50);
        let z200 = z100.pow2k(100).mul(z100);
        let z250 = z200.pow2k(50).mul(z50);
        z250.pow2k(5).mul(z11)
    }

    /// `self^e` for a little-endian exponent; not constant time, only used on public values.
    fn pow(self, e: [u8; 32]) -> Fe {
        (0..256).rev().fold(Fe::ONE, |r, i| {
            let r = r.sqr();
            if e[i / 8] >> (i % 8) & 1 == 1 {
                r.mul(self)
            } else {
                r
            }
        })
    }

    pub fn sqrt(self) -> Option<Fe> {
        let exp = |lo, hi| {
            let mut e = [0xff; 32];
            (e[0], e[31]) = (lo, hi);
            e
        };
        let r = self.pow(exp(0xfe, 0x0f));
        let r = if r.sqr().to_bytes() == self.to_bytes() {
            r
        } else {
            r.mul(Fe([2, 0, 0, 0, 0]).pow(exp(0xfb, 0x1f)))
        };
        (r.sqr().to_bytes() == self.to_bytes()).then_some(r)
    }

    #[inline(always)]
    fn canonical(self) -> [u64; 5] {
        let mut l = self.carry().0;
        let mut q = (l[0] + 19) >> 51;
        for &x in &l[1..] {
            q = (x + q) >> 51;
        }
        l[0] += 19 * q;
        for i in 0..4 {
            l[i + 1] += l[i] >> 51;
            l[i] &= MASK;
        }
        l[4] &= MASK;
        l
    }

    /// First 8 bytes of the canonical encoding as a big-endian integer.
    #[cfg(test)]
    pub fn prefix(self) -> u64 {
        let l = self.canonical();
        (l[0] | l[1] << 51).swap_bytes()
    }

    pub fn to_bytes(self) -> [u8; 32] {
        let l = self.canonical();
        let w = [
            l[0] | l[1] << 51,
            l[1] >> 13 | l[2] << 38,
            l[2] >> 26 | l[3] << 25,
            l[3] >> 39 | l[4] << 12,
        ];
        let mut b = [0u8; 32];
        for (c, w) in b.as_chunks_mut().0.iter_mut().zip(w) {
            *c = w.to_le_bytes();
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_fe() -> Fe {
        let mut b = [0u8; 32];
        getrandom::fill(&mut b).unwrap();
        b[31] &= 0x7f;
        Fe::from_bytes(&b)
    }

    #[test]
    fn roundtrip_and_reduction() {
        let mut p = [0xffu8; 32];
        p[0] = 0xed;
        p[31] = 0x7f;
        assert_eq!(Fe::from_bytes(&p).to_bytes(), [0; 32]);
        p[0] = 0xee;
        assert_eq!(Fe::from_bytes(&p).to_bytes(), Fe::ONE.to_bytes());
        for _ in 0..100 {
            let a = rand_fe();
            let b = a.to_bytes();
            assert_eq!(Fe::from_bytes(&b).to_bytes(), b);
            assert_eq!(a.prefix(), u64::from_be_bytes(b[..8].try_into().unwrap()));
        }
    }

    #[test]
    fn arithmetic() {
        for _ in 0..100 {
            let (a, b, c) = (rand_fe(), rand_fe(), rand_fe());
            assert_eq!(a.mul(a.invert()).to_bytes(), Fe::ONE.to_bytes());
            assert_eq!(a.sqr().to_bytes(), a.mul(a).to_bytes());
            assert_eq!(
                a.add(b).mul(c).to_bytes(),
                a.mul(c).add(b.mul(c)).to_bytes()
            );
            assert_eq!(a.sub(b).add(b).to_bytes(), a.to_bytes());
        }
    }
}
