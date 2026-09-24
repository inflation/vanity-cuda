use color_eyre::eyre::{Result, bail, ensure};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
pub const MAX_LEN: usize = 10;

/// Prefix patterns over the first 64 bits of a public key, grouped by mask.
#[derive(Clone, Debug)]
pub struct Matcher {
    pub groups: Vec<(u64, Vec<u64>)>,
}

fn encode(prefix: &[u8]) -> (u64, u64) {
    prefix
        .iter()
        .enumerate()
        .fold((0, 0), |(mask, value), (i, &c)| {
            let v = ALPHABET.iter().position(|&a| a == c).unwrap() as u64;
            let shift = 58 - 6 * i;
            (mask | 0x3f << shift, value | v << shift)
        })
}

fn variants(prefix: &str, ignore_case: bool) -> Vec<Vec<u8>> {
    prefix.bytes().fold(vec![vec![]], |acc, c| {
        let cases = if ignore_case && c.is_ascii_alphabetic() {
            vec![c.to_ascii_lowercase(), c.to_ascii_uppercase()]
        } else {
            vec![c]
        };
        acc.iter()
            .flat_map(|p| cases.iter().map(move |&c| [p.as_slice(), &[c]].concat()))
            .collect()
    })
}

impl Matcher {
    pub fn new(prefixes: &[String], ignore_case: bool) -> Result<Matcher> {
        let mut groups: Vec<(u64, Vec<u64>)> = vec![];
        for p in prefixes {
            ensure!(
                (1..=MAX_LEN).contains(&p.len()),
                "prefix {p:?} must be 1..={MAX_LEN} characters"
            );
            if let Some(c) = p.bytes().find(|c| !ALPHABET.contains(c)) {
                bail!("prefix {p:?} has non-base64 character {:?}", c as char);
            }
            for v in variants(p, ignore_case) {
                let (mask, value) = encode(&v);
                match groups.iter_mut().find(|g| g.0 == mask) {
                    Some(g) => g.1.push(value),
                    None => groups.push((mask, vec![value])),
                }
            }
        }
        for g in &mut groups {
            g.1.sort_unstable();
            g.1.dedup();
        }
        Ok(Matcher { groups })
    }

    #[inline(always)]
    pub fn matches(&self, x: u64) -> bool {
        self.groups
            .iter()
            .any(|(m, v)| v.binary_search(&(x & m)).is_ok())
    }

    /// Bit `c` of entry `i` is set if some pattern allows base64 character `c` at position `i`.
    pub fn char_sets(&self) -> [u64; 6] {
        std::array::from_fn(|i| {
            let shift = 58 - 6 * i;
            let chars = |(m, v): &(u64, Vec<u64>)| match m >> shift & 0x3f {
                0 => !0,
                _ => v.iter().fold(0, |bits, v| bits | 1 << (v >> shift & 0x3f)),
            };
            self.groups.iter().map(chars).fold(0, |a, b| a | b)
        })
    }

    /// Length of the shortest pattern.
    pub fn min_len(&self) -> u32 {
        self.groups.iter().map(|g| g.0.count_ones() / 6).min().unwrap_or(0)
    }

    /// Probability that a random key matches.
    pub fn probability(&self) -> f64 {
        self.groups
            .iter()
            .map(|(m, v)| v.len() as f64 / 2f64.powi(m.count_ones() as i32))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};

    fn key_prefix(bytes: &[u8; 32]) -> u64 {
        u64::from_be_bytes(bytes[..8].try_into().unwrap())
    }

    #[test]
    fn matches_base64() {
        let mut key = [0u8; 32];
        getrandom::fill(&mut key).unwrap();
        let b64 = STANDARD.encode(key);
        for n in 1..=MAX_LEN {
            let m = Matcher::new(&[b64[..n].to_string()], false).unwrap();
            assert!(m.matches(key_prefix(&key)));
        }
        let m = Matcher::new(&[b64[..6].to_lowercase()], true).unwrap();
        assert!(m.matches(key_prefix(&key)));
    }

    #[test]
    fn case_expansion() {
        let m = Matcher::new(&["ab1".into(), "AB1".into()], true).unwrap();
        assert_eq!(m.groups.len(), 1);
        assert_eq!(m.groups[0].1.len(), 4);
        assert_eq!(m.char_sets()[..3], [1 << 0 | 1 << 26, 1 << 1 | 1 << 27, 1 << 53]);
        assert_eq!(m.min_len(), 3);
        let m = Matcher::new(&["b".into(), "Ab".into()], false).unwrap();
        assert_eq!(m.char_sets()[..2], [1 << 0 | 1 << 27, !0]);
        assert_eq!(m.min_len(), 1);
        assert!(Matcher::new(&["a-b".into()], false).is_err());
        assert!(Matcher::new(&["abcdefghijk".into()], false).is_err());
    }
}
