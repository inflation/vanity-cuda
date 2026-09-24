use crate::field::Fe;
use crate::walk::{Ctx, Seed, step_point};
use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use color_eyre::eyre::{Result, WrapErr};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vanity.ptx"));
const BATCH: usize = define(env!("VANITY_BATCH"));
const BLOCKS_PER_SM: u32 = define(env!("VANITY_BLOCKS_PER_SM")) as u32;
/// Keys each thread checks per iteration: its center and center ± i R for i in 1..=BATCH.
const KEYS: u64 = 2 * BATCH as u64 + 1;
const BLOCK: u32 = define(env!("VANITY_BLOCK")) as u32;
/// About 4096 keys per thread per launch keeps launches short and stop requests responsive.
const ITERS: u32 = 1 + 4095 / KEYS as u32;
const MAX_HITS: u32 = 1024;
const fn define(s: &str) -> usize {
    match usize::from_str_radix(s, 10) {
        Ok(n) => n,
        Err(_) => panic!("invalid kernel define"),
    }
}

const A: Fe = Fe([486662, 0, 0, 0, 0]);

/// Affine point (u, v) on v^2 = u^3 + A u^2 + u.
type Affine = (Fe, Fe);

/// The step point G = ±8B with a fixed sign of v. Centers and the table are recovered relative
/// to G, so every walk is either all of (s0 + 8j)B or all of their negations, which share u.
#[derive(Clone, Copy)]
struct Orientation {
    g: Affine,
    inv_2v: Fe,
}

impl Orientation {
    fn new() -> Orientation {
        let u = step_point(1);
        let v = u
            .sqr()
            .mul(u)
            .add(A.mul(u.sqr()))
            .add(u)
            .sqrt()
            .expect("8B is on the curve");
        Orientation {
            g: (u, v),
            inv_2v: v.add(v).invert(),
        }
    }

    /// v of the point P with u(P) = u1 and u(P + G) = u2 (Okeya–Sakurai recovery).
    fn recover(&self, u1: Fe, u2: Fe) -> Fe {
        let (ug, two_a) = (self.g.0, A.add(A));
        let t = u1.mul(ug).add(Fe::ONE).mul(u1.add(ug).add(two_a));
        t.sub(two_a).sub(u1.sub(ug).sqr().mul(u2)).mul(self.inv_2v)
    }

    /// The per-iteration step `n * G`.
    fn step(&self, n: u64) -> Affine {
        let u = step_point(n);
        (u, self.recover(u, step_point(n + 1)))
    }
}

fn words(f: Fe) -> [u32; 8] {
    let b = f.to_bytes();
    std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// Entry i < BATCH is (i + 1) G and entry BATCH is the center step KEYS * G.
fn build_table(o: Orientation) -> Vec<u32> {
    let steps = (1..=BATCH as u64).chain([KEYS]);
    let pts = steps.map(|i| o.step(i));
    pts.flat_map(|(u, v)| [words(u), words(v)]).flatten().collect()
}

/// A GPU thread's own seed and its first center ±(s0 + 8 BATCH)B as words of u then v.
/// Key j of iteration k is s0 + 8 (k KEYS + j).
struct Lane {
    seed: Seed,
    center: [u32; 16],
}

impl Lane {
    fn new(o: Orientation) -> Lane {
        let seed = Seed::random();
        let (u1, u2) = (seed.u(BATCH as u64), seed.u(BATCH as u64 + 1));
        let center = [words(u1), words(o.recover(u1, u2))].concat();
        Lane {
            seed,
            center: center.try_into().unwrap(),
        }
    }
}

struct Gpu {
    stream: Arc<CudaStream>,
    walk: CudaFunction,
    walk_fast: CudaFunction,
    blocks: u32,
}

impl Gpu {
    fn new() -> Result<Gpu> {
        let cu = CudaContext::new(0)?;
        // Sleep instead of spinning a core while waiting for each launch.
        cu.set_blocking_synchronize()?;
        let module = cu.load_module(Ptx::from_src(PTX))?;
        let sms = cu.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        Ok(Gpu {
            stream: cu.default_stream(),
            walk: module.load_function("walk")?,
            walk_fast: module.load_function("walk_fast")?,
            blocks: sms * BLOCKS_PER_SM,
        })
    }

    fn threads(&self) -> usize {
        (self.blocks * BLOCK) as usize
    }
}

/// CPU cores keep a pool of fresh lanes, so a hit only swaps one thread's center.
pub fn run(ctx: &Ctx) -> Result<()> {
    let o = Orientation::new();
    let (tx, rx) = sync_channel(4096);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(1, |n| n.get()) {
            let tx = tx.clone();
            s.spawn(move || while tx.send(Lane::new(o)).is_ok() {});
        }
        drop(tx);
        search(ctx, o, rx)
    })
}

fn search(ctx: &Ctx, o: Orientation, lanes: Receiver<Lane>) -> Result<()> {
    let gpu = Gpu::new().wrap_err("failed to initialize CUDA")?;
    let stream = &gpu.stream;
    let threads = gpu.threads();
    let table = stream.clone_htod(&build_table(o))?;
    let groups = &ctx.matcher.groups;
    let masks = stream.clone_htod(&groups.iter().map(|g| g.0).collect::<Vec<_>>())?;
    let values: Vec<u64> = groups.iter().flat_map(|g| g.1.iter().copied()).collect();
    let starts: Vec<u32> = std::iter::once(0)
        .chain(groups.iter().scan(0, |n, g| {
            *n += g.1.len() as u32;
            Some(*n)
        }))
        .collect();
    let (values, starts) = (stream.clone_htod(&values)?, stream.clone_htod(&starts)?);
    let (n_groups, max_hits, iters) = (groups.len() as u32, MAX_HITS, ITERS);
    let chars = stream.clone_htod(&ctx.matcher.char_sets())?;
    // The fast kernel filters on characters 3..5, so every pattern must have them.
    let walk = if ctx.matcher.min_len() >= 5 { &gpu.walk_fast } else { &gpu.walk };
    let mut hits = stream.alloc_zeros::<u32>(1 + 5 * MAX_HITS as usize)?;
    let cfg = LaunchConfig {
        grid_dim: (gpu.blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut owned: Vec<Lane> = lanes.iter().take(threads).collect();
    let centers: Vec<u32> = owned.iter().flat_map(|l| l.center).collect();
    let mut state = stream.clone_htod(&centers)?;
    // Iteration at which each thread's current lane started.
    let mut since = vec![0u64; threads];
    for launch in 0u64.. {
        if ctx.done() {
            break;
        }
        let mut b = stream.launch_builder(walk);
        b.arg(&mut state)
            .arg(&table)
            .arg(&masks)
            .arg(&starts)
            .arg(&values)
            .arg(&n_groups)
            .arg(&chars);
        b.arg(&mut hits).arg(&max_hits).arg(&iters);
        unsafe { b.launch(cfg) }?;
        let h = stream.clone_dtoh(&hits)?;
        ctx.count(threads as u64 * KEYS * ITERS as u64);
        if h[0] == 0 {
            continue;
        }
        // Report at most one key per lane, then give that thread a fresh, unrelated lane.
        let next = (launch + 1) * ITERS as u64;
        let records = h[1..].as_chunks().0.iter().take(h[0].min(MAX_HITS) as usize);
        for &[tid, j, it, hi, lo] in records {
            let t = tid as usize;
            if since[t] == next {
                continue;
            }
            let k = launch * ITERS as u64 + it as u64 - since[t];
            ctx.report(&owned[t].seed, k * KEYS + j as u64, (hi as u64) << 32 | lo as u64)?;
            owned[t] = lanes.recv()?;
            stream.memcpy_htod(&owned[t].center, &mut state.slice_mut(16 * t..16 * t + 16))?;
            since[t] = next;
        }
        stream.memset_zeros(&mut hits)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use curve25519_dalek::MontgomeryPoint;

    fn from_words(w: &[u32]) -> Fe {
        let mut b = [0u8; 32];
        for (c, w) in b.as_chunks_mut::<4>().0.iter_mut().zip(w) {
            *c = w.to_le_bytes();
        }
        let top = (b[31] >> 7) as u64;
        Fe::from_bytes(&b).add(Fe([19 * top, 0, 0, 0, 0]))
    }

    fn on_curve((u, v): Affine) -> bool {
        v.sqr().to_bytes() == u.sqr().mul(u).add(A.mul(u.sqr())).add(u).to_bytes()
    }

    fn u_of(seed: &Seed, off: u64) -> [u8; 32] {
        MontgomeryPoint::mul_base_clamped(*seed.key(off).unwrap()).to_bytes()
    }

    #[test]
    fn affine_step_matches_dalek() {
        let o = Orientation::new();
        assert!(on_curve(o.g));
        for n in [1, 40, KEYS] {
            let q = o.step(n);
            assert!(on_curve(q));
            let lane = Lane::new(o);
            let (u, v) = (from_words(&lane.center[..8]), from_words(&lane.center[8..]));
            assert_eq!(u.to_bytes(), u_of(&lane.seed, BATCH as u64));
            assert!(on_curve((u, v)));
            let lambda = q.1.sub(v).mul(q.0.sub(u).invert());
            let u3 = lambda.sqr().sub(A).sub(u).sub(q.0);
            assert_eq!(u3.to_bytes(), u_of(&lane.seed, BATCH as u64 + n));
        }
    }

    #[test]
    fn field_matches_scalar() {
        let gpu = Gpu::new().unwrap();
        let module = gpu
            .stream
            .context()
            .load_module(Ptx::from_src(PTX))
            .unwrap();
        let f = module.load_function("field_test").unwrap();
        let n = 1024u32;
        let mut bytes = vec![0u8; 64 * n as usize];
        getrandom::fill(&mut bytes).unwrap();
        let mut w: Vec<u32> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        w[..8].copy_from_slice(&[0xffffffed, !0, !0, !0, !0, !0, !0, 0x7fffffff]);
        w[8..16].fill(!0);
        w[16..24].copy_from_slice(&[0xffffffee, !0, !0, !0, !0, !0, !0, 0x7fffffff]);
        w[24..32].fill(!0);
        w[32..40].fill(0);
        let b0 = 8 * n as usize;
        w[b0 + 8..b0 + 16].fill(!0);
        w[b0 + 32..b0 + 40].copy_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]);
        let (a, b) = w.split_at(8 * n as usize);
        let (da, db) = (
            gpu.stream.clone_htod(a).unwrap(),
            gpu.stream.clone_htod(b).unwrap(),
        );
        let mut out = gpu.stream.alloc_zeros::<u32>(44 * n as usize).unwrap();
        let mut l = gpu.stream.launch_builder(&f);
        l.arg(&da).arg(&db).arg(&mut out).arg(&n);
        let cfg = LaunchConfig {
            grid_dim: (n / BLOCK, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { l.launch(cfg) }.unwrap();
        let out = gpu.stream.clone_dtoh(&out).unwrap();
        for i in 0..n as usize {
            let (x, y) = (
                from_words(&a[8 * i..8 * i + 8]),
                from_words(&b[8 * i..8 * i + 8]),
            );
            let o = &out[44 * i..44 * i + 44];
            assert_eq!(
                from_words(&o[..8]).to_bytes(),
                x.mul(y).to_bytes(),
                "mul {i}"
            );
            assert_eq!(
                from_words(&o[8..16]).to_bytes(),
                x.add(y).to_bytes(),
                "add {i}"
            );
            assert_eq!(
                from_words(&o[16..24]).to_bytes(),
                x.sub(y).to_bytes(),
                "sub {i}"
            );
            if i != 0 {
                assert_eq!(
                    from_words(&o[24..32]).to_bytes(),
                    x.invert().to_bytes(),
                    "invert {i}"
                );
            }
            assert_eq!(
                from_words(&o[32..40]).to_bytes(),
                x.sqr().to_bytes(),
                "sqr {i}"
            );
            assert_eq!(
                (o[40] as u64) << 32 | o[41] as u64,
                x.prefix(),
                "prefix {i}"
            );
            let approx = (o[43] as u64) << 32 | o[42] as u64;
            let e = sqr_low(&a[8 * i..8 * i + 8]).wrapping_sub(approx);
            assert!(e.is_multiple_of(38) && e / 38 <= 16, "sqr_low {i}: off by {e}");
        }
    }

    /// Low 64 bits of lo + 38 hi for the 512-bit square lo + 2^256 hi of the words `w`.
    fn sqr_low(w: &[u32]) -> u64 {
        let mut x = [0u64; 16];
        for i in 0..8 {
            let mut c = 0u64;
            for j in 0..8 {
                let t = w[i] as u64 * w[j] as u64 + x[i + j] + c;
                x[i + j] = t & 0xffffffff;
                c = t >> 32;
            }
            x[i + 8] = c;
        }
        (x[1] << 32 | x[0]).wrapping_add(38u64.wrapping_mul(x[9] << 32 | x[8]))
    }

    /// Runs `iters` iterations on 256 threads and checks every hit and the final centers with dalek.
    fn walk_matches_dalek(fast: bool, m: crate::pattern::Matcher, iters: u32, min_hits: u32) {
        let gpu = Gpu::new().unwrap();
        let stream = &gpu.stream;
        let (blocks, threads) = (2u32, 2 * BLOCK as usize);
        let o = Orientation::new();
        let lanes: Vec<Lane> = (0..threads).map(|_| Lane::new(o)).collect();
        let centers: Vec<u32> = lanes.iter().flat_map(|l| l.center).collect();
        let mut state = stream.clone_htod(&centers).unwrap();
        let table = stream.clone_htod(&build_table(o)).unwrap();
        let masks = stream
            .clone_htod(&m.groups.iter().map(|g| g.0).collect::<Vec<_>>())
            .unwrap();
        let values: Vec<u64> = m.groups.iter().flat_map(|g| g.1.clone()).collect();
        let starts: Vec<u32> = std::iter::once(0)
            .chain(m.groups.iter().scan(0, |n, g| {
                *n += g.1.len() as u32;
                Some(*n)
            }))
            .collect();
        let starts = stream.clone_htod(&starts).unwrap();
        let values = stream.clone_htod(&values).unwrap();
        let chars = stream.clone_htod(&m.char_sets()).unwrap();
        let groups = m.groups.len() as u32;
        let max_hits = 4 * min_hits + 64;
        let mut hits = stream.alloc_zeros::<u32>(1 + 5 * max_hits as usize).unwrap();
        let mut b = stream.launch_builder(if fast { &gpu.walk_fast } else { &gpu.walk });
        b.arg(&mut state)
            .arg(&table)
            .arg(&masks)
            .arg(&starts)
            .arg(&values)
            .arg(&groups)
            .arg(&chars);
        b.arg(&mut hits).arg(&max_hits).arg(&iters);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.unwrap();

        let h = stream.clone_dtoh(&hits).unwrap();
        assert!(h[0] >= min_hits && h[0] <= max_hits, "hits: {}", h[0]);
        let mut sides = [false; 2];
        for r in h[1..].as_chunks::<5>().0.iter().take(h[0] as usize) {
            let seed = &lanes[r[0] as usize].seed;
            let off = r[2] as u64 * KEYS + r[1] as u64;
            let prefix = u64::from_be_bytes(u_of(seed, off)[..8].try_into().unwrap());
            assert_eq!((r[3] as u64) << 32 | r[4] as u64, prefix);
            assert!(m.matches(prefix));
            sides[(r[1] as usize > BATCH) as usize] = true;
        }
        assert_eq!(sides, [true; 2]);

        let s = stream.clone_dtoh(&state).unwrap();
        let get = |c: usize, tid: usize| from_words(&s[tid * 16 + c * 8..tid * 16 + c * 8 + 8]);
        for tid in [0, 17, threads - 1] {
            let p = (get(0, tid), get(1, tid));
            let center = iters as u64 * KEYS + BATCH as u64;
            assert_eq!(p.0.to_bytes(), u_of(&lanes[tid].seed, center));
            assert!(on_curve(p));
        }
    }

    #[test]
    fn walk_exact() {
        let m = crate::pattern::Matcher::new(&["A".into(), "wg".into()], false).unwrap();
        let expect = 2 * BLOCK * KEYS as u32 * 3 / 64;
        walk_matches_dalek(false, m, 3, expect * 3 / 4);
    }

    /// Prefixes "???AA" and "???gA": 2^19 patterns of 30 bits, so hits are frequent while the fast
    /// filter on characters 3..5 stays selective. 'A' (0) takes the wrap-around path and 'g' (32)
    /// the single evaluation.
    #[test]
    fn walk_fast() {
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".as_bytes();
        let prefixes: Vec<String> = (0..1 << 18)
            .flat_map(|n: usize| {
                let c = |k: usize| alphabet[n >> (6 * k) & 63] as char;
                ["AA", "gA"].map(|t| format!("{}{}{}{t}", c(0), c(1), c(2)))
            })
            .collect();
        let m = crate::pattern::Matcher::new(&prefixes, false).unwrap();
        assert_eq!(m.min_len(), 5);
        let iters = 4;
        let expect = 2 * BLOCK * KEYS as u32 * iters / 2048;
        walk_matches_dalek(true, m, iters, expect * 3 / 4);
    }
}
