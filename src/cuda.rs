use crate::field::Fe;
use crate::walk::{Ctx, Point, Seed, chain, step_point};
use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::error::Error;
use std::sync::Arc;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/vanity.ptx"));
const LANES: usize = match usize::from_str_radix(env!("VANITY_LANES"), 10) {
    Ok(n) => n,
    Err(_) => panic!("invalid VANITY_LANES"),
};
const BLOCK: u32 = 128;
/// Must match `__launch_bounds__` of `walk`; see the comment there.
const BLOCKS_PER_SM: u32 = 1;
const ITERS: u32 = 32;
const MAX_HITS: u32 = 16;
const A: Fe = Fe([486662, 0, 0, 0, 0]);

/// Affine point (u, v) on v^2 = u^3 + A u^2 + u.
type Affine = (Fe, Fe);

/// The step point G = ±8B with a fixed sign of v. Lanes and Q are recovered relative to G,
/// so the whole walk is either all of (s0 + 8j)B or all of their negations, which share u.
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

fn normalize(pts: &[Point]) -> Vec<Fe> {
    let mut acc = Vec::with_capacity(pts.len());
    let mut a = Fe::ONE;
    for p in pts {
        acc.push(a);
        a = a.mul(p.1);
    }
    let mut inv = a.invert();
    let mut u = vec![Fe::ONE; pts.len()];
    for i in (0..pts.len()).rev() {
        u[i] = pts[i].0.mul(inv.mul(acc[i]));
        inv = inv.mul(pts[i].1);
    }
    u
}

/// Affine ±(s0 + 8j)B for `j in 0..n`, split across all cores.
fn affine_lanes(seed: &Seed, o: Orientation, n: usize) -> Vec<Affine> {
    let chunk = n.div_ceil(std::thread::available_parallelism().map_or(1, |p| p.get()));
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                s.spawn(move || {
                    let len = chunk.min(n - start);
                    let u = normalize(&chain(seed, start, len + 1));
                    (0..len)
                        .map(|i| (u[i], o.recover(u[i], u[i + 1])))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap()).collect()
    })
}

fn words(f: Fe) -> [u32; 8] {
    let b = f.to_bytes();
    std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// Lane `l * threads + tid` starts at ±(s0 + 8j)B, stored as word ((l * 2 + coord) * 8 + limb) * threads + tid.
fn build_state(seed: &Seed, o: Orientation, threads: usize) -> Vec<u32> {
    let lanes = LANES * threads;
    let pts = affine_lanes(seed, o, lanes);
    let mut s = vec![0u32; lanes * 16];
    for l in 0..LANES {
        for tid in 0..threads {
            let (u, v) = pts[l * threads + tid];
            for (c, f) in [u, v].into_iter().enumerate() {
                for (i, w) in words(f).into_iter().enumerate() {
                    s[((l * 2 + c) * 8 + i) * threads + tid] = w;
                }
            }
        }
    }
    s
}

struct Gpu {
    stream: Arc<CudaStream>,
    walk: CudaFunction,
    blocks: u32,
}

impl Gpu {
    fn new() -> Result<Gpu, Box<dyn Error>> {
        let cu = CudaContext::new(0)?;
        let module = cu.load_module(Ptx::from_src(PTX))?;
        let sms = cu.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        Ok(Gpu {
            stream: cu.default_stream(),
            walk: module.load_function("walk")?,
            blocks: sms * BLOCKS_PER_SM,
        })
    }

    fn threads(&self) -> usize {
        (self.blocks * BLOCK) as usize
    }
}

pub fn worker(ctx: &Ctx) {
    if let Err(e) = run(ctx) {
        eprintln!("\nCUDA error: {e}");
        std::process::exit(1);
    }
}

fn run(ctx: &Ctx) -> Result<(), Box<dyn Error>> {
    let gpu = Gpu::new()?;
    let stream = &gpu.stream;
    let threads = gpu.threads();
    let lanes = (LANES * threads) as u64;
    let o = Orientation::new();
    let (qu, qv) = o.step(lanes);
    let q = stream.clone_htod(&[words(qu), words(qv)].concat())?;
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
    let mut hits = stream.alloc_zeros::<u32>(1 + 5 * MAX_HITS as usize)?;
    let cfg = LaunchConfig {
        grid_dim: (gpu.blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let prepare = || {
        std::thread::spawn(move || {
            let seed = Seed::random();
            let state = build_state(&seed, o, threads);
            (seed, state)
        })
    };
    let mut next = prepare();
    while !ctx.done() {
        let (seed, state) = next.join().expect("state builder panicked");
        next = prepare();
        let mut state = stream.clone_htod(&state)?;
        for launch in 0u64.. {
            if ctx.done() {
                return Ok(());
            }
            let mut b = stream.launch_builder(&gpu.walk);
            b.arg(&mut state)
                .arg(&q)
                .arg(&masks)
                .arg(&starts)
                .arg(&values)
                .arg(&n_groups);
            b.arg(&mut hits).arg(&max_hits).arg(&iters);
            unsafe { b.launch(cfg) }?;
            let h = stream.clone_dtoh(&hits)?;
            ctx.count(lanes * ITERS as u64);
            if h[0] > 0 {
                let [tid, l, it, hi, lo] = h[1..6].try_into().unwrap();
                let j = l as u64 * threads as u64 + tid as u64;
                let k = launch * ITERS as u64 + it as u64 + 1;
                ctx.report(&seed, j + lanes * k, (hi as u64) << 32 | lo as u64);
                stream.memset_zeros(&mut hits)?;
                break;
            }
        }
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
        let (o, seed, n) = (Orientation::new(), Seed::random(), 40u64);
        let pts = affine_lanes(&seed, o, 100);
        let q = o.step(n);
        assert!(on_curve(o.g) && on_curve(q));
        for (j, &(u, v)) in pts.iter().enumerate() {
            assert_eq!(u.to_bytes(), u_of(&seed, j as u64));
            assert!(on_curve((u, v)));
            let lambda = q.1.sub(v).mul(q.0.sub(u).invert());
            let u3 = lambda.sqr().sub(A).sub(u).sub(q.0);
            assert_eq!(u3.to_bytes(), u_of(&seed, j as u64 + n));
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
        let (a, b) = w.split_at(8 * n as usize);
        let (da, db) = (
            gpu.stream.clone_htod(a).unwrap(),
            gpu.stream.clone_htod(b).unwrap(),
        );
        let mut out = gpu.stream.alloc_zeros::<u32>(34 * n as usize).unwrap();
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
            let o = &out[34 * i..34 * i + 34];
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
                (o[32] as u64) << 32 | o[33] as u64,
                x.prefix(),
                "prefix {i}"
            );
        }
    }

    #[test]
    fn walk_matches_dalek() {
        let gpu = Gpu::new().unwrap();
        let stream = &gpu.stream;
        let (blocks, threads) = (2u32, 2 * BLOCK as usize);
        let lanes = (LANES * threads) as u64;
        let (o, seed) = (Orientation::new(), Seed::random());
        let mut state = stream.clone_htod(&build_state(&seed, o, threads)).unwrap();
        let (qu, qv) = o.step(lanes);
        let q = stream.clone_htod(&[words(qu), words(qv)].concat()).unwrap();
        let m = crate::pattern::Matcher::new(&["A".into(), "wg".into()], false).unwrap();
        let masks = stream
            .clone_htod(&m.groups.iter().map(|g| g.0).collect::<Vec<_>>())
            .unwrap();
        let values: Vec<u64> = m.groups.iter().flat_map(|g| g.1.clone()).collect();
        let starts = stream
            .clone_htod(&[0, m.groups[0].1.len() as u32, values.len() as u32])
            .unwrap();
        let values = stream.clone_htod(&values).unwrap();
        let mut hits = stream
            .alloc_zeros::<u32>(1 + 5 * MAX_HITS as usize)
            .unwrap();
        let (groups, max_hits, iters) = (2u32, MAX_HITS, 3u32);
        let mut b = stream.launch_builder(&gpu.walk);
        b.arg(&mut state)
            .arg(&q)
            .arg(&masks)
            .arg(&starts)
            .arg(&values)
            .arg(&groups);
        b.arg(&mut hits).arg(&max_hits).arg(&iters);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.unwrap();

        let h = stream.clone_dtoh(&hits).unwrap();
        assert!(h[0] as u64 > lanes * 3 / 128, "too few hits: {}", h[0]);
        for r in h[1..]
            .as_chunks::<5>()
            .0
            .iter()
            .take(h[0].min(MAX_HITS) as usize)
        {
            let off = r[1] as u64 * threads as u64 + r[0] as u64 + lanes * (r[2] as u64 + 1);
            let prefix = u64::from_be_bytes(u_of(&seed, off)[..8].try_into().unwrap());
            assert_eq!((r[3] as u64) << 32 | r[4] as u64, prefix);
            assert!(m.matches(prefix));
        }

        let s = stream.clone_dtoh(&state).unwrap();
        let get = |l: usize, c: usize, tid: usize| {
            let w: Vec<u32> = (0..8)
                .map(|i| s[((l * 2 + c) * 8 + i) * threads + tid])
                .collect();
            from_words(&w)
        };
        for (l, tid) in [(0, 0), (5, 17), (LANES - 1, threads - 1)] {
            let p = (get(l, 0, tid), get(l, 1, tid));
            assert_eq!(
                p.0.to_bytes(),
                u_of(&seed, (l * threads + tid) as u64 + lanes * 3)
            );
            assert!(on_curve(p));
        }
    }
}
