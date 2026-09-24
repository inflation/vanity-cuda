# vanity-cuda

A CUDA vanity key searcher for WireGuard. It finds x25519 keypairs whose base64 public key starts with
one of the given prefixes. It runs at about **17 G keys/s on an RTX 4080 SUPER**. Each extra prefix
character makes a match $64\times$ rarer: a 6-character prefix takes about 4 s on average, and a
7-character prefix about 4 minutes.

```
vanity-cuda [-i] [-n COUNT] PREFIX...

  -i, --ignore-case   match prefixes case-insensitively
  -n, --count COUNT   number of keypairs to find (default 1)
  PREFIX              base64 prefixes, 1..=10 characters; several may be given
```

Keypairs go to stdout, and each pair is followed by a separator line. The progress bar, speed and
estimated time go to stderr.

```
$ vanity-cuda -n 2 wg0a
private: yBgAG1jBor2uE4XhZ6XoffWFIkbRdkYjhr7m9u8HBHw=
public:  wg0aDjVHruBH+dvRlGHI3PwBzDI553QY45CxeqyHkQU=
-----------------------------------------------------
...
```

## Building

The build needs the CUDA Toolkit, with `CUDA_PATH` set, plus MSVC on Windows. `build.rs` compiles
`kernels/vanity.cu` to PTX with `nvcc -arch=sm_89`. The binary loads the CUDA driver at runtime
through `cudarc`.

```
cargo build --release
cargo test --release   # field arithmetic and walks checked against curve25519-dalek on the GPU
```

## Security model

- **Seeds:** every search lane starts from 32 bytes of OS CSPRNG output, clamped the way WireGuard
  clamps keys. Key $k$ of a lane is $s_0 + 8k$ with $k < 2^{64}$. The result is still a clamped
  scalar (this is checked) and is statistically uniform.
- **No related keys:** when a lane produces a hit, only that key is reported. The lane is then
  replaced by one with a fresh, unrelated seed. Two printed keys never share a seed, so leaking one
  key doesn't help anyone find another.
- **Secrets stay on the host:** the GPU only sees curve points. Each point is derivable from a
  public key that becomes public anyway, so the variable-time GPU code (including the inversion) has
  no secret-dependent timing. Seeds and private keys live in `Zeroizing` buffers and are only
  written to stdout.
- **Verification:** before a hit is printed, the host recomputes its public key with
  `curve25519-dalek`'s constant-time `mul_base_clamped`. A mismatch is reported as an internal
  error and the unverified key is never printed.

## Algorithm

The public key is the Montgomery $u$ coordinate of $sB$ in little-endian order, so the first base64
characters come from the low bits of $u \bmod p$.

**Center ± table walk.** Each GPU thread holds a center point $C$ in affine $(u, v)$. With
$n$ = `BATCH`, a shared table holds $P_i = (i+1)G$ for $i < n$, with $G = 8B$. Each iteration
checks the $2n + 1$ keys $C$ and $C \pm P_i$, then moves the center by $S = (2n+1)G$. On
$v^2 = u^3 + Au^2 + u$:

```math
u(C \pm P) = \lambda_\pm^2 - A - u_C - u_P, \qquad \lambda_\pm = \frac{\pm v_P - v_C}{u_P - u_C}
```

Both signs share the denominator $u_P - u_C$. All $n + 1$ denominators of a thread (including the
center step) share one Montgomery batch inversion. The work per pair $(C + P_i,\ C - P_i)$ is:

- one multiply for the forward prefix product, and two for the backward pass;
- two multiplies for $\lambda_\pm$;
- two cheap partial squares for the filter.

That makes about **2.5 multiplications per key**. The host recovers $v$ for lane starts with
Okeya–Sakurai, using a fixed sign of $G$, so every point is consistently $P$ or $-P$.

**Block-shared inversion.** Each thread $t$ ends its forward pass with one product $a_t$. Warps
compute prefix and suffix products with shuffles, and warp 0 combines the warp totals and inverts
once per block. Each thread then gets its own inverse:

```math
a_t^{-1} = \Big(\prod_{s<t} a_s\Big)\Big(\prod_{s>t} a_s\Big)\Big(\prod_s a_s\Big)^{-1}
```

The inversion itself is a variable-time safegcd (Bernstein–Yang), ported from libsecp256k1's
`modinv32` with signed 30-bit limbs.

**Field arithmetic.** Elements of $\mathbb{F}_p$, $p = 2^{255} - 19$, are 8 × 32-bit limbs kept
lazily in $[0, 2^{256})$ and reduced with $2^{256} \equiv 38$:

- Multiplication is plain 64-bit C row sums, which compile to one `IMAD.WIDE` per product.
- Squaring is a dedicated PTX carry chain with 36 products.
- Add and sub are PTX carry chains.

**Prefix filter.** Keys are tested on the low 64 bits of $u$:

- **Exact filter:** a 64-bit bitmap test on characters 0 and 1, then a binary search in the sorted
  prefix values of each mask group, then an exact canonical check.
- **Fast filter (every prefix ≥ 5 characters):** skips the full square. `sqr_low` computes the low
  64 bits of $\lambda^2$ folded with $2^{256} \equiv 38$, with a bounded error: it drops the
  carries into the high half, so the result is low by $38e$ with $e \in [0, 16]$. Together with
  the unreduced multiple of $p$, the true low bits differ from the approximation by at most $\pm 1$
  at bit 16. Characters 3–5 (bits 16..39) are tested against widened bitmaps, and only candidates
  get the exact square and check. The filter never misses a key.

**Host.**
- CPU threads keep a queue of fresh lanes, each made from a dalek scalar multiplication, so a hit
  only swaps one thread's 64-byte center on the device.
- Launches take about 15 ms each, and the GPU is idle 0.27% of the time.
- The host thread sleeps (blocking sync) instead of spinning a core while it waits.

## Optimization log

The measurements were taken on an RTX 4080 SUPER (sm_89). The card is power- and thermal-limited
(about 315 W, about 81 °C), so absolute numbers drift by several percent between runs. Each change
was compared against its predecessor in interleaved A/B runs.

| Step | Keys/s |
|---|---|
| x-only differential walk, batch inversion | 2.8 G |
| Affine center ± table walk (shared $\lambda$ denominators) | 7.6 G |
| BATCH / occupancy sweep | 10.5 G |
| `IMAD.WIDE` row multiplication, dedicated squaring | 12.9 G |
| Bitmap pre-filters | 13.4 G |
| Per-thread lanes with CPU-produced seeds | 13.6 G |
| Fast filter on the low 64 bits of $u$ (approximate `sqr_low`) | 14.0 G |
| Variable-time safegcd instead of Fermat inversion (3.5× faster inversion) | 16.8 G |
| Block-shared inversion (one per 128 threads) | +5% |

**Where it stands** (Nsight Compute):
- The integer multiply pipe (fmaheavy) is about 75–80% busy, the ALU about 50%, and the issue
  slots about 50%.
- The rest is dependency latency at 6 warps per scheduler; registers limit any higher occupancy.
- Time splits about 68% backward pass, 19% forward pass and 8% per-iteration work (scans,
  inversion, center step).
- The global-memory traffic of the batch-inversion prefix products is about 3% of runtime.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
