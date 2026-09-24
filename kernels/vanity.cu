// GF(2^255 - 19) in 8 x 32-bit limbs, values kept in [0, 2^256) and reduced via 2^256 = 38.
#include <stdint.h>

// Lanes per thread; build.rs reads this value for src/cuda.rs.
#define LANES 64

struct fe {
    uint32_t v[8];
};

__device__ __forceinline__ fe fe_one() {
    fe r = {{1, 0, 0, 0, 0, 0, 0, 0}};
    return r;
}

// r += 38 * c for small c (< 2^16). A carry out leaves r < 38c, so the second fold cannot carry.
__device__ __forceinline__ void fold(fe &r, uint32_t c) {
    uint32_t c2;
    asm("mad.lo.cc.u32 %0, %9, 38, %0;\n\t"
        "madc.hi.cc.u32 %1, %9, 38, %1;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32 %8, 0, 0;"
        : "+r"(r.v[0]), "+r"(r.v[1]), "+r"(r.v[2]), "+r"(r.v[3]), "+r"(r.v[4]), "+r"(r.v[5]),
          "+r"(r.v[6]), "+r"(r.v[7]), "=r"(c2)
        : "r"(c));
    r.v[0] += 38 * c2;
}

__device__ __forceinline__ fe fe_add(fe a, const fe &b) {
    uint32_t c;
    asm("add.cc.u32 %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, %11;\n\t"
        "addc.cc.u32 %3, %3, %12;\n\t"
        "addc.cc.u32 %4, %4, %13;\n\t"
        "addc.cc.u32 %5, %5, %14;\n\t"
        "addc.cc.u32 %6, %6, %15;\n\t"
        "addc.cc.u32 %7, %7, %16;\n\t"
        "addc.u32 %8, 0, 0;"
        : "+r"(a.v[0]), "+r"(a.v[1]), "+r"(a.v[2]), "+r"(a.v[3]), "+r"(a.v[4]), "+r"(a.v[5]),
          "+r"(a.v[6]), "+r"(a.v[7]), "=r"(c)
        : "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]),
          "r"(b.v[6]), "r"(b.v[7]));
    fold(a, c);
    return a;
}

// a - b; a borrow means the result wrapped by 2^256 = 38, so subtract 38 (twice at most).
__device__ __forceinline__ fe fe_sub(fe a, const fe &b) {
    uint32_t c = 0;
    asm("sub.cc.u32 %0, %0, %9;\n\t"
        "subc.cc.u32 %1, %1, %10;\n\t"
        "subc.cc.u32 %2, %2, %11;\n\t"
        "subc.cc.u32 %3, %3, %12;\n\t"
        "subc.cc.u32 %4, %4, %13;\n\t"
        "subc.cc.u32 %5, %5, %14;\n\t"
        "subc.cc.u32 %6, %6, %15;\n\t"
        "subc.cc.u32 %7, %7, %16;\n\t"
        "subc.u32 %8, 0, 0;\n\t"
        "and.b32 %8, %8, 38;\n\t"
        "sub.cc.u32 %0, %0, %8;\n\t"
        "subc.cc.u32 %1, %1, 0;\n\t"
        "subc.cc.u32 %2, %2, 0;\n\t"
        "subc.cc.u32 %3, %3, 0;\n\t"
        "subc.cc.u32 %4, %4, 0;\n\t"
        "subc.cc.u32 %5, %5, 0;\n\t"
        "subc.cc.u32 %6, %6, 0;\n\t"
        "subc.cc.u32 %7, %7, 0;\n\t"
        "subc.u32 %8, 0, 0;\n\t"
        "and.b32 %8, %8, 38;\n\t"
        "sub.u32 %0, %0, %8;"
        : "+r"(a.v[0]), "+r"(a.v[1]), "+r"(a.v[2]), "+r"(a.v[3]), "+r"(a.v[4]), "+r"(a.v[5]),
          "+r"(a.v[6]), "+r"(a.v[7]), "+r"(c)
        : "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]), "r"(b.v[5]),
          "r"(b.v[6]), "r"(b.v[7]));
    return a;
}

// r[i..i+8] += a * b, where r[i+8] is still unwritten.
#define ROW(i)                                                                                     \
    asm("mad.lo.cc.u32 %0, %9, %10, %0;\n\t"                                                       \
        "madc.lo.cc.u32 %1, %9, %11, %1;\n\t"                                                      \
        "madc.lo.cc.u32 %2, %9, %12, %2;\n\t"                                                      \
        "madc.lo.cc.u32 %3, %9, %13, %3;\n\t"                                                      \
        "madc.lo.cc.u32 %4, %9, %14, %4;\n\t"                                                      \
        "madc.lo.cc.u32 %5, %9, %15, %5;\n\t"                                                      \
        "madc.lo.cc.u32 %6, %9, %16, %6;\n\t"                                                      \
        "madc.lo.cc.u32 %7, %9, %17, %7;\n\t"                                                      \
        "addc.u32 %8, 0, 0;\n\t"                                                                   \
        "mad.hi.cc.u32 %1, %9, %10, %1;\n\t"                                                       \
        "madc.hi.cc.u32 %2, %9, %11, %2;\n\t"                                                      \
        "madc.hi.cc.u32 %3, %9, %12, %3;\n\t"                                                      \
        "madc.hi.cc.u32 %4, %9, %13, %4;\n\t"                                                      \
        "madc.hi.cc.u32 %5, %9, %14, %5;\n\t"                                                      \
        "madc.hi.cc.u32 %6, %9, %15, %6;\n\t"                                                      \
        "madc.hi.cc.u32 %7, %9, %16, %7;\n\t"                                                      \
        "madc.hi.u32 %8, %9, %17, %8;"                                                             \
        : "+r"(r[i]), "+r"(r[i + 1]), "+r"(r[i + 2]), "+r"(r[i + 3]), "+r"(r[i + 4]),              \
          "+r"(r[i + 5]), "+r"(r[i + 6]), "+r"(r[i + 7]), "+r"(r[i + 8])                           \
        : "r"(a.v[i]), "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]), "r"(b.v[4]),           \
          "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]))

__device__ __forceinline__ fe fe_mul(const fe &a, const fe &b) {
    uint32_t r[16] = {0, 0, 0, 0, 0, 0, 0, 0};
    ROW(0);
    ROW(1);
    ROW(2);
    ROW(3);
    ROW(4);
    ROW(5);
    ROW(6);
    ROW(7);
    fe l;
    uint32_t c = 0;
    asm("mad.lo.cc.u32 %0, %9, 38, %0;\n\t"
        "madc.lo.cc.u32 %1, %10, 38, %1;\n\t"
        "madc.lo.cc.u32 %2, %11, 38, %2;\n\t"
        "madc.lo.cc.u32 %3, %12, 38, %3;\n\t"
        "madc.lo.cc.u32 %4, %13, 38, %4;\n\t"
        "madc.lo.cc.u32 %5, %14, 38, %5;\n\t"
        "madc.lo.cc.u32 %6, %15, 38, %6;\n\t"
        "madc.lo.cc.u32 %7, %16, 38, %7;\n\t"
        "addc.u32 %8, 0, 0;\n\t"
        "mad.hi.cc.u32 %1, %9, 38, %1;\n\t"
        "madc.hi.cc.u32 %2, %10, 38, %2;\n\t"
        "madc.hi.cc.u32 %3, %11, 38, %3;\n\t"
        "madc.hi.cc.u32 %4, %12, 38, %4;\n\t"
        "madc.hi.cc.u32 %5, %13, 38, %5;\n\t"
        "madc.hi.cc.u32 %6, %14, 38, %6;\n\t"
        "madc.hi.cc.u32 %7, %15, 38, %7;\n\t"
        "madc.hi.u32 %8, %16, 38, %8;"
        : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]),
          "+r"(r[7]), "+r"(c)
        : "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]),
          "r"(r[15]));
    for (int i = 0; i < 8; i++) l.v[i] = r[i];
    fold(l, c);
    return l;
}

__device__ __forceinline__ fe fe_sqr(const fe &a) { return fe_mul(a, a); }

__device__ __noinline__ fe fe_pow2k(fe a, int k) {
    for (int i = 0; i < k; i++) a = fe_sqr(a);
    return a;
}

__device__ fe fe_invert(const fe &z) {
    fe z2 = fe_sqr(z);
    fe z9 = fe_mul(fe_pow2k(z2, 2), z);
    fe z11 = fe_mul(z9, z2);
    fe z5 = fe_mul(fe_sqr(z11), z9);
    fe z10 = fe_mul(fe_pow2k(z5, 5), z5);
    fe z20 = fe_mul(fe_pow2k(z10, 10), z10);
    fe z40 = fe_mul(fe_pow2k(z20, 20), z20);
    fe z50 = fe_mul(fe_pow2k(z40, 10), z10);
    fe z100 = fe_mul(fe_pow2k(z50, 50), z50);
    fe z200 = fe_mul(fe_pow2k(z100, 100), z100);
    fe z250 = fe_mul(fe_pow2k(z200, 50), z50);
    return fe_mul(fe_pow2k(z250, 5), z11);
}

// First 8 bytes of the canonical encoding, as a big-endian integer.
__device__ __forceinline__ uint64_t fe_prefix(fe a) {
    uint64_t c = (a.v[7] >> 31) * 19;
    a.v[7] &= 0x7fffffff;
    for (int i = 0; i < 8; i++) {
        c += a.v[i];
        a.v[i] = (uint32_t)c;
        c >>= 32;
    }
    fe t;
    c = 19;
    for (int i = 0; i < 8; i++) {
        c += a.v[i];
        t.v[i] = (uint32_t)c;
        c >>= 32;
    }
    if (t.v[7] >> 31) a = t;
    return (uint64_t)__byte_perm(a.v[0], 0, 0x0123) << 32 | __byte_perm(a.v[1], 0, 0x0123);
}

// Prefix groups: values[starts[g]..starts[g + 1]] are sorted and compared against x & masks[g].
__device__ __forceinline__ bool matches(const uint64_t *masks, const uint32_t *starts,
                                        const uint64_t *values, uint32_t groups, uint64_t x) {
    for (uint32_t g = 0; g < groups; g++) {
        uint64_t key = x & masks[g];
        uint32_t lo = starts[g], hi = starts[g + 1];
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            uint64_t v = values[mid];
            if (v == key) return true;
            if (v < key) lo = mid + 1;
            else hi = mid;
        }
    }
    return false;
}

// State layout: word ((lane * 2 + coord) * 8 + limb) * threads + tid, coords affine u, v.
__device__ __forceinline__ fe load(const uint32_t *s, int lane, int coord, uint32_t tid, uint32_t n) {
    fe r;
    for (int i = 0; i < 8; i++) r.v[i] = s[((lane * 2 + coord) * 8 + i) * n + tid];
    return r;
}

__device__ __forceinline__ void store(uint32_t *s, int lane, int coord, uint32_t tid, uint32_t n,
                                      const fe &a) {
    for (int i = 0; i < 8; i++) s[((lane * 2 + coord) * 8 + i) * n + tid] = a.v[i];
}

// For each i, writes mul, add, sub, invert of a[i], b[i] (8 words each) and the prefix (2 words).
extern "C" __global__ void field_test(const uint32_t *a, const uint32_t *b, uint32_t *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, y;
    for (int k = 0; k < 8; k++) x.v[k] = a[8 * i + k], y.v[k] = b[8 * i + k];
    fe r[4] = {fe_mul(x, y), fe_add(x, y), fe_sub(x, y), fe_invert(x)};
    uint32_t *o = out + 34 * i;
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) o[8 * j + k] = r[j].v[k];
    uint64_t p = fe_prefix(x);
    o[32] = (uint32_t)(p >> 32), o[33] = (uint32_t)p;
}

// Advances every lane by the affine step Q = (q[0..8], q[8..16]) and checks the new u, `iters`
// times. On v^2 = u^3 + A u^2 + u: lambda = (vq - v) / (uq - u), u' = lambda^2 - A - u - uq,
// v' = lambda (u - u') - v, with all lanes of a thread sharing one inversion.
// Hits are appended as (tid, lane, iter, prefix hi, prefix lo) after the count in hits[0].
// Launched with one 128-thread block per SM: lane state and prefix products then fit in L2,
// which measured faster than more resident warps spilling to DRAM.
extern "C" __global__ void __launch_bounds__(128, 1)
    walk(uint32_t *state, const uint32_t *q, const uint64_t *masks, const uint32_t *starts,
         const uint64_t *values, uint32_t groups, uint32_t *hits, uint32_t max_hits, uint32_t iters) {
    uint32_t n = gridDim.x * blockDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    fe qu, qv, acc[LANES];
    fe a_plus_qu = {{486662, 0, 0, 0, 0, 0, 0, 0}};
    for (int i = 0; i < 8; i++) qu.v[i] = q[i], qv.v[i] = q[8 + i];
    a_plus_qu = fe_add(a_plus_qu, qu);
    for (uint32_t it = 0; it < iters; it++) {
        fe a = fe_one();
#pragma unroll 1
        for (int l = 0; l < LANES; l++) {
            acc[l] = a;
            a = fe_mul(a, fe_sub(qu, load(state, l, 0, tid, n)));
        }
        fe inv = fe_invert(a);
#pragma unroll 1
        for (int l = LANES - 1; l >= 0; l--) {
            fe u = load(state, l, 0, tid, n), v = load(state, l, 1, tid, n);
            fe d = fe_sub(qu, u);
            fe lambda = fe_mul(fe_sub(qv, v), fe_mul(inv, acc[l]));
            inv = fe_mul(inv, d);
            fe u3 = fe_sub(fe_sub(fe_sqr(lambda), a_plus_qu), u);
            store(state, l, 0, tid, n, u3);
            store(state, l, 1, tid, n, fe_sub(fe_mul(lambda, fe_sub(u, u3)), v));
            uint64_t p = fe_prefix(u3);
            if (matches(masks, starts, values, groups, p)) {
                uint32_t k = atomicAdd(hits, 1);
                if (k < max_hits) {
                    uint32_t *h = hits + 1 + 5 * k;
                    h[0] = tid, h[1] = l, h[2] = it, h[3] = (uint32_t)(p >> 32), h[4] = (uint32_t)p;
                }
            }
        }
    }
}
