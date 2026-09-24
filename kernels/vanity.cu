// GF(2^255 - 19) in 8 x 32-bit limbs, values kept in [0, 2^256) and reduced via 2^256 = 38.
#include <stdint.h>

// build.rs reads these values for src/cuda.rs.
// Table points per batch; each thread checks 2 * BATCH + 1 keys per iteration.
#define BATCH 512
#define BLOCKS_PER_SM 4

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

// Reduces a 512-bit product r to [0, 2^256) via 2^256 = 38.
__device__ __forceinline__ fe reduce(const uint32_t (&r)[16]) {
    fe l;
    for (int i = 0; i < 8; i++) l.v[i] = r[i];
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
        : "+r"(l.v[0]), "+r"(l.v[1]), "+r"(l.v[2]), "+r"(l.v[3]), "+r"(l.v[4]), "+r"(l.v[5]),
          "+r"(l.v[6]), "+r"(l.v[7]), "+r"(c)
        : "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]),
          "r"(r[15]));
    fold(l, c);
    return l;
}

// Plain 64-bit row sums compile to one IMAD.WIDE per product, cheaper than mad.lo/madc.hi pairs.
__device__ __forceinline__ fe fe_mul(const fe &a, const fe &b) {
    uint32_t r[16] = {0};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t c = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a.v[i] * b.v[j] + r[i + j] + c;
            r[i + j] = (uint32_t)t;
            c = t >> 32;
        }
        r[i + 8] = (uint32_t)c;
    }
    return reduce(r);
}

// Single carry-chain steps. Volatile keeps them in order, so the carry flag flows between them.
#define CHAIN_OP(name, op)                                                                         \
    __device__ __forceinline__ uint32_t name(uint32_t a, uint32_t b, uint32_t c) {                 \
        uint32_t d;                                                                                \
        asm volatile(op " %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));                    \
        return d;                                                                                  \
    }
CHAIN_OP(mad_lo_cc, "mad.lo.cc.u32")
CHAIN_OP(madc_lo_cc, "madc.lo.cc.u32")
CHAIN_OP(mad_hi_cc, "mad.hi.cc.u32")
CHAIN_OP(madc_hi_cc, "madc.hi.cc.u32")

__device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
    uint32_t d;
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
    uint32_t d;
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t carry() {
    uint32_t d;
    asm volatile("addc.u32 %0, 0, 0;" : "=r"(d));
    return d;
}

// Cross products a[i] a[j] (i < j) once, doubled, plus the squares: 36 products instead of 64.
__device__ __forceinline__ fe fe_sqr(const fe &a) {
    uint32_t r[16] = {0};
#pragma unroll
    for (int i = 0; i < 7; i++) {
        r[2 * i + 1] = mad_lo_cc(a.v[i], a.v[i + 1], r[2 * i + 1]);
#pragma unroll
        for (int j = i + 2; j < 8; j++) r[i + j] = madc_lo_cc(a.v[i], a.v[j], r[i + j]);
        r[i + 8] = carry();
        r[2 * i + 2] = mad_hi_cc(a.v[i], a.v[i + 1], r[2 * i + 2]);
#pragma unroll
        for (int j = i + 2; j < 8; j++) r[i + j + 1] = madc_hi_cc(a.v[i], a.v[j], r[i + j + 1]);
    }
    r[1] = add_cc(r[1], r[1]);
#pragma unroll
    for (int k = 2; k < 15; k++) r[k] = addc_cc(r[k], r[k]);
    r[15] = carry();
    r[0] = mad_lo_cc(a.v[0], a.v[0], r[0]);
    r[1] = madc_hi_cc(a.v[0], a.v[0], r[1]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
        r[2 * i] = madc_lo_cc(a.v[i], a.v[i], r[2 * i]);
        r[2 * i + 1] = madc_hi_cc(a.v[i], a.v[i], r[2 * i + 1]);
    }
    return reduce(r);
}

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

// State layout: word tid * 16 + coord * 8 + limb, coords affine u, v, so the host can replace
// one thread's center with a single copy.
__device__ __forceinline__ fe load(const uint32_t *s, int coord, uint32_t tid) {
    fe r;
    for (int i = 0; i < 8; i++) r.v[i] = s[tid * 16 + coord * 8 + i];
    return r;
}

__device__ __forceinline__ void store(uint32_t *s, int coord, uint32_t tid, const fe &a) {
    for (int i = 0; i < 8; i++) s[tid * 16 + coord * 8 + i] = a.v[i];
}

// Table entries are 32-byte aligned, so each coordinate is two 128-bit loads.
__device__ __forceinline__ fe entry(const uint32_t *__restrict__ table, int i, int coord) {
    const uint4 *p = (const uint4 *)(table + (i * 2 + coord) * 8);
    uint4 lo = __ldg(p), hi = __ldg(p + 1);
    return {{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w}};
}

// For each i, writes mul, add, sub, invert, sqr of a[i], b[i] (8 words each) and the prefix (2 words).
extern "C" __global__ void field_test(const uint32_t *a, const uint32_t *b, uint32_t *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe x, y;
    for (int k = 0; k < 8; k++) x.v[k] = a[8 * i + k], y.v[k] = b[8 * i + k];
    fe r[5] = {fe_mul(x, y), fe_add(x, y), fe_sub(x, y), fe_invert(x), fe_sqr(x)};
    uint32_t *o = out + 42 * i;
    for (int j = 0; j < 5; j++)
        for (int k = 0; k < 8; k++) o[8 * j + k] = r[j].v[k];
    uint64_t p = fe_prefix(x);
    o[40] = (uint32_t)(p >> 32), o[41] = (uint32_t)p;
}

struct Matcher {
    const uint64_t *masks;
    const uint32_t *starts;
    const uint64_t *values;
    uint32_t groups;
    uint64_t first_chars, second_chars;
    uint32_t *hits;
    uint32_t max_hits;
};

// Hits are appended as (tid, j, iter, prefix hi, prefix lo) after the count in hits[0].
// The prefix filter assumes u mod p = u - p * (u >> 255), which only fails for u in [p, 2^255) or
// u >= 2p (probability ~2^-250); matches are confirmed with the exact prefix.
__device__ __forceinline__ void check(const Matcher &m, const fe &u, uint32_t tid, uint32_t j, uint32_t it) {
    uint64_t lo = ((uint64_t)u.v[1] << 32 | u.v[0]) + 19 * (u.v[7] >> 31);
    uint64_t p = (uint64_t)__byte_perm((uint32_t)lo, 0, 0x0123) << 32 | __byte_perm((uint32_t)(lo >> 32), 0, 0x0123);
    bool maybe = m.first_chars >> (p >> 58) & m.second_chars >> (p >> 52 & 63) & 1;
    if (maybe && matches(m.masks, m.starts, m.values, m.groups, p) &&
        fe_prefix(u) == p) {
        uint32_t k = atomicAdd(m.hits, 1);
        if (k < m.max_hits) {
            uint32_t *h = m.hits + 1 + 5 * k;
            h[0] = tid, h[1] = j, h[2] = it, h[3] = (uint32_t)(p >> 32), h[4] = (uint32_t)p;
        }
    }
}

// Each thread holds a center C and checks u(C + (j - BATCH) R) for j in 0..=2 BATCH, where
// table entry i < BATCH is (i + 1) R and entry BATCH is the center step (2 BATCH + 1) R.
// On v^2 = u^3 + A u^2 + u, u(C +- P) = lambda^2 - A - uC - uP with lambda = (+-vP - vC) / (uP - uC),
// so both signs share one inverse, and all inverses of a thread share one batch inversion.
extern "C" __global__ void __launch_bounds__(128, BLOCKS_PER_SM)
    walk(uint32_t *state, const uint32_t *__restrict__ table, const uint64_t *masks, const uint32_t *starts,
         const uint64_t *values, uint32_t groups, uint64_t first_chars, uint64_t second_chars,
         uint32_t *hits, uint32_t max_hits, uint32_t iters) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    Matcher m = {masks, starts, values, groups, first_chars, second_chars, hits, max_hits};
    fe cu = load(state, 0, tid), cv = load(state, 1, tid), acc[BATCH + 1];
    const fe a_coef = {{486662, 0, 0, 0, 0, 0, 0, 0}};
    for (uint32_t it = 0; it < iters; it++) {
        fe a = fe_one();
#pragma unroll 1
        for (int i = 0; i <= BATCH; i++) {
            acc[i] = a;
            a = fe_mul(a, fe_sub(entry(table, i, 0), cu));
        }
        fe inv = fe_invert(a);
        fe su = entry(table, BATCH, 0);
        fe inv_s = fe_mul(inv, acc[BATCH]);
        inv = fe_mul(inv, fe_sub(su, cu));
        fe a_cu = fe_add(a_coef, cu);
        check(m, cu, tid, BATCH, it);
#pragma unroll 1
        for (int i = BATCH - 1; i >= 0; i--) {
            fe ru = entry(table, i, 0), rv = entry(table, i, 1);
            fe di = fe_mul(inv, acc[i]);
            inv = fe_mul(inv, fe_sub(ru, cu));
            fe base = fe_add(a_cu, ru);
            fe lp = fe_mul(fe_sub(rv, cv), di), lm = fe_mul(fe_add(rv, cv), di);
            check(m, fe_sub(fe_sqr(lp), base), tid, BATCH + 1 + i, it);
            check(m, fe_sub(fe_sqr(lm), base), tid, BATCH - 1 - i, it);
        }
        fe lambda = fe_mul(fe_sub(entry(table, BATCH, 1), cv), inv_s);
        fe nu = fe_sub(fe_sub(fe_sqr(lambda), a_cu), su);
        cv = fe_sub(fe_mul(lambda, fe_sub(cu, nu)), cv);
        cu = nu;
    }
    store(state, 0, tid, cu);
    store(state, 1, tid, cv);
}
