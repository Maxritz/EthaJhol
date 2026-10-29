#include "vg/cpu_ops.h"
#include "vg/quant.h"
#include "vg/threadpool.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/* --- Quantized block structures (GGML-compatible, byte exact) --- */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;               /* 18 B */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;                /* 34 B */
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K; /* 144 B */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K; /* 210 B */
typedef struct { uint8_t qs[48]; uint8_t qh[4]; uint16_t d; } block_tq1_0;   /* 54 B / 256 */
typedef struct { uint8_t qs[64]; uint16_t d; } block_tq2_0;                  /* 66 B / 256 */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q1_0;                   /* 18 B / 128 */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q2_0;                   /* 18 B /  64 */

/* The ternary kernels stream 1.7-2.1 bits per weight, so they are pure memory-bound
 * workloads: the only lever is keeping the load pipe full while the unpack runs. */
#if defined(__GNUC__) || defined(__clang__)
#define VG_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#else
#define VG_PREFETCH(p) ((void)0)
#endif

#define VG_ROW_BLOCK 256u


/* --- Worker pool ---------------------------------------------------------------- */
static int g_threads_active = 1;
static VG_ThreadPool *g_pool = NULL;

static int hardware_threads(void) {
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* Worker count actually usable: 0 means "all hardware threads", capped at 16 so a 128-core
 * box does not spawn a thread per row chunk. */
static int clamp_threads(int want) {
    if (want <= 0) want = hardware_threads();
    if (want > 16) want = 16;
    if (want < 1) want = 1;
    return want;
}

/* The pool used to be created once and then never resized, so a later request for more threads
 * was silently ignored while the old count was still reported. Recreate it whenever the
 * effective count changes. Call this during start-up, not from inside a running dispatch. */
int vg_cpu_set_threads(int n_threads) {
    const int want = clamp_threads(n_threads);
    if (want == g_threads_active && (want == 1) == (g_pool == NULL)) return g_threads_active;
    if (g_pool) { vg_threadpool_destroy(g_pool); g_pool = NULL; }
    g_threads_active = want;
    if (want > 1) {
        g_pool = vg_threadpool_create((unsigned)want);
        if (!g_pool) g_threads_active = 1;
    }
    return g_threads_active;
}

int vg_cpu_get_threads(void) {
    if (g_threads_active < 1) g_threads_active = 1;
    return g_threads_active;
}

/* --- Per-type row kernels ------------------------------------------------------- */

typedef void (*vg_row_fn)(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim);

static void rows_f32(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const float *w = (const float *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const float *wr = w + (uint64_t)r * in_dim;
        float acc = 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) acc += wr[j] * x[j];
        y[r] = acc;
    }
}

static void rows_f16(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint16_t *w = (const uint16_t *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const uint16_t *wr = w + (uint64_t)r * in_dim;
        float acc = 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) acc += vg_half_to_float(wr[j]) * x[j];
        y[r] = acc;
    }
}

static void rows_q8_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t blocks = in_dim / 32u;
    const block_q8_0 *w = (const block_q8_0 *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_q8_0 *row = w + (uint64_t)r * blocks;
        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; ++b) {
            const float d = vg_half_to_float(row[b].d);
            const int8_t *qs = row[b].qs;
            const float *xb = x + b * 32u;
            float sum = 0.0f;
            for (uint32_t j = 0; j < 32u; ++j) sum += (float)qs[j] * xb[j];
            acc += d * sum;
        }
        y[r] = acc;
    }
}

/* Two rows per pass: the nibble unpack of one row is dependent scalar work, so interleaving a
 * second row gives the converter an independent chain to fill the pipeline. The per-row
 * accumulation order is unchanged, so results stay bit-identical to the reference path. */
static void rows_q4_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t blocks = in_dim / 32u;
    const block_q4_0 *w = (const block_q4_0 *)W;
    uint32_t r = r0;
    for (; r + 1u < r1; r += 2u) {
        const block_q4_0 *row0 = w + (uint64_t)r * blocks;
        const block_q4_0 *row1 = row0 + blocks;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint8_t *q0 = row0[b].qs;
            const uint8_t *q1 = row1[b].qs;
            const float *xb = x + b * 32u;
            float s0 = 0.0f, s1 = 0.0f;
            for (uint32_t j = 0; j < 16u; ++j) {
                const float xl = xb[j], xh = xb[16u + j];
                s0 += (float)((int)(q0[j] & 0x0Fu) - 8) * xl + (float)((int)(q0[j] >> 4) - 8) * xh;
                s1 += (float)((int)(q1[j] & 0x0Fu) - 8) * xl + (float)((int)(q1[j] >> 4) - 8) * xh;
            }
            acc0 += vg_half_to_float(row0[b].d) * s0;
            acc1 += vg_half_to_float(row1[b].d) * s1;
        }
        y[r] = acc0;
        y[r + 1u] = acc1;
    }
    for (; r < r1; ++r) {
        const block_q4_0 *row = w + (uint64_t)r * blocks;
        float acc = 0.0f;
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint8_t *qs = row[b].qs;
            const float *xb = x + b * 32u;
            float sum = 0.0f;
            for (uint32_t j = 0; j < 16u; ++j) {
                const int lo = (int)(qs[j] & 0x0Fu) - 8;
                const int hi = (int)(qs[j] >> 4) - 8;
                sum += (float)lo * xb[j] + (float)hi * xb[16u + j];
            }
            acc += vg_half_to_float(row[b].d) * sum;
        }
        y[r] = acc;
    }
}

/* --- Ternary / 1-bit fused kernels ---------------------------------------------
 * Same block layouts and element ORDER as ggml. The scale is factored out of the inner
 * loop (sum of q*x, then one multiply), which keeps the accumulation in one dependency
 * chain while the unpack of the next byte runs in parallel. */
static const uint8_t k_pow3[5] = { 1, 3, 9, 27, 81 };

/* Constant table instead of a per-element int->float conversion; the codes are
 * {-1,0,+1,+2} for TQ2_0 / Q2_0 and stay in L1 as a single cache line. */
static const float k_tern2[4] = { -1.0f, 0.0f, 1.0f, 2.0f };
/* Branchless sign for 1-bit weights: indexing a two-entry table beats a ternary here,
 * which the compiler otherwise turns into a branch inside the innermost loop. */
static const float k_sign[2] = { -1.0f, 1.0f };

static void rows_tq2_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / 256u;
    const block_tq2_0 *w = (const block_tq2_0 *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_tq2_0 *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            VG_PREFETCH(row + ib + 2);
            const uint8_t *restrict qs = row[ib].qs;
            const float *restrict xb = x + (size_t)ib * 256u;
            /* One byte feeds FOUR independent planes, so load it once and run four
             * accumulator chains: the packed byte is the scarce resource here, not the x
             * value, and each chain breaks the dependency on the previous MAC. */
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (uint32_t g = 0; g < 2u; ++g) {
                const uint8_t *q = qs + g * 32u;
                const float *x0 = xb + g * 128u;
                const float *x1 = x0 + 32u, *x2 = x0 + 64u, *x3 = x0 + 96u;
                for (uint32_t m = 0; m < 32u; ++m) {
                    const uint8_t b = q[m];
                    s0 += k_tern2[b & 3u] * x0[m];
                    s1 += k_tern2[(b >> 2) & 3u] * x1[m];
                    s2 += k_tern2[(b >> 4) & 3u] * x2[m];
                    s3 += k_tern2[(b >> 6) & 3u] * x3[m];
                }
            }
            acc += vg_half_to_float(row[ib].d) * ((s0 + s1) + (s2 + s3));
        }
        y[r] = acc;
    }
}

/* One radix-3 value out of a packed byte, exactly as ggml decodes it. */
static inline int tq1_val(uint8_t b, uint8_t p3) {
    const uint8_t packed = (uint8_t)(b * p3);
    return (int)(((uint16_t)packed * 3u) >> 8) - 1;
}

/* Radix-3 decoding costs a multiply, a shift and a subtract per weight. The mapping depends
 * only on the packed byte and the plane, so it is a compile-time table (1.25 KiB, stays in
 * L1). A lazily-built table would race between worker threads on the first dispatch. */
static const int8_t k_tq1_lut[256][5] = {
    {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}, {-1, -1, -1, -1, 0}, {-1, -1, -1, -1, 1}, {-1, -1, -1, 0, -1}, {-1, -1, -1, 0, 0}, {-1, -1, -1, 0, 1}, {-1, -1, -1, 1, -1},
    {-1, -1, -1, 1, 0}, {-1, -1, -1, 1, 1}, {-1, -1, 0, -1, -1}, {-1, -1, 0, -1, 0}, {-1, -1, 0, -1, 1}, {-1, -1, 0, 0, -1}, {-1, -1, 0, 0, 0}, {-1, -1, 0, 0, 1},
    {-1, -1, 0, 1, -1}, {-1, -1, 0, 1, 0}, {-1, -1, 0, 1, 1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, 0}, {-1, -1, 1, -1, 1}, {-1, -1, 1, 0, -1},
    {-1, -1, 1, 0, 0}, {-1, -1, 1, 0, 1}, {-1, -1, 1, 1, -1}, {-1, -1, 1, 1, 0}, {-1, -1, 1, 1, 1}, {-1, 0, -1, -1, -1}, {-1, 0, -1, -1, 0}, {-1, 0, -1, -1, 1},
    {-1, 0, -1, 0, -1}, {-1, 0, -1, 0, 0}, {-1, 0, -1, 0, 1}, {-1, 0, -1, 1, -1}, {-1, 0, -1, 1, 0}, {-1, 0, -1, 1, 1}, {-1, 0, 0, -1, -1}, {-1, 0, 0, -1, 0},
    {-1, 0, 0, -1, 0}, {-1, 0, 0, -1, 1}, {-1, 0, 0, 0, -1}, {-1, 0, 0, 0, 0}, {-1, 0, 0, 0, 1}, {-1, 0, 0, 1, -1}, {-1, 0, 0, 1, 0}, {-1, 0, 0, 1, 1},
    {-1, 0, 1, -1, -1}, {-1, 0, 1, -1, 0}, {-1, 0, 1, -1, 1}, {-1, 0, 1, 0, -1}, {-1, 0, 1, 0, 0}, {-1, 0, 1, 0, 1}, {-1, 0, 1, 1, -1}, {-1, 0, 1, 1, 0},
    {-1, 0, 1, 1, 1}, {-1, 1, -1, -1, -1}, {-1, 1, -1, -1, 0}, {-1, 1, -1, -1, 1}, {-1, 1, -1, -1, 1}, {-1, 1, -1, 0, -1}, {-1, 1, -1, 0, 0}, {-1, 1, -1, 0, 1},
    {-1, 1, -1, 1, -1}, {-1, 1, -1, 1, 0}, {-1, 1, -1, 1, 1}, {-1, 1, 0, -1, -1}, {-1, 1, 0, -1, 0}, {-1, 1, 0, -1, 1}, {-1, 1, 0, 0, -1}, {-1, 1, 0, 0, 0},
    {-1, 1, 0, 0, 1}, {-1, 1, 0, 1, -1}, {-1, 1, 0, 1, 0}, {-1, 1, 0, 1, 1}, {-1, 1, 1, -1, -1}, {-1, 1, 1, -1, 0}, {-1, 1, 1, -1, 1}, {-1, 1, 1, -1, 1},
    {-1, 1, 1, 0, -1}, {-1, 1, 1, 0, 0}, {-1, 1, 1, 0, 1}, {-1, 1, 1, 1, -1}, {-1, 1, 1, 1, 0}, {-1, 1, 1, 1, 1}, {0, -1, -1, -1, -1}, {0, -1, -1, -1, 0},
    {0, -1, -1, -1, 1}, {0, -1, -1, 0, -1}, {0, -1, -1, 0, 0}, {0, -1, -1, 0, 1}, {0, -1, -1, 1, -1}, {0, -1, -1, 1, 0}, {0, -1, -1, 1, 1}, {0, -1, 0, -1, -1},
    {0, -1, 0, -1, 0}, {0, -1, 0, -1, 1}, {0, -1, 0, 0, -1}, {0, -1, 0, 0, -1}, {0, -1, 0, 0, 0}, {0, -1, 0, 0, 1}, {0, -1, 0, 1, -1}, {0, -1, 0, 1, 0},
    {0, -1, 0, 1, 1}, {0, -1, 1, -1, -1}, {0, -1, 1, -1, 0}, {0, -1, 1, -1, 1}, {0, -1, 1, 0, -1}, {0, -1, 1, 0, 0}, {0, -1, 1, 0, 1}, {0, -1, 1, 1, -1},
    {0, -1, 1, 1, 0}, {0, -1, 1, 1, 1}, {0, 0, -1, -1, -1}, {0, 0, -1, -1, 0}, {0, 0, -1, -1, 1}, {0, 0, -1, 0, -1}, {0, 0, -1, 0, 0}, {0, 0, -1, 0, 0},
    {0, 0, -1, 0, 1}, {0, 0, -1, 1, -1}, {0, 0, -1, 1, 0}, {0, 0, -1, 1, 1}, {0, 0, 0, -1, -1}, {0, 0, 0, -1, 0}, {0, 0, 0, -1, 1}, {0, 0, 0, 0, -1},
    {0, 0, 0, 0, 0}, {0, 0, 0, 0, 1}, {0, 0, 0, 1, -1}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 1, -1, -1}, {0, 0, 1, -1, 0}, {0, 0, 1, -1, 1},
    {0, 0, 1, 0, -1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 1}, {0, 0, 1, 1, -1}, {0, 0, 1, 1, 0}, {0, 0, 1, 1, 1}, {0, 1, -1, -1, -1},
    {0, 1, -1, -1, 0}, {0, 1, -1, -1, 1}, {0, 1, -1, 0, -1}, {0, 1, -1, 0, 0}, {0, 1, -1, 0, 1}, {0, 1, -1, 1, -1}, {0, 1, -1, 1, 0}, {0, 1, -1, 1, 1},
    {0, 1, 0, -1, -1}, {0, 1, 0, -1, 0}, {0, 1, 0, -1, 1}, {0, 1, 0, 0, -1}, {0, 1, 0, 0, 0}, {0, 1, 0, 0, 1}, {0, 1, 0, 0, 1}, {0, 1, 0, 1, -1},
    {0, 1, 0, 1, 0}, {0, 1, 0, 1, 1}, {0, 1, 1, -1, -1}, {0, 1, 1, -1, 0}, {0, 1, 1, -1, 1}, {0, 1, 1, 0, -1}, {0, 1, 1, 0, 0}, {0, 1, 1, 0, 1},
    {0, 1, 1, 1, -1}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, 0}, {1, -1, -1, -1, 1}, {1, -1, -1, 0, -1}, {1, -1, -1, 0, 0},
    {1, -1, -1, 0, 1}, {1, -1, -1, 1, -1}, {1, -1, -1, 1, -1}, {1, -1, -1, 1, 0}, {1, -1, -1, 1, 1}, {1, -1, 0, -1, -1}, {1, -1, 0, -1, 0}, {1, -1, 0, -1, 1},
    {1, -1, 0, 0, -1}, {1, -1, 0, 0, 0}, {1, -1, 0, 0, 1}, {1, -1, 0, 1, -1}, {1, -1, 0, 1, 0}, {1, -1, 0, 1, 1}, {1, -1, 1, -1, -1}, {1, -1, 1, -1, 0},
    {1, -1, 1, -1, 1}, {1, -1, 1, 0, -1}, {1, -1, 1, 0, 0}, {1, -1, 1, 0, 1}, {1, -1, 1, 1, -1}, {1, -1, 1, 1, -1}, {1, -1, 1, 1, 0}, {1, -1, 1, 1, 1},
    {1, 0, -1, -1, -1}, {1, 0, -1, -1, 0}, {1, 0, -1, -1, 1}, {1, 0, -1, 0, -1}, {1, 0, -1, 0, 0}, {1, 0, -1, 0, 1}, {1, 0, -1, 1, -1}, {1, 0, -1, 1, 0},
    {1, 0, -1, 1, 1}, {1, 0, 0, -1, -1}, {1, 0, 0, -1, 0}, {1, 0, 0, -1, 1}, {1, 0, 0, 0, -1}, {1, 0, 0, 0, 0}, {1, 0, 0, 0, 1}, {1, 0, 0, 1, -1},
    {1, 0, 0, 1, 0}, {1, 0, 0, 1, 0}, {1, 0, 0, 1, 1}, {1, 0, 1, -1, -1}, {1, 0, 1, -1, 0}, {1, 0, 1, -1, 1}, {1, 0, 1, 0, -1}, {1, 0, 1, 0, 0},
    {1, 0, 1, 0, 1}, {1, 0, 1, 1, -1}, {1, 0, 1, 1, 0}, {1, 0, 1, 1, 1}, {1, 1, -1, -1, -1}, {1, 1, -1, -1, 0}, {1, 1, -1, -1, 1}, {1, 1, -1, 0, -1},
    {1, 1, -1, 0, 0}, {1, 1, -1, 0, 1}, {1, 1, -1, 1, -1}, {1, 1, -1, 1, 0}, {1, 1, -1, 1, 1}, {1, 1, -1, 1, 1}, {1, 1, 0, -1, -1}, {1, 1, 0, -1, 0},
    {1, 1, 0, -1, 1}, {1, 1, 0, 0, -1}, {1, 1, 0, 0, 0}, {1, 1, 0, 0, 1}, {1, 1, 0, 1, -1}, {1, 1, 0, 1, 0}, {1, 1, 0, 1, 1}, {1, 1, 1, -1, -1},
    {1, 1, 1, -1, 0}, {1, 1, 1, -1, 1}, {1, 1, 1, 0, -1}, {1, 1, 1, 0, 0}, {1, 1, 1, 0, 1}, {1, 1, 1, 1, -1}, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 1},
};


static void rows_tq1_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / 256u;
    const int8_t (*restrict lut)[5] = k_tq1_lut;
    const block_tq1_0 *w = (const block_tq1_0 *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_tq1_0 *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            VG_PREFETCH(row + ib + 2);
            const uint8_t *qs = row[ib].qs;
            const uint8_t *qh = row[ib].qh;
            const float *xb = x + (size_t)ib * 256u;
            float s = 0.0f;
            /* 32-byte group: 5 planes of 32 consecutive values. Each byte carries one value
             * per plane, so it is loaded once and reused across the five chains. */
            {
                const float *xp = xb;
                for (uint32_t m = 0; m < 32u; ++m) {
                    const uint8_t b = qs[m];
                    s += (float)lut[b][0] * xp[m];
                    s += (float)lut[b][1] * xp[32u + m];
                    s += (float)lut[b][2] * xp[64u + m];
                    s += (float)lut[b][3] * xp[96u + m];
                    s += (float)lut[b][4] * xp[128u + m];
                }
            }
            /* 16-byte tail: 5 planes of 16 values each. */
            {
                const float *xp = xb + 160u;
                for (uint32_t m = 0; m < 16u; ++m) {
                    const uint8_t b = qs[32u + m];
                    s += (float)lut[b][0] * xp[m];
                    s += (float)lut[b][1] * xp[16u + m];
                    s += (float)lut[b][2] * xp[32u + m];
                    s += (float)lut[b][3] * xp[48u + m];
                    s += (float)lut[b][4] * xp[64u + m];
                }
            }
            /* qh: 4 planes of 4 values. */
            for (uint32_t n = 0; n < 4u; ++n) {
                const uint8_t p3 = k_pow3[n];
                const float *xp = xb + 240u + n * 4u;
                for (uint32_t m = 0; m < 4u; ++m) {
                    const uint8_t packed = (uint8_t)(qh[m] * p3);
                    const int xi = (int)(((uint16_t)packed * 3u) >> 8) - 1;
                    s += (float)xi * xp[m];
                }
            }
            acc += vg_half_to_float(row[ib].d) * s;
        }
        y[r] = acc;
    }
}

static void rows_q1_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / 128u;
    const block_q1_0 *w = (const block_q1_0 *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_q1_0 *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            VG_PREFETCH(row + ib + 2);
            const float *xb = x + (size_t)ib * 128u;
            const uint8_t *restrict qs = row[ib].qs;
            /* One byte -> 8 weights. The old loop divided and modded per element to find its
             * byte; hoisting that out removes 255 integer divisions per block. */
            float s = 0.0f;
            for (uint32_t j = 0; j < 16u; ++j) {
                const uint8_t b = qs[j];
                const float *xp = xb + j * 8u;
                s += k_sign[(b >> 0) & 1u] * xp[0];
                s += k_sign[(b >> 1) & 1u] * xp[1];
                s += k_sign[(b >> 2) & 1u] * xp[2];
                s += k_sign[(b >> 3) & 1u] * xp[3];
                s += k_sign[(b >> 4) & 1u] * xp[4];
                s += k_sign[(b >> 5) & 1u] * xp[5];
                s += k_sign[(b >> 6) & 1u] * xp[6];
                s += k_sign[(b >> 7) & 1u] * xp[7];
            }
            acc += vg_half_to_float(row[ib].d) * s;
        }
        y[r] = acc;
    }
}

static void rows_q2_0(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / 64u;
    const block_q2_0 *w = (const block_q2_0 *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_q2_0 *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            VG_PREFETCH(row + ib + 2);
            const float *xb = x + (size_t)ib * 64u;
            const uint8_t *restrict qs = row[ib].qs;
            float s = 0.0f;
            /* 4 values per byte: one load, four chains. */
            for (uint32_t j = 0; j < 16u; ++j) {
                const uint8_t b = qs[j];
                s += k_tern2[b & 3u] * xb[j * 4u + 0u];
                s += k_tern2[(b >> 2) & 3u] * xb[j * 4u + 1u];
                s += k_tern2[(b >> 4) & 3u] * xb[j * 4u + 2u];
                s += k_tern2[(b >> 6) & 3u] * xb[j * 4u + 3u];
            }
            acc += vg_half_to_float(row[ib].d) * s;
        }
        y[r] = acc;
    }
}

static uint8_t get_scale_k4(int j, const uint8_t *q) {
    return j < 4 ? (uint8_t)(q[j] & 63u) : (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
}
static uint8_t get_min_k4(int j, const uint8_t *q) {
    return j < 4 ? (uint8_t)(q[j + 4] & 63u) : (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

/* Two rows per pass, same interleaving trick as Q4_0: twice the independent accumulator chains
 * for the same loads of the input vector. Per-row arithmetic order is unchanged. */
static void rows_q4_k(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / VG_ROW_BLOCK;
    const block_q4_K *w = (const block_q4_K *)W;
    uint32_t r = r0;
    for (; r + 1u < r1; r += 2u) {
        const block_q4_K *row0 = w + (uint64_t)r * nbs;
        const block_q4_K *row1 = row0 + nbs;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            const float d0 = vg_half_to_float(row0[ib].d), mn0 = vg_half_to_float(row0[ib].dmin);
            const float d1v = vg_half_to_float(row1[ib].d), mn1 = vg_half_to_float(row1[ib].dmin);
            const uint8_t *sc0 = row0[ib].scales, *sc1 = row1[ib].scales;
            const uint8_t *q0 = row0[ib].qs, *q1s = row1[ib].qs;
            const float *xb = x + ib * VG_ROW_BLOCK;
            uint32_t off = 0;
            for (int is = 0; is < 8; is += 2) {
                const float a0 = d0 * get_scale_k4(is + 0, sc0), b0 = mn0 * get_min_k4(is + 0, sc0);
                const float c0 = d0 * get_scale_k4(is + 1, sc0), e0 = mn0 * get_min_k4(is + 1, sc0);
                const float a1 = d1v * get_scale_k4(is + 0, sc1), b1 = mn1 * get_min_k4(is + 0, sc1);
                const float c1 = d1v * get_scale_k4(is + 1, sc1), e1 = mn1 * get_min_k4(is + 1, sc1);
                for (uint32_t l = 0; l < 32u; ++l) {
                    const float xl = xb[off + l], xh = xb[off + 32u + l];
                    acc0 += (a0 * (float)(q0[l] & 0x0Fu) - b0) * xl;
                    acc0 += (c0 * (float)(q0[l] >> 4) - e0) * xh;
                    acc1 += (a1 * (float)(q1s[l] & 0x0Fu) - b1) * xl;
                    acc1 += (c1 * (float)(q1s[l] >> 4) - e1) * xh;
                }
                q0 += 32; q1s += 32; off += 64u;
            }
        }
        y[r] = acc0;
        y[r + 1u] = acc1;
    }
    for (; r < r1; ++r) {
        const block_q4_K *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            const float d = vg_half_to_float(row[ib].d);
            const float mn = vg_half_to_float(row[ib].dmin);
            const uint8_t *sc = row[ib].scales;
            const uint8_t *qs = row[ib].qs;
            const float *xb = x + ib * VG_ROW_BLOCK;
            int is = 0;
            uint32_t off = 0;
            for (int j = 0; j < 256; j += 64) {
                const float d1 = d * get_scale_k4(is + 0, sc);
                const float m1 = mn * get_min_k4(is + 0, sc);
                const float d2 = d * get_scale_k4(is + 1, sc);
                const float m2 = mn * get_min_k4(is + 1, sc);
                for (uint32_t l = 0; l < 32u; ++l) {
                    acc += (d1 * (float)(qs[l] & 0x0Fu) - m1) * xb[off + l];
                    acc += (d2 * (float)(qs[l] >> 4) - m2) * xb[off + 32u + l];
                }
                qs += 32; is += 2; off += 64u;
            }
        }
        y[r] = acc;
    }
}

static void rows_q6_k(const void *W, const float *x, float *y, uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const uint32_t nbs = in_dim / VG_ROW_BLOCK;
    const block_q6_K *w = (const block_q6_K *)W;
    for (uint32_t r = r0; r < r1; ++r) {
        const block_q6_K *row = w + (uint64_t)r * nbs;
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            const float d = vg_half_to_float(row[ib].d);
            const int8_t *sc = row[ib].scales;
            const uint8_t *ql = row[ib].ql;
            const uint8_t *qh = row[ib].qh;
            const float *xb = x + ib * VG_ROW_BLOCK;
            /* A Q6_K block holds two 128-value halves: 64 ql bytes, 32 qh bytes and 8
             * scales each. Both halves must be consumed. */
            for (int half = 0; half < 2; ++half) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int v1 = (int)((ql[l] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                    const int v2 = (int)((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                    const int v3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                    const int v4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                    acc += d * (float)sc[is + 0] * (float)v1 * xb[half * 128 + l];
                    acc += d * (float)sc[is + 2] * (float)v2 * xb[half * 128 + 32 + l];
                    acc += d * (float)sc[is + 4] * (float)v3 * xb[half * 128 + 64 + l];
                    acc += d * (float)sc[is + 6] * (float)v4 * xb[half * 128 + 96 + l];
                }
                ql += 64; qh += 32; sc += 8;
            }
        }
        y[r] = acc;
    }
}

/* Every other dequantizable type: expand one row at a time and dot it. No full-matrix
 * temporary is ever allocated, so a 4096x4096 Q5_K tensor costs one 16 KiB row buffer. */
static void rows_generic_deq(uint32_t type, const void *W, const float *x, float *y,
                             uint32_t r0, uint32_t r1, uint32_t in_dim) {
    const VG_QuantInfo *qi = vg_quant_info(type);
    if (!qi || !qi->supported_scalar) {
        for (uint32_t r = r0; r < r1; ++r) y[r] = NAN;
        return;
    }
    const uint64_t row_bytes = (uint64_t)(in_dim / qi->block_size) * qi->bytes_per_block;
    float *row = (float *)malloc((size_t)in_dim * sizeof(float));
    if (!row) { for (uint32_t r = r0; r < r1; ++r) y[r] = NAN; return; }
    for (uint32_t r = r0; r < r1; ++r) {
        const void *src = (const unsigned char *)W + (uint64_t)r * row_bytes;
        if (vg_dequantize_row(type, src, (size_t)row_bytes, row, in_dim) != VG_OK) { y[r] = NAN; continue; }
        float acc = 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[r] = acc;
    }
    free(row);
}

static vg_row_fn pick_rows(uint32_t type) {
    switch (type) {
        case VG_QUANT_F32: return rows_f32;
        case VG_QUANT_F16: return rows_f16;
        case VG_QUANT_Q8_0: return rows_q8_0;
        case VG_QUANT_Q4_0: return rows_q4_0;
        case VG_QUANT_Q4_K: return rows_q4_k;
        case VG_QUANT_Q6_K: return rows_q6_k;
        case VG_QUANT_TQ1_0: return rows_tq1_0;
        case VG_QUANT_TQ2_0: return rows_tq2_0;
        case VG_QUANT_Q1_0: return rows_q1_0;
        case VG_QUANT_Q2_0: return rows_q2_0;
        default: return NULL;
    }
}

typedef struct VG_MatJob {
    uint32_t type;
    const void *W;
    const float *x;
    float *y;
    uint32_t in_dim;
    vg_row_fn rows;
} VG_MatJob;

static void mat_job(void *arg, uint32_t begin, uint32_t end) {
    VG_MatJob *j = (VG_MatJob *)arg;
    if (begin >= end) return;
    if (j->rows) j->rows(j->W, j->x, j->y, begin, end, j->in_dim);
    else rows_generic_deq(j->type, j->W, j->x, j->y, begin, end, j->in_dim);
}

static VG_Status matmul_impl(uint32_t type, const void *W, const float *x, float *y,
                             uint32_t out_dim, uint32_t in_dim, int threaded) {
    if (!W || !x || !y || !out_dim || !in_dim) return VG_E_INVALID;
    const VG_QuantInfo *qi = vg_quant_info(type);
    if (!qi) return VG_E_UNSUPPORTED;
    if (in_dim % qi->block_size) return VG_E_FORMAT;
    if (!qi->supported_scalar) return VG_E_UNSUPPORTED;

    VG_MatJob job;
    job.type = type; job.W = W; job.x = x; job.y = y; job.in_dim = in_dim; job.rows = pick_rows(type);
    if (threaded && g_pool && g_threads_active > 1 && out_dim > 1 && in_dim >= 64u) {
        vg_threadpool_parallel_for(g_pool, mat_job, &job, out_dim);
    } else {
        mat_job(&job, 0, out_dim);
    }
    return VG_OK;
}

VG_Status vg_cpu_matmul(uint32_t type, const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matmul_impl(type, W, x, y, out_dim, in_dim, 1);
}

VG_Status vg_cpu_matmul_ref(uint32_t type, const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matmul_impl(type, W, x, y, out_dim, in_dim, 0);
}

void vg_cpu_matmul_q4_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_Q4_0, W, x, y, out_dim, in_dim);
}
void vg_cpu_matmul_q4_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_Q4_K, W, x, y, out_dim, in_dim);
}
void vg_cpu_matmul_q6_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_Q6_K, W, x, y, out_dim, in_dim);
}
void vg_cpu_matmul_q8_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_Q8_0, W, x, y, out_dim, in_dim);
}
void vg_cpu_matmul_f32(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_F32, W, x, y, out_dim, in_dim);
}
void vg_cpu_matmul_f16(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vg_cpu_matmul(VG_QUANT_F16, W, x, y, out_dim, in_dim);
}

/* --- Elementwise / reduction kernels -------------------------------------------- */

void vg_cpu_rmsnorm(float *out, const float *x, const float *weight, uint32_t dim, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)dim + eps);
    if (weight) {
        for (uint32_t i = 0; i < dim; ++i) out[i] = x[i] * scale * weight[i];
    } else {
        for (uint32_t i = 0; i < dim; ++i) out[i] = x[i] * scale;
    }
}

void vg_cpu_rope_table(float *cos_out, float *sin_out, uint32_t n_rot, uint32_t head_dim,
                       float freq_scale, float theta_base, int32_t pos) {
    const uint32_t half = head_dim / 2u;
    const uint32_t iters = n_rot < half ? n_rot : half;
    for (uint32_t i = 0; i < iters; ++i) {
        /* theta^(-2i/head_dim) == exp2(-2i*log2(theta)/head_dim) */
        const float inv = (float)(2.0 * (double)i) / (float)head_dim;
        const float freq = (float)pos * freq_scale * exp2f(-inv * log2f(theta_base));
        cos_out[i] = cosf(freq);
        sin_out[i] = sinf(freq);
    }
}

void vg_cpu_rope_apply(float *v, uint32_t head_dim, uint32_t n_rot, const float *cos_tab, const float *sin_tab) {
    const uint32_t half = head_dim / 2u;
    const uint32_t iters = n_rot < half ? n_rot : half;
    for (uint32_t i = 0; i < iters; ++i) {
        const float c = cos_tab[i], s = sin_tab[i];
        const float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

void vg_cpu_rope_apply_all(float *v, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                           const float *cos_tab, const float *sin_tab) {
    for (uint32_t h = 0; h < n_head; ++h) vg_cpu_rope_apply(v + (size_t)h * head_dim, head_dim, n_rot, cos_tab, sin_tab);
}

void vg_cpu_rope(float *q, float *k, uint32_t head_dim, uint32_t n_rot,
                 float freq_scale, float theta_base, int32_t pos) {
    const uint32_t half = head_dim / 2u;
    const uint32_t iters = n_rot < half ? n_rot : half;
    for (uint32_t i = 0; i < iters; ++i) {
        const float inv = (float)(2.0 * (double)i) / (float)head_dim;
        const float freq = (float)pos * freq_scale * exp2f(-inv * log2f(theta_base));
        const float c = cosf(freq), s = sinf(freq);
        const float q0 = q[i], q1 = q[i + half];
        q[i] = q0 * c - q1 * s;
        q[i + half] = q0 * s + q1 * c;
        if (k && k != q) {
            const float k0 = k[i], k1 = k[i + half];
            k[i] = k0 * c - k1 * s;
            k[i + half] = k0 * s + k1 * c;
        }
    }
}

void vg_cpu_softmax(float *x, uint32_t dim) {
    if (!dim) return;
    float mx = x[0];
    for (uint32_t i = 1; i < dim; ++i) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) { x[i] = expf(x[i] - mx); s += x[i]; }
    if (s > 0.0f) {
        const float inv = 1.0f / s;
        for (uint32_t i = 0; i < dim; ++i) x[i] *= inv;
    }
}

void vg_cpu_swiglu(float *out, const float *gate, const float *up, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void vg_cpu_gelu(float *x, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void vg_cpu_add(float *out, const float *a, const float *b, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) out[i] = a[i] + b[i];
}

void vg_cpu_scale(float *x, uint32_t dim, float scale) {
    for (uint32_t i = 0; i < dim; ++i) x[i] *= scale;
}

float vg_cpu_dot(const float *a, const float *b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) acc += a[i] * b[i];
    return acc;
}
