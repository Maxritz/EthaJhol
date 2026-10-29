/* CPU matvec benchmark: reports GFLOP/s for the fused quantized kernels is not meaningful for a
 * matvec (which is memory bound), so this reports the metric that actually predicts decode speed:
 * packed-weight bytes streamed per second, plus the derived token rate for a dense 7B-shaped
 * matvec set. Run with no arguments for the default sweep.
 *
 *   vg-mmbench [out_dim] [in_dim] [iters]
 */
#include "vg/cpu_ops.h"
#include "vg/quant.h"
#include "vg/trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct VG_MMRow {
    uint32_t type;
    const char *name;
} VG_MMRow;

static uint16_t f32_to_f16_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = (bits >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/* Deterministic pseudo-random payload with a sane block scale so the values are normal floats
 * (denormals would distort the timing on some cores). */
static void fill_packed(uint32_t type, unsigned char *packed, size_t blocks, size_t block_bytes) {
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < blocks * block_bytes; ++i) {
        s = s * 1664525u + 1013904223u;
        packed[i] = (unsigned char)(s >> 24);
    }
    const uint16_t half = f32_to_f16_bits(0.03125f);
    const unsigned char lo = (unsigned char)(half & 0xFFu), hi = (unsigned char)(half >> 8);
    size_t d_off = SIZE_MAX, m_off = SIZE_MAX;
    switch (type) {
        case VG_QUANT_Q4_0: case VG_QUANT_Q5_0: case VG_QUANT_Q8_0: case VG_QUANT_IQ4_NL: d_off = 0; break;
        case VG_QUANT_Q4_1: case VG_QUANT_Q5_1: d_off = 0; m_off = 2; break;
        case VG_QUANT_Q4_K: d_off = 0; m_off = 2; break;
        case VG_QUANT_Q5_K: d_off = 0; m_off = 2; break;
        case VG_QUANT_Q6_K: d_off = 208; break;
        case VG_QUANT_Q2_K: d_off = 80; m_off = 82; break;
        case VG_QUANT_Q3_K: d_off = 108; break;
        default: break;
    }
    if (d_off == SIZE_MAX) return;
    for (size_t b = 0; b < blocks; ++b) {
        unsigned char *blk = packed + b * block_bytes;
        blk[d_off] = lo;
        blk[d_off + 1] = hi;
        if (m_off != SIZE_MAX) { blk[m_off] = lo; blk[m_off + 1] = hi; }
    }
}

static void bench_type(uint32_t type, const char *label, uint32_t out_dim, uint32_t in_dim, int iters) {
    const VG_QuantInfo *qi = vg_quant_info(type);
    if (!qi || !qi->supported_scalar) return;
    if (in_dim % qi->block_size) {
        printf("%-8s skipped: in_dim %u is not a multiple of block %u\n", label, in_dim, qi->block_size);
        return;
    }
    const size_t blocks = (size_t)out_dim * (in_dim / qi->block_size);
    const size_t bytes = blocks * qi->bytes_per_block;
    unsigned char *packed = (unsigned char *)malloc(bytes);
    float *x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *y = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!packed || !x || !y) { free(packed); free(x); free(y); return; }
    fill_packed(type, packed, blocks, qi->bytes_per_block);
    for (uint32_t i = 0; i < in_dim; ++i) x[i] = (float)((int)(i % 17) - 8) / 16.0f;

    /* Warm-up: page the weight buffer in and size the thread pool. */
    (void)vg_cpu_matmul(type, packed, x, y, out_dim, in_dim);

    /* Per-iteration timing with the MINIMUM reported. On a shared cloud vCPU the same
     * workload varied by 35% run to run (Q4_0 5.8-8.2 ms/op), which is enough to hide a real
     * 15% kernel change; the minimum is the least noisy estimator of the hardware ceiling. */
    double best = 0.0, total = 0.0;
    for (int it = 0; it < iters; ++it) {
        const uint64_t t0 = vg_trace_now_ns();
        (void)vg_cpu_matmul(type, packed, x, y, out_dim, in_dim);
        const uint64_t t1 = vg_trace_now_ns();
        const double ms = (double)(t1 - t0) / 1e6;
        if (best == 0.0 || ms < best) best = ms;
        total += ms;
    }
    const double secs = best / 1000.0;
    const double gbytes = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    const double flops = 2.0 * (double)out_dim * (double)in_dim * (double)iters;
    printf("%-8s %6.2f MiB  %8.2f ms/op (best)  %7.2f ms (mean)  %7.2f GiB/s  %6.2f GFLOP/s  (%.3f MiB read per token)\n",
           label, (double)bytes / (1024.0 * 1024.0), best,
           total / (double)iters, gbytes / secs, flops / secs / 1e9,
           (double)bytes / (1024.0 * 1024.0));
    free(packed);
    free(x);
    free(y);
}

int main(int argc, char **argv) {
    uint32_t out_dim = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 4096;
    uint32_t in_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 4096;
    int iters = argc > 3 ? atoi(argv[3]) : 10;
    if (out_dim < 32 || in_dim < 32 || iters < 1) {
        fprintf(stderr, "usage: %s [out_dim>=32] [in_dim>=32] [iters>=1]\n", argv[0]);
        return 1;
    }

    const int threads = vg_cpu_set_threads(0);
    printf("# vg-mmbench %ux%u, %d iterations, %d thread(s)\n", out_dim, in_dim, iters, threads);
    bench_type(VG_QUANT_Q4_0, "Q4_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q8_0, "Q8_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q4_K, "Q4_K", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q6_K, "Q6_K", out_dim, in_dim, iters);
    /* Ternary families: the whole point of 1.58-bit weights is fewer bytes streamed per token. */
    bench_type(VG_QUANT_TQ2_0, "TQ2_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_TQ1_0, "TQ1_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q2_0, "Q2_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q1_0, "Q1_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_F32, "F32", out_dim, in_dim, iters);
    bench_type(VG_QUANT_F16, "F16", out_dim, in_dim, iters);

    /* Scaling check: the same Q4_0 workload on one thread shows how much row parallelism buys. */
    const int single = vg_cpu_set_threads(1);
    printf("# vg-mmbench single-thread (%d)\n", single);
    bench_type(VG_QUANT_Q4_0, "Q4_0", out_dim, in_dim, iters);
    bench_type(VG_QUANT_Q6_K, "Q6_K", out_dim, in_dim, iters);
    vg_cpu_set_threads(0);
    return 0;
}
