#include "vg/quant.h"
#include "vg/cpu_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same reasoning as test_core.c: every check and every call used to sit inside assert(), so a
 * Release build compiled the whole suite away. The unused-variable warnings were the tell. */
static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
} while (0)

#define CHECK_OK(expr) do { \
    VG_Status s_ = (expr); \
    if (s_ != VG_OK) { fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, (int)s_); ++failures; } \
} while (0)

static void test_legacy(void) {
    /* Q4_0: d = 1.0, every nibble 8, which is the zero point -> all outputs 0. */
    unsigned char q4[18] = { 0x00, 0x3c };
    memset(q4 + 2, 0x88, 16);
    float out[32];
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q4_0, q4, sizeof(q4), out, 32));
    for (size_t i = 0; i < 32; ++i) CHECK(fabsf(out[i]) < 1e-6f);

    /* Q8_0: d = 1.0, quants 1..32, dot with all-ones -> sum 1..32 == 528. */
    unsigned char q8[34] = { 0x00, 0x3c };
    for (size_t i = 0; i < 32; ++i) q8[2 + i] = (unsigned char)(i + 1);
    float x[32];
    for (size_t i = 0; i < 32; ++i) x[i] = 1.0f;
    float dot = 0.0f;
    CHECK_OK(vg_quantized_dot(VG_QUANT_Q8_0, q8, sizeof(q8), x, 32, &dot));
    CHECK(fabsf(dot - 528.0f) < 1e-3f);
}

static void test_scalar(void) {
    uint16_t f16[2] = { 0x3c00, 0xc000 };   /* 1.0, -2.0 */
    float out[2];
    CHECK_OK(vg_dequantize_row(VG_QUANT_F16, f16, sizeof(f16), out, 2));
    CHECK(fabsf(out[0] - 1.0f) < 1e-6f);
    CHECK(fabsf(out[1] + 2.0f) < 1e-6f);

    CHECK_OK(vg_quant_validate(VG_QUANT_Q2_0, 64, 18));
    CHECK(vg_quant_validate(VG_QUANT_Q2_0, 32, 18) != VG_OK);

    const VG_QuantInfo *q6k = vg_quant_info(VG_QUANT_Q6_K);
    CHECK(q6k != NULL);
    if (q6k) CHECK(q6k->bytes_per_block == 210);
}

static void test_q4_k(void) {
    /* d = 1.0, dmin = 0, all sub-scales 1, all 4-bit quants 1 -> every value is 1.0. */
    unsigned char q4k[144];
    memset(q4k, 0, sizeof(q4k));
    memcpy(q4k + 0, "\x00\x3c", 2);   /* d = fp16 1.0 */
    memcpy(q4k + 2, "\x00\x00", 2);   /* dmin = 0 */
    q4k[4] = q4k[5] = q4k[6] = q4k[7] = 1;
    q4k[8] = q4k[9] = q4k[10] = q4k[11] = 0;
    q4k[12] = q4k[13] = q4k[14] = q4k[15] = 1;
    memset(q4k + 16, 0x11, 128);

    float out[256];
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q4_K, q4k, sizeof(q4k), out, 256));
    for (size_t i = 0; i < 256; ++i) CHECK(fabsf(out[i] - 1.0f) < 1e-4f);
}

static void test_q6_k(void) {
    /* ql nibbles 1, qh 2-bit fields 2, scales 1, d = 1.0 -> composite value 1.0. */
    unsigned char q6[210];
    memset(q6, 0x11, 128);
    memset(q6 + 128, 0xAA, 64);
    memset(q6 + 192, 1, 16);
    memcpy(q6 + 208, "\x00\x3c", 2);   /* d = fp16 1.0 */

    float out[256];
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q6_K, q6, sizeof(q6), out, 256));
    for (size_t i = 0; i < 256; ++i) CHECK(fabsf(out[i] - 1.0f) < 1e-5f);
}

/* --- Ternary / 1-bit types -------------------------------------------------------
 * The expected values below come from an independent Python port of ggml's
 * dequantize_row_tq1_0 / tq2_0 / q1_0 / q2_0 over the same byte patterns, not from this
 * library. Both the first eight values (plane ORDER) and the position-weighted sum
 * (element MAPPING) are checked, because a decoder that walks the bytes linearly gets the
 * right multiset in the wrong places and would otherwise pass a plain sum comparison. */
static void build_ternary_blocks(unsigned char *tq1, unsigned char *tq2, unsigned char *s16) {
    /* d = 1.0 for TQ1_0, 0.5 for TQ2_0, 2.0 for Q1_0, 0.25 for Q2_0 (see callers). */
    for (int i = 0; i < 48; ++i) tq1[i] = (unsigned char)(i * 37 + 11);
    tq1[48] = 0x1B; tq1[49] = 0xC7; tq1[50] = 0x00; tq1[51] = 0xFF;
    for (int i = 0; i < 64; ++i) tq2[i] = (unsigned char)(i * 29 + 7);
    /* Q1_0 / Q2_0 put the scale FIRST (d, qs[16]); TQ1_0 / TQ2_0 put it last. */
    s16[0] = 0; s16[1] = 0;
    for (int i = 0; i < 16; ++i) s16[2 + i] = (unsigned char)(i * 53 + 3);
}

static void check_golden(const char *label, uint32_t type, const unsigned char *block, size_t bytes,
                         size_t n, const float *first8, float expect_sum, float expect_possum) {
    float out[256];
    if (n > sizeof(out) / sizeof(out[0])) { fprintf(stderr, "FAIL: %s too many elements\n", label); ++failures; return; }
    VG_Status st = vg_dequantize_row(type, block, bytes, out, n);
    if (st != VG_OK) { fprintf(stderr, "FAIL: %s dequantize -> %d\n", label, (int)st); ++failures; return; }
    for (size_t i = 0; i < 8; ++i) {
        if (fabsf(out[i] - first8[i]) > 1e-6f) {
            fprintf(stderr, "FAIL: %s[%zu] = %.6f, expected %.6f\n", label, i, out[i], first8[i]);
            ++failures;
        }
    }
    double sum = 0.0, possum = 0.0;
    for (size_t i = 0; i < n; ++i) { sum += out[i]; possum += (double)i * out[i]; }
    if (fabs(sum - expect_sum) > 1e-2) {
        fprintf(stderr, "FAIL: %s sum = %.4f, expected %.4f\n", label, sum, expect_sum);
        ++failures;
    }
    if (fabs(possum - expect_possum) > 1e-1) {
        fprintf(stderr, "FAIL: %s position-weighted sum = %.4f, expected %.4f (element order is wrong)\n",
                label, possum, expect_possum);
        ++failures;
    }
}

static void test_ternary(void) {
    unsigned char tq1[54], tq2[66], s16[18];
    build_ternary_blocks(tq1, tq2, s16);
    memset(tq1 + 52, 0, 2); memcpy(tq1 + 52, "\x00\x3c", 2);   /* d = 1.0   */
    memset(tq2 + 64, 0, 2); memcpy(tq2 + 64, "\x00\x38", 2);   /* d = 0.5   */

    /* Block geometry has to match ggml exactly or every offset downstream is wrong. */
    CHECK_OK(vg_quant_validate(VG_QUANT_TQ1_0, 256, 54));
    CHECK_OK(vg_quant_validate(VG_QUANT_TQ2_0, 256, 66));
    CHECK_OK(vg_quant_validate(VG_QUANT_Q1_0, 128, 18));
    CHECK_OK(vg_quant_validate(VG_QUANT_Q2_0, 64, 18));
    CHECK(vg_quant_validate(VG_QUANT_TQ2_0, 256, 54) != VG_OK);
    CHECK(vg_quant_validate(VG_QUANT_Q1_0, 32, 18) != VG_OK);   /* Q1_0 is 128 weights/block */

    const float tq1_first[8] = { -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f };
    check_golden("TQ1_0", VG_QUANT_TQ1_0, tq1, sizeof(tq1), 256, tq1_first, 1.0f, 648.0f);

    const float tq2_first[8] = { 1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -0.5f, 0.0f, 0.5f };
    check_golden("TQ2_0", VG_QUANT_TQ2_0, tq2, sizeof(tq2), 256, tq2_first, 62.5f, 8139.0f);

    /* Q1_0: d = 2.0. */
    s16[0] = 0x00; s16[1] = 0x40;
    const float q1_first[8] = { 2.0f, 2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f, -2.0f };
    check_golden("Q1_0", VG_QUANT_Q1_0, s16, sizeof(s16), 128, q1_first, -8.0f, 556.0f);

    /* Q2_0: d = 0.25. */
    s16[0] = 0x00; s16[1] = 0x34;
    const float q2_first[8] = { 0.5f, -0.25f, -0.25f, -0.25f, -0.25f, 0.25f, 0.5f, -0.25f };
    check_golden("Q2_0", VG_QUANT_Q2_0, s16, sizeof(s16), 64, q2_first, 7.25f, 276.5f);

    /* Ternary values must never leave {-1,0,+1}*d: catches a radix-3 packer that
     * accidentally acquires a +2 code from the (*3)>>8 trick. */
    float out[256];
    CHECK_OK(vg_dequantize_row(VG_QUANT_TQ1_0, tq1, sizeof(tq1), out, 256));
    for (size_t i = 0; i < 256; ++i) {
        if (fabsf(out[i]) > 1e-6f && fabsf(fabsf(out[i]) - 1.0f) > 1e-6f) {
            fprintf(stderr, "FAIL: TQ1_0[%zu] = %.6f is not ternary\n", i, out[i]);
            ++failures;
        }
    }
    CHECK_OK(vg_dequantize_row(VG_QUANT_TQ2_0, tq2, sizeof(tq2), out, 256));
    for (size_t i = 0; i < 256; ++i) {
        const float v = out[i] * 2.0f;   /* d = 0.5 -> {-1,0,1,2} */
        if (v < -1.5f || v > 2.5f) { fprintf(stderr, "FAIL: TQ2_0[%zu] = %.6f out of range\n", i, out[i]); ++failures; }
    }
}

/* Matmul round-trip tests: fused kernel vs dequantize + f32 */
static void test_matmul_roundtrip_q4_0(void) {
    const uint32_t out_dim = 64, in_dim = 128;
    const size_t n_elements = (size_t)out_dim * in_dim;
    const size_t n_blocks = n_elements / 32;
    
    float *w_f32 = (float *)malloc(n_elements * sizeof(float));
    float *x = (float *)malloc(in_dim * sizeof(float));
    float *y_fused = (float *)calloc(out_dim, sizeof(float));
    float *y_ref = (float *)calloc(out_dim, sizeof(float));
    unsigned char *w_q4 = (unsigned char *)malloc(n_blocks * 18);
    
    srand(0xC0FFEE);
    for (size_t i = 0; i < n_elements; ++i) w_f32[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    for (size_t i = 0; i < in_dim; ++i) x[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    
    /* Quantize to Q4_0 using dequantize logic in reverse - construct valid Q4_0 blocks */
    for (size_t b = 0; b < n_blocks; ++b) {
        float d = 0.1f;
        uint16_t d_fp16 = 0x3c00; /* 1.0 in fp16 */
        w_q4[b * 18 + 0] = (unsigned char)(d_fp16 & 0xFF);
        w_q4[b * 18 + 1] = (unsigned char)(d_fp16 >> 8);
        for (size_t j = 0; j < 16; ++j) {
            float v0 = w_f32[b * 32 + j];
            float v1 = w_f32[b * 32 + 16 + j];
            int q0 = (int)roundf(v0 / d + 8.0f);
            int q1 = (int)roundf(v1 / d + 8.0f);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            w_q4[b * 18 + 2 + j] = (unsigned char)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    
    /* Fused matmul */
    vg_cpu_matmul_q4_0(w_q4, x, y_fused, out_dim, in_dim);
    
    /* Reference: dequantize + f32 matmul */
    float *w_deq = (float *)malloc(n_elements * sizeof(float));
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q4_0, w_q4, n_blocks * 18, w_deq, n_elements));
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    /* Compare */
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff >= 1e-3f) { fprintf(stderr, "FAIL: fused matmul drifted %.6f from dequantized reference\n", (double)max_diff); ++failures; }
    
    free(w_f32); free(x); free(y_fused); free(y_ref); free(w_q4); free(w_deq);
}

static void test_matmul_roundtrip_q8_0(void) {
    const uint32_t out_dim = 32, in_dim = 64;
    const size_t n_elements = (size_t)out_dim * in_dim;
    const size_t n_blocks = n_elements / 32;
    
    float *w_f32 = (float *)malloc(n_elements * sizeof(float));
    float *x = (float *)malloc(in_dim * sizeof(float));
    float *y_fused = (float *)calloc(out_dim, sizeof(float));
    float *y_ref = (float *)calloc(out_dim, sizeof(float));
    unsigned char *w_q8 = (unsigned char *)malloc(n_blocks * 34);
    
    srand(0xC0FFEE);
    for (size_t i = 0; i < n_elements; ++i) w_f32[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    for (size_t i = 0; i < in_dim; ++i) x[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    
    for (size_t b = 0; b < n_blocks; ++b) {
        uint16_t d_fp16 = 0x3c00;
        w_q8[b * 34 + 0] = (unsigned char)(d_fp16 & 0xFF);
        w_q8[b * 34 + 1] = (unsigned char)(d_fp16 >> 8);
        float d = 1.0f;
        for (size_t j = 0; j < 32; ++j) {
            float v = w_f32[b * 32 + j];
            int q = (int)roundf(v / d + 128.0f);
            q = q < 0 ? 0 : (q > 255 ? 255 : q);
            w_q8[b * 34 + 2 + j] = (unsigned char)q;
        }
    }
    
    vg_cpu_matmul_q8_0(w_q8, x, y_fused, out_dim, in_dim);
    
    float *w_deq = (float *)malloc(n_elements * sizeof(float));
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q8_0, w_q8, n_blocks * 34, w_deq, n_elements));
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff >= 1e-3f) { fprintf(stderr, "FAIL: fused matmul drifted %.6f from dequantized reference\n", (double)max_diff); ++failures; }
    
    free(w_f32); free(x); free(y_fused); free(y_ref); free(w_q8); free(w_deq);
}

static void test_matmul_roundtrip_q4_k(void) {
    const uint32_t out_dim = 32, in_dim = 256;
    const size_t n_elements = (size_t)out_dim * in_dim;
    const size_t n_blocks = n_elements / 256;
    
    float *w_f32 = (float *)malloc(n_elements * sizeof(float));
    float *x = (float *)malloc(in_dim * sizeof(float));
    float *y_fused = (float *)calloc(out_dim, sizeof(float));
    float *y_ref = (float *)calloc(out_dim, sizeof(float));
    unsigned char *w_q4k = (unsigned char *)malloc(n_blocks * 144);
    
    srand(0xC0FFEE);
    for (size_t i = 0; i < n_elements; ++i) w_f32[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    for (size_t i = 0; i < in_dim; ++i) x[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    
    for (size_t b = 0; b < n_blocks; ++b) {
        uint16_t d_fp16 = 0x3c00; /* 1.0 */
        uint16_t dmin_fp16 = 0x0000; /* 0.0 */
        w_q4k[b * 144 + 0] = (unsigned char)(d_fp16 & 0xFF);
        w_q4k[b * 144 + 1] = (unsigned char)(d_fp16 >> 8);
        w_q4k[b * 144 + 2] = (unsigned char)(dmin_fp16 & 0xFF);
        w_q4k[b * 144 + 3] = (unsigned char)(dmin_fp16 >> 8);
        for (size_t j = 0; j < 12; ++j) w_q4k[b * 144 + 4 + j] = 1; /* scales */
        for (size_t j = 0; j < 128; ++j) w_q4k[b * 144 + 16 + j] = 0x11; /* qs = 1 */
    }
    
    vg_cpu_matmul_q4_k(w_q4k, x, y_fused, out_dim, in_dim);
    
    float *w_deq = (float *)malloc(n_elements * sizeof(float));
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q4_K, w_q4k, n_blocks * 144, w_deq, n_elements));
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff >= 1e-3f) { fprintf(stderr, "FAIL: fused matmul drifted %.6f from dequantized reference\n", (double)max_diff); ++failures; }
    
    free(w_f32); free(x); free(y_fused); free(y_ref); free(w_q4k); free(w_deq);
}

static void test_matmul_roundtrip_q6_k(void) {
    const uint32_t out_dim = 32, in_dim = 256;
    const size_t n_elements = (size_t)out_dim * in_dim;
    const size_t n_blocks = n_elements / 256;
    
    float *w_f32 = (float *)malloc(n_elements * sizeof(float));
    float *x = (float *)malloc(in_dim * sizeof(float));
    float *y_fused = (float *)calloc(out_dim, sizeof(float));
    float *y_ref = (float *)calloc(out_dim, sizeof(float));
    unsigned char *w_q6k = (unsigned char *)malloc(n_blocks * 210);
    
    srand(0xC0FFEE);
    for (size_t i = 0; i < n_elements; ++i) w_f32[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    for (size_t i = 0; i < in_dim; ++i) x[i] = (float)(rand() / (double)RAND_MAX - 0.5) * 2.0f;
    
    for (size_t b = 0; b < n_blocks; ++b) {
        memset(w_q6k + b * 210, 0x11, 128);  /* ql: all nibbles = 1 */
        memset(w_q6k + b * 210 + 128, 0xAA, 64); /* qh: all 2-bit = 2 */
        memset(w_q6k + b * 210 + 192, 1, 16); /* scales: all = 1 */
        uint16_t d_fp16 = 0x3c00; /* 1.0 */
        w_q6k[b * 210 + 208] = (unsigned char)(d_fp16 & 0xFF);
        w_q6k[b * 210 + 209] = (unsigned char)(d_fp16 >> 8);
    }
    
    vg_cpu_matmul_q6_k(w_q6k, x, y_fused, out_dim, in_dim);
    
    float *w_deq = (float *)malloc(n_elements * sizeof(float));
    CHECK_OK(vg_dequantize_row(VG_QUANT_Q6_K, w_q6k, n_blocks * 210, w_deq, n_elements));
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff >= 1e-3f) { fprintf(stderr, "FAIL: fused matmul drifted %.6f from dequantized reference\n", (double)max_diff); ++failures; }
    
    free(w_f32); free(x); free(y_fused); free(y_ref); free(w_q6k); free(w_deq);
}
int main(void) {
    test_matmul_roundtrip_q4_0();
    test_matmul_roundtrip_q8_0();
    test_matmul_roundtrip_q4_k();
    test_matmul_roundtrip_q6_k();
    test_legacy();
    test_scalar();
    test_q4_k();
    test_q6_k();
    test_ternary();
    if (failures) { fprintf(stderr, "vg-quant: %d failure(s)\n", failures); return 1; }
    puts("vg-quant: all tests passed");
    return 0;
}
