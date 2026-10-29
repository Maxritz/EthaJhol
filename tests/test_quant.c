#include "vg/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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

int main(void) {
    test_legacy();
    test_scalar();
    test_q4_k();
    test_q6_k();
    test_ternary();
    if (failures) { fprintf(stderr, "vg-quant: %d failure(s)\n", failures); return 1; }
    puts("vg-quant: all tests passed");
    return 0;
}
