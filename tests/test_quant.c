#include "vg/quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static void test_legacy(void) {
    unsigned char q4[18] = {0x00, 0x3c}; memset(q4 + 2, 0x88, 16); float out[32]; assert(vg_dequantize_row(VG_QUANT_Q4_0, q4, sizeof(q4), out, 32) == VG_OK); for (size_t i = 0; i < 32; ++i) assert(fabsf(out[i]) < 1e-6f);
    unsigned char q8[34] = {0x00, 0x3c}; for (size_t i = 0; i < 32; ++i) q8[2 + i] = (unsigned char)(i + 1); float x[32]; for (size_t i = 0; i < 32; ++i) x[i] = 1.0f; float dot = 0; assert(vg_quantized_dot(VG_QUANT_Q8_0, q8, sizeof(q8), x, 32, &dot) == VG_OK); assert(fabsf(dot - 528.0f) < 1e-3f);
}
static void test_scalar(void) {
    uint16_t f16[2] = {0x3c00, 0xc000}; float out[2]; assert(vg_dequantize_row(VG_QUANT_F16, f16, sizeof(f16), out, 2) == VG_OK); assert(fabsf(out[0] - 1.0f) < 1e-6f && fabsf(out[1] + 2.0f) < 1e-6f);
    assert(vg_quant_validate(VG_QUANT_Q2_0, 64, 18) == VG_OK); assert(vg_quant_validate(VG_QUANT_Q2_0, 32, 18) != VG_OK); assert(vg_quant_info(VG_QUANT_Q6_K)->bytes_per_block == 210);
}
static void test_q4_k(void) {
    /* Q4_K: all values should dequantize to 1.0
     * Block: d=1.0(fp16), dmin=0.0(fp16), scales all 1, qs all 1 (4-bit)
     * Scales encoding: j=0..3 use scales[j]&63 and scales[j+4]&63
     *                  j=4..7 use different bit packing of scales[8..11] */
    unsigned char q4k[144];
    memset(q4k, 0, sizeof(q4k));
    *((uint16_t *)(q4k + 0)) = 0x3c00;   /* d = 1.0 */
    *((uint16_t *)(q4k + 2)) = 0x0000;   /* dmin = 0.0 */
    q4k[4] = 1; q4k[5] = 1; q4k[6] = 1; q4k[7] = 1;  /* d for j=0..3 */
    q4k[8] = 0; q4k[9] = 0; q4k[10] = 0; q4k[11] = 0; /* m for j=0..3 */
    q4k[12] = 1; q4k[13] = 1; q4k[14] = 1; q4k[15] = 1; /* d for j=4..7 */
    memset(q4k + 16, 0x11, 128); /* qs: all 4-bit values = 1 */
    float out[256];
    assert(vg_dequantize_row(VG_QUANT_Q4_K, q4k, sizeof(q4k), out, 256) == VG_OK);
    for (size_t i = 0; i < 256; ++i) {
        assert(fabsf(out[i] - 1.0f) < 1e-4f);
    }
}
static void test_q6_k(void) {
    /* Q6_K: all values should dequantize to 1.0
     * Block layout: ql[128], qh[64], scales[16], d(fp16) */
    unsigned char q6[210];
    memset(q6, 0x11, 128);     /* ql: all nibbles = 1 */
    memset(q6 + 128, 0xAA, 64); /* qh: all 2-bit fields = 2 */
    memset(q6 + 192, 1, 16);    /* scales: all = 1 */
    *((uint16_t *)(q6 + 208)) = 0x3c00; /* d = 1.0 (fp16) */

    float out[256];
    assert(vg_dequantize_row(VG_QUANT_Q6_K, q6, sizeof(q6), out, 256) == VG_OK);
    for (size_t i = 0; i < 256; ++i) {
        assert(fabsf(out[i] - 1.0f) < 1e-5f);
    }
}
int main(void) { test_legacy(); test_scalar(); test_q4_k(); test_q6_k(); return 0; }
