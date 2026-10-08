#include "vg/quant.h"
#include "vg/cpu_ops.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

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
    assert(vg_dequantize_row(VG_QUANT_Q4_0, w_q4, n_blocks * 18, w_deq, n_elements) == VG_OK);
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    /* Compare */
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-3f);
    
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
    assert(vg_dequantize_row(VG_QUANT_Q8_0, w_q8, n_blocks * 34, w_deq, n_elements) == VG_OK);
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-3f);
    
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
    assert(vg_dequantize_row(VG_QUANT_Q4_K, w_q4k, n_blocks * 144, w_deq, n_elements) == VG_OK);
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-3f);
    
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
    assert(vg_dequantize_row(VG_QUANT_Q6_K, w_q6k, n_blocks * 210, w_deq, n_elements) == VG_OK);
    vg_cpu_matmul_f32(w_deq, x, y_ref, out_dim, in_dim);
    
    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_dim; ++i) {
        float diff = fabsf(y_fused[i] - y_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-3f);
    
    free(w_f32); free(x); free(y_fused); free(y_ref); free(w_q6k); free(w_deq);
}
int main(void) { test_legacy(); test_scalar(); test_q4_k(); test_q6_k(); test_matmul_roundtrip_q4_0(); test_matmul_roundtrip_q8_0(); test_matmul_roundtrip_q4_k(); test_matmul_roundtrip_q6_k(); return 0; }
