#include "vg/quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } block_q4_1;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
typedef struct { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
typedef struct { uint16_t d; uint8_t qs[64]; } block_q2_0;
typedef struct { uint16_t d; uint8_t qs[32]; } block_q3_0;
typedef struct { uint16_t d; uint8_t qs[64]; } block_q4_0_64;
typedef struct { uint16_t d; uint8_t qs[64]; } block_q5_0_64;
typedef struct { uint16_t d; uint8_t qs[128]; } block_q6_0_256;

static const VG_QuantInfo g_info[] = {
    {VG_QUANT_F32, "F32", 1, 4, 4, 1, 1}, {VG_QUANT_F16, "F16", 1, 2, 2, 1, 1},
    {VG_QUANT_Q4_0, "Q4_0", 32, 18, 32, 1, 1}, {VG_QUANT_Q4_1, "Q4_1", 32, 20, 32, 1, 1},
    {VG_QUANT_F64, "F64", 1, 8, 8, 1, 0},
    {VG_QUANT_Q5_0, "Q5_0", 32, 22, 32, 1, 1}, {VG_QUANT_Q5_1, "Q5_1", 32, 24, 32, 1, 1},
    {VG_QUANT_Q8_0, "Q8_0", 32, 34, 32, 1, 1}, {VG_QUANT_Q8_1, "Q8_1", 32, 36, 32, 0, 0},
    {VG_QUANT_Q2_K, "Q2_K", 256, 84, 256, 0, 1},
    {VG_QUANT_Q3_K, "Q3_K", 256, 110, 256, 0, 1}, {VG_QUANT_Q4_K, "Q4_K", 256, 144, 256, 0, 1},
    {VG_QUANT_Q5_K, "Q5_K", 256, 176, 256, 0, 1}, {VG_QUANT_Q6_K, "Q6_K", 256, 210, 256, 0, 1},
    {VG_QUANT_Q8_K, "Q8_K", 256, 292, 256, 0, 1},     {VG_QUANT_IQ2_XXS, "IQ2_XXS", 256, 66, 256, 0, 0},
    {VG_QUANT_IQ2_XS, "IQ2_XS", 256, 74, 256, 0, 0},
    {VG_QUANT_IQ3_XXS, "IQ3_XXS", 256, 98, 256, 0, 0},
    {VG_QUANT_IQ1_S, "IQ1_S", 256, 50, 256, 0, 0},
    {VG_QUANT_IQ4_NL, "IQ4_NL", 32, 18, 32, 0, 0},
    {VG_QUANT_IQ3_S, "IQ3_S", 256, 110, 256, 0, 0},
    {VG_QUANT_IQ2_S, "IQ2_S", 256, 82, 256, 0, 0},
    {VG_QUANT_IQ4_XS, "IQ4_XS", 256, 136, 256, 0, 0},
    {VG_QUANT_I8, "I8", 1, 1, 1, 1, 0},
    {VG_QUANT_I16, "I16", 1, 2, 2, 1, 0},
    {VG_QUANT_I32, "I32", 1, 4, 4, 1, 0},
    {VG_QUANT_I64, "I64", 1, 8, 8, 1, 0},
    {VG_QUANT_IQ1_M, "IQ1_M", 256, 56, 256, 0, 0},
    {VG_QUANT_BF16, "BF16", 1, 2, 2, 1, 1},
    {VG_QUANT_MXFP4, "MXFP4", 32, 17, 32, 0, 0},
    {VG_QUANT_NVFP4, "NVFP4", 64, 36, 64, 0, 0},
    {VG_QUANT_Q1_0, "Q1_0", 32, 18, 32, 0, 0},
    {VG_QUANT_Q2_0, "Q2_0", 32, 18, 32, 0, 0}
};
static const VG_QuantInfo *find(uint32_t type) { for (size_t i = 0; i < sizeof(g_info)/sizeof(g_info[0]); ++i) if ((uint32_t)g_info[i].kind == type) return &g_info[i]; return NULL; }
const VG_QuantInfo *vg_quant_info(uint32_t type) { return find(type); }
VG_Status vg_quant_validate(uint32_t type, uint64_t elements, uint64_t bytes) { const VG_QuantInfo *q = find(type); if (!q || !elements || elements % q->block_size || elements / q->block_size > UINT64_MAX / q->bytes_per_block || elements / q->block_size * q->bytes_per_block != bytes) return VG_E_FORMAT; return VG_OK; }
static float half_to_float(uint16_t h) { uint32_t sign = (uint32_t)(h >> 15) << 31, exp = (h >> 10) & 31u, mant = h & 1023u, bits; if (!exp) bits = mant ? sign | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13) : sign; else if (exp == 31u) bits = sign | 0x7f800000u | (mant << 13); else bits = sign | ((exp + 112u) << 23) | (mant << 13); float f; memcpy(&f, &bits, sizeof(f)); return f; }
static float bf16_to_float(uint16_t h) { uint32_t bits = (uint32_t)h << 16; float f; memcpy(&f, &bits, sizeof(f)); return f; }
static void deq_q4_0(const unsigned char *src, float *y, size_t n) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const block_q4_0 *q = (const block_q4_0 *)(src + b * 18); float d = half_to_float(q->d); for (size_t j = 0; j < 16; ++j) { y[b*32+j] = ((int)(q->qs[j] & 15u) - 8) * d; y[b*32+16+j] = ((int)(q->qs[j] >> 4) - 8) * d; } } }
static void deq_q4_1(const unsigned char *src, float *y, size_t n) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const block_q4_1 *q = (const block_q4_1 *)(src + b * 20); float d = half_to_float(q->d), m = half_to_float(q->m); for (size_t j = 0; j < 16; ++j) { y[b*32+j] = (q->qs[j] & 15u) * d + m; y[b*32+16+j] = (q->qs[j] >> 4) * d + m; } } }
static void deq_q5_0(const unsigned char *src, float *y, size_t n) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const block_q5_0 *q = (const block_q5_0 *)(src + b * 22); uint32_t qh; memcpy(&qh, q->qh, 4); float d = half_to_float(q->d); for (size_t j = 0; j < 16; ++j) { int x0 = ((q->qs[j] & 15u) | (((qh >> j) & 1u) << 4)) - 16; int x1 = ((q->qs[j] >> 4) | (((qh >> (j + 12)) & 1u) << 4)) - 16; y[b*32+j] = x0*d; y[b*32+16+j] = x1*d; } } }
static void deq_q5_1(const unsigned char *src, float *y, size_t n) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const block_q5_1 *q = (const block_q5_1 *)(src + b * 24); uint32_t qh; memcpy(&qh, q->qh, 4); float d = half_to_float(q->d), m = half_to_float(q->m); for (size_t j = 0; j < 16; ++j) { int x0 = (q->qs[j] & 15u) | (((qh >> j) & 1u) << 4); int x1 = (q->qs[j] >> 4) | (((qh >> (j + 12)) & 1u) << 4); y[b*32+j] = x0*d + m; y[b*32+16+j] = x1*d + m; } } }
static void deq_q8_0(const unsigned char *src, float *y, size_t n) { size_t nb = n / 32; for (size_t b = 0; b < nb; ++b) { const block_q8_0 *q = (const block_q8_0 *)(src + b * 34); float d = half_to_float(q->d); for (size_t j = 0; j < 32; ++j) y[b*32+j] = q->qs[j] * d; } }
VG_Status vg_dequantize_row(uint32_t type, const void *packed, size_t bytes, float *out, size_t elements) {
    if (!packed || !out || !elements) return VG_E_INVALID; if (vg_quant_validate(type, elements, bytes) != VG_OK) return VG_E_FORMAT;
    if (type == VG_QUANT_F32) { memcpy(out, packed, elements * sizeof(float)); return VG_OK; }
    if (type == VG_QUANT_F16 || type == VG_QUANT_BF16) { const uint16_t *p = (const uint16_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = type == VG_QUANT_F16 ? half_to_float(p[i]) : bf16_to_float(p[i]); return VG_OK; }
    if (type == VG_QUANT_F64) { const double *p = (const double *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I8) { const int8_t *p = (const int8_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I16) { const int16_t *p = (const int16_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I32) { const int32_t *p = (const int32_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I64) { const int64_t *p = (const int64_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_Q4_0) deq_q4_0((const unsigned char *)packed, out, elements); else if (type == VG_QUANT_Q4_1) deq_q4_1((const unsigned char *)packed, out, elements); else if (type == VG_QUANT_Q5_0) deq_q5_0((const unsigned char *)packed, out, elements); else if (type == VG_QUANT_Q5_1) deq_q5_1((const unsigned char *)packed, out, elements); else if (type == VG_QUANT_Q8_0) deq_q8_0((const unsigned char *)packed, out, elements); else return VG_E_UNSUPPORTED;
    return VG_OK;
}
VG_Status vg_quantized_dot(uint32_t type, const void *packed, size_t bytes, const float *x, size_t elements, float *out) {
    if (!out || !x) return VG_E_INVALID; float *tmp = (float *)malloc(elements * sizeof(float)); if (!tmp) return VG_E_NOMEM; VG_Status st = vg_dequantize_row(type, packed, bytes, tmp, elements); if (st == VG_OK) { float acc = 0.0f; for (size_t i = 0; i < elements; ++i) acc += tmp[i] * x[i]; *out = acc; } free(tmp); return st;
}
