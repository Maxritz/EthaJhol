#include "vg/quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* --- GGML block layouts (byte-for-byte, no padding) ------------------------------ */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;                       /* 18 B  */
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } block_q4_1;           /* 20 B  */
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;        /* 22 B  */
typedef struct { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1; /* 24 B */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;                        /* 34 B  */
typedef struct { uint16_t d; uint8_t qs[16]; } block_iq4_nl;                     /* 18 B  */

/* K-quants (QK_K = 256) */
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K; /* 84  */
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } block_q3_K; /* 110 */
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K; /* 144 */
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K; /* 176 */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K; /* 210 */
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;       /* 292 */

/* Ternary / 1-bit blocks (ggml-compatible byte layout). */
typedef struct { uint8_t qs[48]; uint8_t qh[4]; uint16_t d; } block_tq1_0;      /* 54 B / 256 */
typedef struct { uint8_t qs[64]; uint16_t d; } block_tq2_0;                     /* 66 B / 256 */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q1_0;                      /* 18 B / 128 */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q2_0;                      /* 18 B /  64 */

#define QK_K 256

/* `supported_vulkan` is true only for types that have a dedicated compute shader in
 * shaders/. Advertising a type the GPU cannot execute silently falls back to the CPU. */
static const VG_QuantInfo g_info[] = {
    {VG_QUANT_F32,   "F32",   1,   4,   1, 1, 1},
    {VG_QUANT_F16,   "F16",   1,   2,   1, 1, 1},
    {VG_QUANT_BF16,  "BF16",  1,   2,   1, 1, 0},
    {VG_QUANT_F64,   "F64",   1,   8,   1, 1, 0},
    {VG_QUANT_Q4_0,  "Q4_0",  32,  18,  32, 1, 1},
    {VG_QUANT_Q4_1,  "Q4_1",  32,  20,  32, 1, 0},
    {VG_QUANT_Q5_0,  "Q5_0",  32,  22,  32, 1, 0},
    {VG_QUANT_Q5_1,  "Q5_1",  32,  24,  32, 1, 0},
    {VG_QUANT_Q8_0,  "Q8_0",  32,  34,  32, 1, 1},
    {VG_QUANT_IQ4_NL,"IQ4_NL",32,  18,  32, 1, 0},
    {VG_QUANT_Q2_K,  "Q2_K",  256, 84,  256, 1, 0},
    {VG_QUANT_Q3_K,  "Q3_K",  256, 110, 256, 1, 0},
    {VG_QUANT_Q4_K,  "Q4_K",  256, 144, 256, 1, 0},
    {VG_QUANT_Q5_K,  "Q5_K",  256, 176, 256, 1, 0},
    {VG_QUANT_Q6_K,  "Q6_K",  256, 210, 256, 1, 0},
    {VG_QUANT_Q8_K,  "Q8_K",  256, 292, 256, 1, 0},
    {VG_QUANT_I8,    "I8",    1,   1,   1, 1, 0},
    {VG_QUANT_I16,   "I16",   1,   2,   1, 1, 0},
    {VG_QUANT_I32,   "I32",   1,   4,   1, 1, 0},
    {VG_QUANT_I64,   "I64",   1,   8,   1, 1, 0},
    /* Not implemented yet: reported so diagnostics can name the exact type. */
    {VG_QUANT_Q8_1,  "Q8_1",  32,  36,  32, 0, 0},
    {VG_QUANT_IQ2_XXS,"IQ2_XXS",256,66, 256, 0, 0},
    {VG_QUANT_IQ2_XS, "IQ2_XS", 256,74, 256, 0, 0},
    {VG_QUANT_IQ3_XXS,"IQ3_XXS",256,98, 256, 0, 0},
    {VG_QUANT_IQ1_S,  "IQ1_S",  256,50, 256, 0, 0},
    {VG_QUANT_IQ3_S,  "IQ3_S",  256,110,256, 0, 0},
    {VG_QUANT_IQ2_S,  "IQ2_S",  256,82, 256, 0, 0},
    {VG_QUANT_IQ4_XS, "IQ4_XS", 256,136,256, 0, 0},
    {VG_QUANT_IQ1_M,  "IQ1_M",  256,56, 256, 0, 0},
    {VG_QUANT_MXFP4,  "MXFP4",  32,  17, 32, 0, 0},
    {VG_QUANT_NVFP4,  "NVFP4",  64,  36, 64, 0, 0},
    /* Ternary / 1-bit. Q1_0 packs 128 weights into its 18-byte block (2 B scale + 16 B signs), not 32. */
    {VG_QUANT_TQ1_0,  "TQ1_0",  256, 54, 256, 1, 0},
    {VG_QUANT_TQ2_0,  "TQ2_0",  256, 66, 256, 1, 0},
    {VG_QUANT_Q1_0,   "Q1_0",   128, 18, 128, 1, 0},
    {VG_QUANT_Q2_0,   "Q2_0",   64,  18, 64, 1, 0}
};

static const VG_QuantInfo *find(uint32_t type) {
    for (size_t i = 0; i < sizeof(g_info) / sizeof(g_info[0]); ++i)
        if ((uint32_t)g_info[i].kind == type) return &g_info[i];
    return NULL;
}

const VG_QuantInfo *vg_quant_info(uint32_t type) { return find(type); }

VG_Status vg_quant_validate(uint32_t type, uint64_t elements, uint64_t bytes) {
    const VG_QuantInfo *q = find(type);
    if (!q || !elements || elements % q->block_size) return VG_E_FORMAT;
    if (elements / q->block_size > UINT64_MAX / q->bytes_per_block) return VG_E_FORMAT;
    return elements / q->block_size * q->bytes_per_block == bytes ? VG_OK : VG_E_FORMAT;
}

static float bf16_to_float(uint16_t h) { uint32_t bits = (uint32_t)h << 16; float f; memcpy(&f, &bits, sizeof(f)); return f; }

/* 6-bit scale/min unpacking shared by Q4_K and Q5_K. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else { *d = (uint8_t)((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

static void deq_q2_k(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q2_K *q = (const block_q2_K *)(src + b * 84u);
        const float d = vg_half_to_float(q->d), mn = vg_half_to_float(q->dmin);
        const uint8_t *qs = q->qs;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = q->scales[is++];
                float dl = d * (sc & 0x0F), ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float)((qs[l] >> shift) & 3u) - ml;
                sc = q->scales[is++];
                dl = d * (sc & 0x0F); ml = mn * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * (float)((qs[l + 16] >> shift) & 3u) - ml;
                shift += 2;
            }
            qs += 32;
        }
    }
}

static void deq_q3_k(const unsigned char *src, float *y, size_t nb) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0F0F0F0Fu;
    for (size_t b = 0; b < nb; ++b) {
        const block_q3_K *q = (const block_q3_K *)(src + b * 110u);
        const float d_all = vg_half_to_float(q->d);
        uint32_t aux[4];
        memcpy(aux, q->scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int8_t scales[16];
        for (int i = 0; i < 4; ++i) memcpy(scales + 4 * i, aux + i, 4);
        for (int j = 0; j < 16; ++j) scales[j] = (int8_t)(scales[j] - 32);
        const uint8_t *qs = q->qs, *hm = q->hmask;
        uint8_t m = 1;
        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                const float dl = d_all * (float)scales[is++];
                for (int l = 0; l < 16; ++l) {
                    int v = (int)((qs[l] >> shift) & 3u) - ((hm[l] & m) ? 0 : 4);
                    *y++ = dl * (float)v;
                }
                const float dl2 = d_all * (float)scales[is++];
                for (int l = 0; l < 16; ++l) {
                    int v = (int)((qs[l + 16] >> shift) & 3u) - ((hm[l + 16] & m) ? 0 : 4);
                    *y++ = dl2 * (float)v;
                }
                shift += 2; m = (uint8_t)(m << 1);
            }
            qs += 32;
        }
    }
}

static void deq_q4_k(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q4_K *q = (const block_q4_K *)(src + b * 144u);
        const uint8_t *qs = q->qs;
        const float d = vg_half_to_float(q->d), mn = vg_half_to_float(q->dmin);
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, q->scales, &sc, &m);
            const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, q->scales, &sc, &m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(qs[l] & 0x0Fu) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(qs[l] >> 4) - m2;
            qs += 32; is += 2;
        }
    }
}

static void deq_q5_k(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q5_K *q = (const block_q5_K *)(src + b * 176u);
        const float d = vg_half_to_float(q->d), mn = vg_half_to_float(q->dmin);
        const uint8_t *ql = q->qs, *qh = q->qh;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, q->scales, &sc, &m);
            const float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, q->scales, &sc, &m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0Fu) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void deq_q6_k(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q6_K *q = (const block_q6_K *)(src + b * 210u);
        const float d = vg_half_to_float(q->d);
        const uint8_t *ql = q->ql, *qh = q->qh;
        const int8_t *sc = q->scales;
        for (int n_off = 0; n_off < QK_K; n_off += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l + 0] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void deq_q8_k(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q8_K *q = (const block_q8_K *)(src + b * 292u);
        const float d = q->d;
        for (int j = 0; j < QK_K; ++j) *y++ = d * (float)q->qs[j];
    }
}

static void deq_q4_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q4_0 *q = (const block_q4_0 *)(src + b * 18u);
        const float d = vg_half_to_float(q->d);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 16u; ++j) {
            out[j] = (float)((int)(q->qs[j] & 0x0Fu) - 8) * d;
            out[16u + j] = (float)((int)(q->qs[j] >> 4) - 8) * d;
        }
    }
}
static void deq_q4_1(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q4_1 *q = (const block_q4_1 *)(src + b * 20u);
        const float d = vg_half_to_float(q->d), m = vg_half_to_float(q->m);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 16u; ++j) {
            out[j] = (float)(q->qs[j] & 0x0Fu) * d + m;
            out[16u + j] = (float)(q->qs[j] >> 4) * d + m;
        }
    }
}
static void deq_q5_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q5_0 *q = (const block_q5_0 *)(src + b * 22u);
        uint32_t qh; memcpy(&qh, q->qh, 4);
        const float d = vg_half_to_float(q->d);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 16u; ++j) {
            const int x0 = (int)((q->qs[j] & 0x0Fu) | (((qh >> j) & 1u) << 4)) - 16;
            const int x1 = (int)((q->qs[j] >> 4) | (((qh >> (j + 12u)) & 1u) << 4)) - 16;
            out[j] = (float)x0 * d;
            out[16u + j] = (float)x1 * d;
        }
    }
}
static void deq_q5_1(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q5_1 *q = (const block_q5_1 *)(src + b * 24u);
        uint32_t qh; memcpy(&qh, q->qh, 4);
        const float d = vg_half_to_float(q->d), m = vg_half_to_float(q->m);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 16u; ++j) {
            const int x0 = (int)((q->qs[j] & 0x0Fu) | (((qh >> j) & 1u) << 4));
            const int x1 = (int)((q->qs[j] >> 4) | (((qh >> (j + 12u)) & 1u) << 4));
            out[j] = (float)x0 * d + m;
            out[16u + j] = (float)x1 * d + m;
        }
    }
}
static void deq_q8_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q8_0 *q = (const block_q8_0 *)(src + b * 34u);
        const float d = vg_half_to_float(q->d);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 32u; ++j) out[j] = (float)q->qs[j] * d;
    }
}
/* IQ4_NL: 4-bit indices into a signed 16-entry codebook, one fp16 scale per 32 values. */
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};
static void deq_iq4_nl(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_iq4_nl *q = (const block_iq4_nl *)(src + b * 18u);
        const float d = vg_half_to_float(q->d);
        float *out = y + b * 32u;
        for (size_t j = 0; j < 16u; ++j) {
            out[j] = d * (float)kvalues_iq4nl[q->qs[j] & 0x0Fu];
            out[16u + j] = d * (float)kvalues_iq4nl[q->qs[j] >> 4];
        }
    }
}

/* --- Ternary / 1-bit decoding ---------------------------------------------------
 * Bit-exact ports of ggml's dequantize_row_tq1_0 / tq2_0 / q1_0 / q2_0, including their
 * element ORDER: the ternary packers interleave bit-planes, so a decoder that walks the
 * bytes linearly produces the right multiset in the wrong positions. */
static const uint8_t k_pow3[5] = { 1, 3, 9, 27, 81 };

/* 1.6875 bpw: 5 ternary values per byte (radix-3: byte*3^n, top bits via (*3)>>8),
 * plus 4 quaternary-ish values in qh consumed as a fourth plane. */
static void deq_tq1_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_tq1_0 *q = (const block_tq1_0 *)(src + b * 54u);
        const float d = vg_half_to_float(q->d);
        for (size_t j = 0; j < 32u; j += 32u) {
            for (size_t n = 0; n < 5u; ++n) {
                for (size_t m = 0; m < 32u; ++m) {
                    const uint8_t packed = (uint8_t)(q->qs[j + m] * k_pow3[n]);
                    const int xi = (int)(((uint16_t)packed * 3u) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        for (size_t j = 32u; j < 48u; j += 16u) {
            for (size_t n = 0; n < 5u; ++n) {
                for (size_t m = 0; m < 16u; ++m) {
                    const uint8_t packed = (uint8_t)(q->qs[j + m] * k_pow3[n]);
                    const int xi = (int)(((uint16_t)packed * 3u) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        for (size_t n = 0; n < 4u; ++n) {
            for (size_t j = 0; j < 4u; ++j) {
                const uint8_t packed = (uint8_t)(q->qh[j] * k_pow3[n]);
                const int xi = (int)(((uint16_t)packed * 3u) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
        }
    }
}

/* 2.0625 bpw: four 32-value planes per 32-byte group, codes {0,1,2,3} -> {-1,0,+1,+2}. */
static void deq_tq2_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_tq2_0 *q = (const block_tq2_0 *)(src + b * 66u);
        const float d = vg_half_to_float(q->d);
        for (size_t j = 0; j < 64u; j += 32u) {
            for (size_t l = 0; l < 4u; ++l) {
                for (size_t m = 0; m < 32u; ++m) {
                    const int v = (int)((q->qs[j + m] >> (l * 2u)) & 3u);
                    *y++ = (float)(v - 1) * d;
                }
            }
        }
    }
}

/* 1 bit per weight, LSB-first within each byte. */
static void deq_q1_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q1_0 *q = (const block_q1_0 *)(src + b * 18u);
        const float d = vg_half_to_float(q->d);
        float *out = y + b * 128u;
        for (size_t j = 0; j < 128u; ++j) {
            const int bit = (q->qs[j / 8u] >> (j % 8u)) & 1;
            out[j] = bit ? d : -d;
        }
    }
}

/* 2 bits per weight, codes {0,1,2,3} -> {-1,0,+1,+2}. */
static void deq_q2_0(const unsigned char *src, float *y, size_t nb) {
    for (size_t b = 0; b < nb; ++b) {
        const block_q2_0 *q = (const block_q2_0 *)(src + b * 18u);
        const float d = vg_half_to_float(q->d);
        for (int j = 0; j < 64; ++j) {
            const int v = (int)((q->qs[j / 4] >> ((j % 4) * 2)) & 3u);
            y[b * 64 + (size_t)j] = (float)(v - 1) * d;
        }
    }
}

typedef void (*deq_blocks_fn)(const unsigned char *, float *, size_t);

static deq_blocks_fn block_fn(uint32_t type) {
    switch (type) {
        case VG_QUANT_Q4_0: return deq_q4_0;
        case VG_QUANT_Q4_1: return deq_q4_1;
        case VG_QUANT_Q5_0: return deq_q5_0;
        case VG_QUANT_Q5_1: return deq_q5_1;
        case VG_QUANT_Q8_0: return deq_q8_0;
        case VG_QUANT_IQ4_NL: return deq_iq4_nl;
        case VG_QUANT_Q2_K: return deq_q2_k;
        case VG_QUANT_Q3_K: return deq_q3_k;
        case VG_QUANT_Q4_K: return deq_q4_k;
        case VG_QUANT_Q5_K: return deq_q5_k;
        case VG_QUANT_Q6_K: return deq_q6_k;
        case VG_QUANT_Q8_K: return deq_q8_k;
        case VG_QUANT_TQ1_0: return deq_tq1_0;
        case VG_QUANT_TQ2_0: return deq_tq2_0;
        case VG_QUANT_Q1_0: return deq_q1_0;
        case VG_QUANT_Q2_0: return deq_q2_0;
        default: return NULL;
    }
}

VG_Status vg_dequantize_block(uint32_t type, const void *block, float *out) {
    const VG_QuantInfo *qi = find(type);
    if (!qi || !block || !out || !qi->supported_scalar) return VG_E_UNSUPPORTED;
    if (type == VG_QUANT_F32) { memcpy(out, block, 4); return VG_OK; }
    if (type == VG_QUANT_F16) { out[0] = vg_half_to_float(*(const uint16_t *)block); return VG_OK; }
    if (type == VG_QUANT_BF16) { out[0] = bf16_to_float(*(const uint16_t *)block); return VG_OK; }
    if (type == VG_QUANT_F64) { out[0] = (float)*(const double *)block; return VG_OK; }
    if (type == VG_QUANT_I8) { out[0] = (float)*(const int8_t *)block; return VG_OK; }
    if (type == VG_QUANT_I16) { out[0] = (float)*(const int16_t *)block; return VG_OK; }
    if (type == VG_QUANT_I32) { out[0] = (float)*(const int32_t *)block; return VG_OK; }
    if (type == VG_QUANT_I64) { out[0] = (float)*(const int64_t *)block; return VG_OK; }
    deq_blocks_fn fn = block_fn(type);
    if (!fn) return VG_E_UNSUPPORTED;
    fn((const unsigned char *)block, out, 1);
    return VG_OK;
}

VG_Status vg_dequantize_row(uint32_t type, const void *packed, size_t bytes, float *out, size_t elements) {
    if (!packed || !out || !elements) return VG_E_INVALID;
    if (vg_quant_validate(type, elements, bytes) != VG_OK) return VG_E_FORMAT;
    const VG_QuantInfo *qi = find(type);
    if (!qi || !qi->supported_scalar) return VG_E_UNSUPPORTED;
    if (type == VG_QUANT_F32) { memcpy(out, packed, elements * sizeof(float)); return VG_OK; }
    if (type == VG_QUANT_F16 || type == VG_QUANT_BF16) {
        const uint16_t *p = (const uint16_t *)packed;
        for (size_t i = 0; i < elements; ++i) out[i] = type == VG_QUANT_F16 ? vg_half_to_float(p[i]) : bf16_to_float(p[i]);
        return VG_OK;
    }
    if (type == VG_QUANT_F64) { const double *p = (const double *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I8) { const int8_t *p = (const int8_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I16) { const int16_t *p = (const int16_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I32) { const int32_t *p = (const int32_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    if (type == VG_QUANT_I64) { const int64_t *p = (const int64_t *)packed; for (size_t i = 0; i < elements; ++i) out[i] = (float)p[i]; return VG_OK; }
    deq_blocks_fn fn = block_fn(type);
    if (!fn) return VG_E_UNSUPPORTED;
    fn((const unsigned char *)packed, out, elements / qi->block_size);
    return VG_OK;
}

VG_Status vg_quantized_dot(uint32_t type, const void *packed, size_t bytes, const float *x, size_t elements, float *out) {
    if (!out || !x) return VG_E_INVALID;
    float *tmp = (float *)malloc(elements * sizeof(float));
    if (!tmp) return VG_E_NOMEM;
    VG_Status st = vg_dequantize_row(type, packed, bytes, tmp, elements);
    if (st == VG_OK) {
        float acc = 0.0f;
        for (size_t i = 0; i < elements; ++i) acc += tmp[i] * x[i];
        *out = acc;
    }
    free(tmp);
    return st;
}
