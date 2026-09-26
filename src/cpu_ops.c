#include "vg/cpu_ops.h"
#include "vg/gguf.h"
#include "vg/quant.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* --- Quantized block structures (matching GGUF ggml format) --- */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
typedef struct { uint16_t d; uint8_t qs[32]; } block_q4_0;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K;

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    uint32_t f;
    if (exp == 0) { f = (sign << 31) | (mantissa ? ((mantissa << 13) + (127u - 14u) * (1u << 23)) : 0); }
    else if (exp == 31) { f = (sign << 31) | 0x7f800000u | (mantissa << 13); }
    else { f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mantissa << 13); }
    float v; memcpy(&v, &f, 4); return v;
}

void vg_cpu_matmul_q8_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const unsigned char *pw = (const unsigned char *)W;
    uint32_t bs = 32;
    for (uint32_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        const block_q8_0 *b = (const block_q8_0 *)(pw + (uint64_t)row * (in_dim / bs * sizeof(block_q8_0)));
        for (uint32_t gb = 0; gb < in_dim / bs; ++gb) {
            float scale = half_to_float(b[gb].d);
            for (uint32_t j = 0; j < 32; ++j) {
                acc += (float)b[gb].qs[j] * scale * x[gb * 32 + j];
            }
        }
        y[row] = acc;
    }
}

void vg_cpu_matmul_q4_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const unsigned char *pw = (const unsigned char *)W;
    uint32_t bs = 32;
    for (uint32_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        const block_q4_0 *b = (const block_q4_0 *)(pw + (uint64_t)row * (in_dim / bs * sizeof(block_q4_0)));
        for (uint32_t gb = 0; gb < in_dim / bs; ++gb) {
            float d = half_to_float(b[gb].d);
            for (uint32_t j = 0; j < 16; ++j) {
                acc += ((float)(b[gb].qs[j] & 0x0f) - 8.0f) * d * x[gb * 32 + j];
                acc += ((float)(b[gb].qs[j] >> 4) - 8.0f) * d * x[gb * 32 + 16 + j];
            }
        }
        y[row] = acc;
    }
}

/* Q4_K fused matmul: dequantize block-by-block per output row, never the whole weight */
static uint8_t get_scale_k4(int j, const uint8_t *q) {
    return j < 4 ? q[j] & 63 : (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
}
static uint8_t get_min_k4(int j, const uint8_t *q) {
    return j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}
void vg_cpu_matmul_q4_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const unsigned char *pw = (const unsigned char *)W;
    uint32_t bs = 256;
    uint32_t nbs = in_dim / bs;
    size_t bsize = sizeof(block_q4_K);
    for (uint32_t row = 0; row < out_dim; ++row) {
        const block_q4_K *b = (const block_q4_K *)(pw + (uint64_t)row * nbs * bsize);
        float acc = 0.0f;
        uint32_t off = 0;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            float d = half_to_float(b[ib].d);
            float mmin = half_to_float(b[ib].dmin);
            const uint8_t *sc = b[ib].scales;
            const uint8_t *qs = b[ib].qs;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                float d1 = d * get_scale_k4(is + 0, sc);
                float m1 = mmin * get_min_k4(is + 0, sc);
                float d2 = d * get_scale_k4(is + 1, sc);
                float m2 = mmin * get_min_k4(is + 1, sc);
                for (int l = 0; l < 32; ++l) {
                    acc += (d1 * (float)(qs[l] & 0xF) - m1) * x[off + j + l];
                }
                for (int l = 0; l < 32; ++l) {
                    acc += (d2 * (float)(qs[l] >> 4) - m2) * x[off + j + 32 + l];
                }
                qs += 32; is += 2;
            }
            off += bs;
        }
        y[row] = acc;
    }
}

/* Q6_K fused matmul */
void vg_cpu_matmul_q6_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const unsigned char *pw = (const unsigned char *)W;
    uint32_t bs = 256;
    uint32_t nbs = in_dim / bs;
    size_t bsize = sizeof(block_q6_K);
    for (uint32_t row = 0; row < out_dim; ++row) {
        const block_q6_K *b = (const block_q6_K *)(pw + (uint64_t)row * nbs * bsize);
        float acc = 0.0f;
        for (uint32_t ib = 0; ib < nbs; ++ib) {
            float d = half_to_float(b[ib].d);
            const int8_t *sc = b[ib].scales;
            const uint8_t *ql = b[ib].ql;
            const uint8_t *qh = b[ib].qh;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int32_t v1 = (int32_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int32_t v2 = (int32_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int32_t v3 = (int32_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int32_t v4 = (int32_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                float dx1 = d * sc[is + 0];
                float dx2 = d * sc[is + 2];
                float dx3 = d * sc[is + 4];
                float dx4 = d * sc[is + 6];
                acc += dx1 * v1 * x[ib*bs + l];
                acc += dx2 * v2 * x[ib*bs + 32 + l];
                acc += dx3 * v3 * x[ib*bs + 64 + l];
                acc += dx4 * v4 * x[ib*bs + 96 + l];
            }
        }
        y[row] = acc;
    }
}

void vg_cpu_matmul_f32(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const float *wf = (const float *)W;
    for (uint32_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        const float *wrow = wf + (uint64_t)row * in_dim;
        for (uint32_t j = 0; j < in_dim; ++j) acc += wrow[j] * x[j];
        y[row] = acc;
    }
}

void vg_cpu_matmul_f16(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    const uint16_t *wf = (const uint16_t *)W;
    for (uint32_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        const uint16_t *wrow = wf + (uint64_t)row * in_dim;
        for (uint32_t j = 0; j < in_dim; ++j) acc += half_to_float(wrow[j]) * x[j];
        y[row] = acc;
    }
}

VG_Status vg_cpu_matmul(uint32_t ggml_type, const void *W, const float *x, float *y,
                         uint32_t out_dim, uint32_t in_dim) {
    switch (ggml_type) {
        case 0: vg_cpu_matmul_f32(W, x, y, out_dim, in_dim); return VG_OK;
        case 1: vg_cpu_matmul_f16(W, x, y, out_dim, in_dim); return VG_OK;
        case 8: vg_cpu_matmul_q8_0(W, x, y, out_dim, in_dim); return VG_OK;
        case 2: case 42: vg_cpu_matmul_q4_0(W, x, y, out_dim, in_dim); return VG_OK;
        case 12: vg_cpu_matmul_q4_k(W, x, y, out_dim, in_dim); return VG_OK;
        case 14: vg_cpu_matmul_q6_k(W, x, y, out_dim, in_dim); return VG_OK;
        default: {
            /* Fallback: dequantize to f32 then matmul. */
            const VG_QuantInfo *qi = vg_quant_info(ggml_type);
            if (!qi) return VG_E_UNSUPPORTED;
            size_t n_el = (size_t)out_dim * in_dim;
            size_t block_count = n_el / qi->block_size;
            size_t wbytes = block_count * qi->bytes_per_block;
            float *deq = (float *)malloc(n_el * sizeof(float));
            if (!deq) return VG_E_NOMEM;
            VG_Status st = vg_dequantize_row(ggml_type, W, wbytes, deq, n_el);
            if (st == VG_OK) { vg_cpu_matmul_f32(deq, x, y, out_dim, in_dim); }
            free(deq);
            return st;
        }
    }
}

void vg_cpu_rmsnorm(float *out, const float *x, const float *weight, uint32_t dim, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / dim + eps);
    for (uint32_t i = 0; i < dim; ++i) out[i] = x[i] * ss * weight[i];
}

void vg_cpu_rope(float *q, float *k, uint32_t head_dim, uint32_t n_rot,
                 float freq_scale, float theta_base, int32_t pos) {
    uint32_t half = head_dim / 2;
    uint32_t iters = n_rot ? (n_rot < half ? n_rot : half) : half;
    for (uint32_t i = 0; i < iters; ++i) {
        float freq = pos * freq_scale / powf(theta_base, (float)(2 * i) / (float)head_dim);
        float cosf_val = cosf(freq), sinf_val = sinf(freq);
        float q0 = q[i], q1 = q[i + half];
        q[i] = q0 * cosf_val - q1 * sinf_val;
        q[i + half] = q0 * sinf_val + q1 * cosf_val;
        if (q != k) {
            float k0 = k[i], k1 = k[i + half];
            k[i] = k0 * cosf_val - k1 * sinf_val;
            k[i + half] = k0 * sinf_val + k1 * cosf_val;
        }
    }
}

void vg_cpu_softmax(float *x, uint32_t dim) {
    float mx = x[0]; for (uint32_t i = 1; i < dim; ++i) if (x[i] > mx) mx = x[i];
    float s = 0.0f; for (uint32_t i = 0; i < dim; ++i) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (uint32_t i = 0; i < dim; ++i) x[i] /= s;
}

void vg_cpu_swiglu(float *out, const float *gate, const float *up, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) {
        float g = gate[i] / (1.0f + expf(-gate[i]));  /* swish approximation */
        out[i] = g * up[i];
    }
}

void vg_cpu_gelu(float *x, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void vg_cpu_add(float *out, const float *a, const float *b, uint32_t dim) {
    for (uint32_t i = 0; i < dim; ++i) out[i] = a[i] + b[i];
}

void vg_cpu_scale(float *x, uint32_t dim, float scale) {
    for (uint32_t i = 0; i < dim; ++i) x[i] *= scale;
}
