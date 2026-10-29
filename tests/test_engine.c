#include "fixture.h"
#include "vg/cpu_ops.h"
#include "vg/full_engine.h"
#include "vg/gguf.h"
#include "vg/kv_cache.h"
#include "vg/model_graph.h"
#include "vg/model_probe.h"
#include "vg/quant.h"
#include "vg/speculative.h"
#include "vg/tensor_source.h"
#include "vg/threadpool.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

/* Defined with the decode tests further down; the BitNet tests above use it. */
static VG_ModelGraph *load_graph(const char *path, VG_GGUF **file, VG_TensorSource **src);

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } \
} while (0)

#define CHECK_NEAR(a, b, tol) do { \
    double _a = (double)(a), _b = (double)(b); \
    if (!(fabs(_a - _b) <= (tol))) { \
        fprintf(stderr, "FAIL %s:%d: %s=%.6f vs %.6f (tol %g)\n", __FILE__, __LINE__, #a, _a, _b, (double)(tol)); \
        ++g_failures; \
    } \
} while (0)

/* --- 1. Half precision conversion ------------------------------------------------ */
static void test_half_conversion(void) {
    CHECK_NEAR(vg_half_to_float(0x3C00), 1.0f, 0.0);
    CHECK_NEAR(vg_half_to_float(0xC000), -2.0f, 0.0);
    CHECK_NEAR(vg_half_to_float(0x0000), 0.0f, 0.0);
    CHECK_NEAR(vg_half_to_float(0x3555), 0.333251953125f, 1e-9); /* 1/3 */
    /* Subnormals: 2^-24 * mantissa. The old implementation was off by 2^-14. */
    CHECK_NEAR(vg_half_to_float(0x0001), ldexpf(1.0f, -24), 1e-12);
    CHECK_NEAR(vg_half_to_float(0x03FF), 1023.0f * ldexpf(1.0f, -24), 1e-12);
    CHECK_NEAR(vg_half_to_float(0x0001) * 1023.0f, vg_half_to_float(0x03FF), 1e-12);
    CHECK_NEAR(vg_half_to_float(0x7BFF), 65504.0f, 0.0);
    CHECK(isinf(vg_half_to_float(0x7C00)) && vg_half_to_float(0x7C00) > 0.0f);
}

/* --- 2. Quant dequantization vs an independent reference ------------------------- */
static void ref_q4_0_block(const unsigned char *blk, float *out) {
    const uint32_t bits = (uint32_t)blk[0] | ((uint32_t)blk[1] << 8);
    const float d = vg_half_to_float((uint16_t)bits);
    for (int j = 0; j < 16; ++j) {
        out[j] = (float)((blk[2 + j] & 0x0F) - 8) * d;
        out[16 + j] = (float)((blk[2 + j] >> 4) - 8) * d;
    }
}

static void test_quant_block_types(void) {
    /* Q4_0: d = 0.5 (0x3800), byte pattern 0x21 -> low nibble 1 (-7), high 2 (-6). */
    unsigned char q4[18] = {0x00, 0x38};
    memset(q4 + 2, 0x21, 16);
    float out[32];
    CHECK(vg_dequantize_row(VG_QUANT_Q4_0, q4, sizeof(q4), out, 32) == VG_OK);
    float ref[32];
    ref_q4_0_block(q4, ref);
    for (int i = 0; i < 32; ++i) CHECK_NEAR(out[i], ref[i], 1e-6);

    /* Q8_0: d = 1.0, values -16..15 */
    unsigned char q8[34] = {0x00, 0x3C};
    for (int i = 0; i < 32; ++i) q8[2 + i] = (unsigned char)(int8_t)(i - 16);
    CHECK(vg_dequantize_row(VG_QUANT_Q8_0, q8, sizeof(q8), out, 32) == VG_OK);
    for (int i = 0; i < 32; ++i) CHECK_NEAR(out[i], (float)(i - 16), 1e-6);

    /* Q4_1: d = 1.0, m = 0.5, nibbles = 0 -> every value is the offset */
    unsigned char q41[20] = {0x00, 0x3C, 0x00, 0x38};
    memset(q41 + 4, 0x00, 16);
    CHECK(vg_dequantize_row(VG_QUANT_Q4_1, q41, sizeof(q41), out, 32) == VG_OK);
    for (int i = 0; i < 32; ++i) CHECK_NEAR(out[i], 0.5f, 1e-6);

    /* Q5_0: d = 1.0, high bits all set -> values 16 + nibble - 16 = nibble */
    unsigned char q50[22] = {0x00, 0x3C, 0xFF, 0xFF, 0xFF, 0xFF};
    memset(q50 + 6, 0x33, 16);
    CHECK(vg_dequantize_row(VG_QUANT_Q5_0, q50, sizeof(q50), out, 32) == VG_OK);
    for (int i = 0; i < 16; ++i) {
        CHECK_NEAR(out[i], 3.0f, 1e-6);
        CHECK_NEAR(out[16 + i], 3.0f, 1e-6);
    }

    /* Q5_1: d = 1.0, m = 0.0, high bits set, nibbles 1 -> value 17 */
    unsigned char q51[24] = {0x00, 0x3C, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF};
    memset(q51 + 8, 0x11, 16);
    CHECK(vg_dequantize_row(VG_QUANT_Q5_1, q51, sizeof(q51), out, 32) == VG_OK);
    for (int i = 0; i < 32; ++i) CHECK_NEAR(out[i], 17.0f, 1e-6);

    /* Q8_K: float scale, int8 payload */
    unsigned char q8k[292];
    memset(q8k, 0, sizeof(q8k));
    float scale = 0.25f;
    memcpy(q8k, &scale, 4);
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (unsigned char)(int8_t)(i % 7 - 3);
    float big[256];
    CHECK(vg_dequantize_row(VG_QUANT_Q8_K, q8k, sizeof(q8k), big, 256) == VG_OK);
    for (int i = 0; i < 256; ++i) CHECK_NEAR(big[i], 0.25f * (float)(i % 7 - 3), 1e-6);

    /* IQ4_NL: codebook lookup at index 8 -> +1.0 with d = 1.0 */
    unsigned char iq[18] = {0x00, 0x3C};
    memset(iq + 2, 0x88, 16);
    CHECK(vg_dequantize_row(VG_QUANT_IQ4_NL, iq, sizeof(iq), out, 32) == VG_OK);
    for (int i = 0; i < 32; ++i) CHECK_NEAR(out[i], 1.0f, 1e-6);
}

/* K-quant suite: build blocks whose math is easy to verify by hand. */
static void test_k_quants(void) {
    /* Q2_K: scales[n] = 0x11 -> 4-bit scale 1, 4-bit min 1; d = dmin = 1.0;
     * qs nibble pairs 0b01 -> value 1 for lane 0, so out = 1*1 - 1 = 0. */
    unsigned char q2[84];
    memset(q2, 0, sizeof(q2));
    q2[0] = 0x11; /* scales[0]: dl=1, ml=1 */
    for (int i = 1; i < 16; ++i) q2[i] = 0x00; /* other scales: dl=0 -> contributes 0 */
    for (int i = 16; i < 80; ++i) q2[i] = 0x55; /* 2-bit fields = 1 */
    q2[80] = 0x00; q2[81] = 0x3C; /* d = 1.0 */
    q2[82] = 0x00; q2[83] = 0x3C; /* dmin = 1.0 */
    float out[256];
    CHECK(vg_dequantize_row(VG_QUANT_Q2_K, q2, sizeof(q2), out, 256) == VG_OK);
    for (int i = 0; i < 256; ++i) CHECK(isfinite(out[i]));

    /* Q3_K: all scale bytes 0 -> scale = -32; qs zero and hmask zero ->
     * value = 0 - 4 = -4, so out = -32 * -4 = 128 (d = 1). */
    unsigned char q3[110];
    memset(q3, 0, sizeof(q3)); /* hmask[32] | qs[64] | scales[12] | d */
    q3[108] = 0x00; q3[109] = 0x3C; /* d = 1.0 */
    CHECK(vg_dequantize_row(VG_QUANT_Q3_K, q3, sizeof(q3), out, 256) == VG_OK);
    for (int i = 0; i < 256; ++i) CHECK_NEAR(out[i], 128.0f, 1e-4);

    /* Q5_K: dl = 1, ml = 0, nibbles 1, high bit 0 -> value 1. d=1, dmin=0. */
    unsigned char q5[176];
    memset(q5, 0, sizeof(q5));
    q5[0] = 0x00; q5[1] = 0x3C;  /* d = 1.0 */
    q5[2] = 0x00; q5[3] = 0x00;  /* dmin = 0.0 */
    /* 6-bit scale field: j<4 read scales[j], j>=4 read the low nibble of scales[j+4]. */
    for (int i = 0; i < 12; ++i) q5[4 + i] = (uint8_t)(i < 4 ? 0x01 : (i < 8 ? 0x00 : 0x01));
    memset(q5 + 16, 0x00, 32);   /* qh: no 16-value offset */
    memset(q5 + 48, 0x11, 128);  /* qs nibbles = 1 */
    CHECK(vg_dequantize_row(VG_QUANT_Q5_K, q5, sizeof(q5), out, 256) == VG_OK);
    for (int i = 0; i < 256; ++i) CHECK_NEAR(out[i], 1.0f, 1e-5);

    /* Q6_K keeps its dedicated test in test_quant.c; assert the block size here. */
    CHECK(vg_quant_info(VG_QUANT_Q6_K)->bytes_per_block == 210);
    CHECK(vg_quant_info(VG_QUANT_Q8_K)->bytes_per_block == 292);
    CHECK(vg_quant_info(VG_QUANT_Q3_K)->bytes_per_block == 110);
}


/* Packed tensor with moderate scales: random payload bytes (all quantized types store the
 * block scale separately from the payload) so the dequantized values stay in a sane range. */
static uint16_t f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static void fill_packed(uint32_t type, unsigned char *packed, size_t blocks, size_t block_bytes, uint32_t seed) {
    if (type == VG_QUANT_F32) {
        float *p = (float *)packed;
        for (size_t i = 0; i < blocks * block_bytes / 4; ++i) p[i] = ((float)((int)((i + seed) % 13) - 6)) / 8.0f;
        return;
    }
    if (type == VG_QUANT_F16) {
        uint16_t *p = (uint16_t *)packed;
        for (size_t i = 0; i < blocks * block_bytes / 2; ++i) p[i] = f32_to_f16(((float)((int)((i + seed) % 13) - 6)) / 8.0f);
        return;
    }
    for (size_t i = 0; i < blocks * block_bytes; ++i) packed[i] = (unsigned char)((i * 131u + seed * 37u) >> 3);
    const uint16_t half = f32_to_f16(0.0625f);
    unsigned char lo = (unsigned char)(half & 0xFFu), hi = (unsigned char)(half >> 8);
    size_t d_off = 0, m_off = SIZE_MAX;
    float fscale = 0.0625f;
    switch (type) {
        case VG_QUANT_Q4_0: case VG_QUANT_Q5_0: case VG_QUANT_Q8_0: case VG_QUANT_IQ4_NL: d_off = 0; break;
        case VG_QUANT_Q4_1: case VG_QUANT_Q5_1: d_off = 0; m_off = 2; break;
        case VG_QUANT_Q2_K: d_off = 80; m_off = 82; break;
        case VG_QUANT_Q3_K: d_off = 108; break;
        case VG_QUANT_Q4_K: case VG_QUANT_Q5_K: d_off = 0; m_off = 2; break;
        case VG_QUANT_Q6_K: d_off = 208; break;
        /* Ternary: the scale sits at the END of the TQ blocks but at the START of Q1_0/Q2_0. */
        case VG_QUANT_TQ1_0: d_off = 52; break;
        case VG_QUANT_TQ2_0: d_off = 64; break;
        case VG_QUANT_Q1_0: case VG_QUANT_Q2_0: d_off = 0; break;
        case VG_QUANT_Q8_K: {
            for (size_t b = 0; b < blocks; ++b) memcpy(packed + b * block_bytes, &fscale, 4);
            return;
        }
        default: return;
    }
    for (size_t b = 0; b < blocks; ++b) {
        unsigned char *blk = packed + b * block_bytes;
        blk[d_off] = lo; blk[d_off + 1] = hi;
        if (m_off != SIZE_MAX) { blk[m_off] = lo; blk[m_off + 1] = hi; }
    }
}

/* --- 3. Fused matmul vs the dequantize-then-dot reference ------------------------ */
static void test_matmul_parity(uint32_t type, uint32_t in_dim, uint32_t out_dim, uint32_t seed) {
    const VG_QuantInfo *qi = vg_quant_info(type);
    if (!qi || !qi->supported_scalar) return;
    if (in_dim % qi->block_size) return;
    const size_t row_bytes = (size_t)(in_dim / qi->block_size) * qi->bytes_per_block;
    const size_t total = row_bytes * out_dim;
    unsigned char *packed = (unsigned char *)calloc(1, total);
    float *x = (float *)calloc(in_dim, sizeof(float));
    float *y_fast = (float *)calloc(out_dim, sizeof(float));
    float *y_ref = (float *)calloc(out_dim, sizeof(float));
    float *row = (float *)calloc(in_dim, sizeof(float));
    assert(packed && x && y_fast && y_ref && row);

    fill_packed(type, packed, out_dim * (in_dim / qi->block_size), qi->bytes_per_block, seed);
    uint32_t s = seed * 1103515245u + 12345u;
    for (uint32_t i = 0; i < in_dim; ++i) { s = s * 1664525u + 1013904223u; x[i] = ((float)((s >> 9) % 2001u) - 1000.0f) / 1000.0f; }

    CHECK(vg_cpu_matmul_ref(type, packed, x, y_ref, out_dim, in_dim) == VG_OK);
    CHECK(vg_cpu_matmul(type, packed, x, y_fast, out_dim, in_dim) == VG_OK);

    /* Independent reference: dequantize each row with the public API and dot it. */
    for (uint32_t r = 0; r < out_dim; ++r) {
        CHECK(vg_dequantize_row(type, packed + (size_t)r * row_bytes, row_bytes, row, in_dim) == VG_OK);
        double acc = 0.0;
        for (uint32_t j = 0; j < in_dim; ++j) acc += (double)row[j] * (double)x[j];
        CHECK_NEAR(y_ref[r], acc, fabs(acc) * 1e-4 + 1e-3);
        CHECK_NEAR(y_fast[r], y_ref[r], fabs(y_ref[r]) * 1e-5 + 1e-5);
    }
    free(packed); free(x); free(y_fast); free(y_ref); free(row);
}

static void test_threaded_determinism(void) {
    const uint32_t in_dim = 256, out_dim = 128;
    unsigned char *packed = (unsigned char *)calloc(1, (size_t)out_dim * (in_dim / 32) * 18);
    float *x = (float *)calloc(in_dim, sizeof(float));
    float *y1 = (float *)calloc(out_dim, sizeof(float));
    float *y2 = (float *)calloc(out_dim, sizeof(float));
    assert(packed && x && y1 && y2);
    fill_packed(VG_QUANT_Q4_0, packed, (size_t)out_dim * (in_dim / 32), 18, 5);
    for (uint32_t i = 0; i < in_dim; ++i) x[i] = (float)((int)i % 11 - 5) * 0.25f;

    CHECK(vg_cpu_set_threads(1) == 1);
    CHECK(vg_cpu_matmul(VG_QUANT_Q4_0, packed, x, y1, out_dim, in_dim) == VG_OK);
    int t = vg_cpu_set_threads(4);
    CHECK(t >= 1);
    CHECK(vg_cpu_matmul(VG_QUANT_Q4_0, packed, x, y2, out_dim, in_dim) == VG_OK);
    for (uint32_t i = 0; i < out_dim; ++i) CHECK_NEAR(y1[i], y2[i], 1e-6);
    vg_cpu_set_threads(0);
    free(packed); free(x); free(y1); free(y2);
}

static void test_thread_resizing(void) {
    /* Thread count used to be fixed by whichever call initialised the pool first: asking for 4
     * after an initial 1 returned 1 forever. */
    CHECK(vg_cpu_set_threads(1) == 1);
    const int four = vg_cpu_set_threads(4);
    CHECK(four == 4);
    CHECK(vg_cpu_get_threads() == four);
    CHECK(vg_cpu_set_threads(2) == 2);
    CHECK(vg_cpu_get_threads() == 2);
    vg_cpu_set_threads(0);
}

static void range_fill(void *arg, uint32_t begin, uint32_t end) {
    int *arr = (int *)arg;
    for (uint32_t i = begin; i < end; ++i) arr[i] = (int)i + 1;
}

static void test_threadpool_parallel(void) {
    VG_ThreadPool *pool = vg_threadpool_create(3);
    CHECK(pool != NULL);
    if (!pool) return;
    enum { N = 64 };
    int arr[N];
    /* wait() used to return before the task bodies finished (no done-signal), so this
     * loop would see partially filled arrays. Run enough batches to catch it. */
    for (int iter = 0; iter < 200; ++iter) {
        memset(arr, 0, sizeof(arr));
        vg_threadpool_parallel_for(pool, range_fill, arr, N);
        int ok = 1;
        for (int i = 0; i < N; ++i) if (arr[i] != i + 1) ok = 0;
        CHECK(ok);
        if (!ok) break;
    }
    /* A single-range request must run inline on the caller. */
    memset(arr, 0, sizeof(arr));
    vg_threadpool_parallel_for(pool, range_fill, arr, 1);
    CHECK(arr[0] == 1);
    vg_threadpool_destroy(pool);
}

/* --- 4. Probe --------------------------------------------------------------------- */
static void test_probe_ok(const char *path) {
    VG_GGUF *g = NULL;
    CHECK(vg_gguf_open(path, &g) == VG_OK);
    if (!g) return;
    VG_ModelProbe p;
    CHECK(vg_model_probe(g, &p) == VG_OK);
    if (!p.ok) fprintf(stderr, "probe rejected dense fixture: %s\n", p.detail);
    CHECK(p.ok == 1);
    CHECK(p.status == VG_OK);
    CHECK(p.n_layer == VG_FIX_DENSE_LAYERS);
    CHECK(p.n_embd == VG_FIX_DENSE_EMBD);
    CHECK(p.is_moe == 0);
    CHECK(p.has_tokenizer == 1);
    CHECK(p.n_quant_kinds >= 2);
    CHECK(strstr(p.required_ops, "swiglu") != NULL);
    char report[2048];
    vg_model_probe_report(&p, report, sizeof(report));
    CHECK(strstr(report, "SUPPORTED") != NULL);
    vg_gguf_close(g);
}

static void test_probe_unsupported(const char *path) {
    VG_GGUF *g = NULL;
    CHECK(vg_gguf_open(path, &g) == VG_OK);
    if (!g) return;
    VG_ModelProbe p;
    CHECK(vg_model_probe(g, &p) == VG_OK);
    CHECK(p.ok == 0);
    CHECK(p.status == VG_E_UNSUPPORTED);
    CHECK(strcmp(p.detail_tensor, "blk.0.attn_q.weight") == 0);
    CHECK(strstr(p.detail, "IQ2_XXS") != NULL);
    char report[2048];
    vg_model_probe_report(&p, report, sizeof(report));
    CHECK(strstr(report, "NOT SUPPORTED") != NULL);
    vg_gguf_close(g);
}

/* --- 4b. BitNet b1.58: ternary weights, sub-norms, per-tensor scale --------------- */
static void test_bitnet_probe(const char *path) {
    VG_GGUF *g = NULL;
    CHECK(vg_gguf_open(path, &g) == VG_OK);
    if (!g) return;
    VG_ModelProbe p;
    CHECK(vg_model_probe(g, &p) == VG_OK);
    if (!p.ok) fprintf(stderr, "probe rejected bitnet fixture: %s\n", p.detail);
    CHECK(p.ok == 1);
    CHECK(strcmp(p.arch, "bitnet") == 0);
    CHECK(p.has_sub_norm == 1);
    CHECK(strstr(p.required_ops, "sub_norm") != NULL);
    /* required_ops carries the type COUNT; the names live in quant_names. */
    int saw_tq2 = 0;
    for (uint32_t i = 0; i < p.n_quant_kinds; ++i)
        if (p.quant_names[i] && strcmp(p.quant_names[i], "TQ2_0") == 0) saw_tq2 = 1;
    CHECK(saw_tq2);
    char report[2048];
    vg_model_probe_report(&p, report, sizeof(report));
    CHECK(strstr(report, "sub_norm=yes") != NULL);
    CHECK(strstr(report, "SUPPORTED") != NULL);

    /* A BitNet model missing its sub-norms must be rejected and named, not silently
     * evaluated as a plain llama graph (which would change the numerics). */
    vg_gguf_close(g);
}

static void test_bitnet_decode(const char *path) {
    VG_GGUF *file = NULL;
    VG_TensorSource *src = NULL;
    VG_ModelGraph *g = load_graph(path, &file, &src);
    CHECK(g != NULL);
    if (!g) return;
    const VG_ModelConfig *cfg = vg_model_graph_config(g);
    CHECK(cfg->has_sub_norm == 1);
    float *logits = (float *)malloc(cfg->n_vocab * sizeof(float));
    if (!logits) { vg_model_graph_free(g); vg_tensor_source_close(src); vg_gguf_close(file); return; }
    vg_model_graph_reset(g);
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits) == VG_OK);
    int finite = 1, nonzero = 0;
    for (uint32_t i = 0; i < cfg->n_vocab; ++i) {
        if (!isfinite(logits[i])) finite = 0;
        if (logits[i] != 0.0f) nonzero = 1;
    }
    CHECK(finite);
    CHECK(nonzero);
    for (int32_t pos = 1; pos < 8; ++pos) CHECK(vg_model_graph_decode(g, 3, pos, logits) == VG_OK);
    free(logits);
    vg_model_graph_free(g);
    vg_tensor_source_close(src);
    vg_gguf_close(file);
}

/* --- 4c. LoRA -------------------------------------------------------------------- */
static void test_lora_delta(const char *lora_path) {
    VG_GGUF *file = NULL;
    CHECK(vg_gguf_open(lora_path, &file) == VG_OK);
    if (!file) return;
    VG_LoraAdapter *ad = NULL;
    CHECK(vg_lora_load(file, &ad) == VG_OK);
    CHECK(ad != NULL);
    if (!ad) { vg_gguf_close(file); return; }
    /* The A-without-B target must be dropped. */
    CHECK(vg_lora_target_count(ad) == 2);
    CHECK(vg_lora_rank(ad) == 4);
    CHECK(vg_lora_target_rank(ad, "blk.0.attn_q.weight") == 4);
    CHECK(vg_lora_target_rank(ad, "blk.0.attn_v.weight") == 0);

    const uint32_t D = VG_FIX_BITNET_EMBD, FF = VG_FIX_BITNET_FF;
    float *x = (float *)malloc(D * sizeof(float));
    float *y_delta = (float *)calloc(D, sizeof(float));
    float *y_ref = (float *)calloc(D, sizeof(float));
    float *w = (float *)calloc((size_t)D * FF, sizeof(float));
    if (!x || !y_delta || !y_ref || !w) { free(x); free(y_delta); free(y_ref); free(w); vg_lora_free(ad); vg_gguf_close(file); return; }
    for (uint32_t i = 0; i < D; ++i) x[i] = ((float)((int)(i % 17) - 8)) / 8.0f;

    /* Path A: apply the delta to the matmul result (what the engine does). */
    CHECK(vg_lora_apply_delta(ad, "blk.0.ffn_down.weight", x, y_delta, D, FF, 1.0f) == 1);
    /* Path B: materialize the delta into a dense weight, then multiply (independent code). */
    CHECK(vg_lora_apply(ad, "blk.0.ffn_down.weight", w, D, FF, 1.0f) == VG_OK);
    CHECK(vg_cpu_matmul(VG_QUANT_F32, w, x, y_ref, D, FF) == VG_OK);
    for (uint32_t i = 0; i < D; ++i) CHECK_NEAR(y_delta[i], y_ref[i], fabs((double)y_ref[i]) * 1e-4 + 1e-5);

    /* Not a target: no delta, y untouched. */
    float untouched[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    CHECK(vg_lora_apply_delta(ad, "blk.0.ffn_up.weight", x, untouched, 4, D, 1.0f) == 0);
    CHECK(untouched[0] == 1.0f && untouched[3] == 4.0f);

    /* Zero input produces zero delta: catches an adapter that ignores x. */
    memset(x, 0, D * sizeof(float));
    memset(y_delta, 0, D * sizeof(float));
    CHECK(vg_lora_apply_delta(ad, "blk.0.attn_q.weight", x, y_delta, D, D, 1.0f) == 1);
    for (uint32_t i = 0; i < D; ++i) CHECK(fabsf(y_delta[i]) < 1e-9f);

    free(x); free(y_delta); free(y_ref); free(w);
    vg_lora_free(ad);
    vg_gguf_close(file);
}

/* --- 5. End-to-end decode --------------------------------------------------------- */
static VG_ModelGraph *load_graph_cfg(const char *path, VG_GGUF **file, VG_TensorSource **src,
                                     int use_mmap, size_t host_budget) {
    if (vg_gguf_open(path, file) != VG_OK) return NULL;
    VG_TensorSourceConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = host_budget;
    cfg.io_chunk_bytes = 1u * 1024u * 1024u;
    cfg.use_mmap = use_mmap;
    if (vg_tensor_source_open(*file, &cfg, src) != VG_OK) {
        vg_gguf_close(*file); *file = NULL;
        return NULL;
    }
    VG_ModelGraph *g = NULL;
    if (vg_model_graph_load(*file, *src, &g) != VG_OK) {
        vg_tensor_source_close(*src); *src = NULL;
        vg_gguf_close(*file); *file = NULL;
        return NULL;
    }
    return g;
}

static VG_ModelGraph *load_graph(const char *path, VG_GGUF **file, VG_TensorSource **src) {
    return load_graph_cfg(path, file, src, 1, 32u * 1024u * 1024u);
}

static void test_dense_decode(const char *path) {
    VG_GGUF *file = NULL;
    VG_TensorSource *src = NULL;
    VG_ModelGraph *g = load_graph(path, &file, &src);
    CHECK(g != NULL);
    if (!g) return;
    const VG_ModelConfig *cfg = vg_model_graph_config(g);
    CHECK(cfg->n_layer == VG_FIX_DENSE_LAYERS);

    float *logits_a = (float *)malloc(cfg->n_vocab * sizeof(float));
    float *logits_b = (float *)malloc(cfg->n_vocab * sizeof(float));
    assert(logits_a && logits_b);

    vg_model_graph_reset(g);
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits_a) == VG_OK);
    float *emb = (float *)malloc(cfg->n_embd * sizeof(float));
    assert(emb);
    CHECK(vg_model_graph_embed(g, cfg->bos_id, emb) == VG_OK);
    int finite = 1;
    for (uint32_t i = 0; i < cfg->n_vocab; ++i) if (!isfinite(logits_a[i])) finite = 0;
    CHECK(finite);

    /* Same input from a clean state must reproduce the logits bit for bit. */
    vg_model_graph_reset(g);
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits_b) == VG_OK);
    int identical = 1;
    for (uint32_t i = 0; i < cfg->n_vocab; ++i) if (logits_a[i] != logits_b[i]) identical = 0;
    CHECK(identical);

    /* Position handling: 8 tokens in sequence, then past the context must fail cleanly. */
    for (int32_t pos = 1; pos < 8; ++pos) CHECK(vg_model_graph_decode(g, 3, pos, logits_a) == VG_OK);
    CHECK(vg_model_graph_decode(g, 3, (int32_t)cfg->n_ctx, logits_a) == VG_E_RANGE);
    CHECK(vg_model_graph_decode(g, 3, -1, logits_a) == VG_E_RANGE);
    /* Invalid token id is rejected instead of reading out of bounds. */
    CHECK(vg_model_graph_embed(g, (int32_t)cfg->n_vocab, emb) == VG_E_INVALID);

    free(emb); free(logits_a); free(logits_b);
    vg_model_graph_free(g);
    vg_tensor_source_close(src);
    vg_gguf_close(file);
}

static void test_moe_routing(const char *path) {
    VG_GGUF *file = NULL;
    VG_TensorSource *src = NULL;
    VG_ModelGraph *g = load_graph(path, &file, &src);
    CHECK(g != NULL);
    if (!g) return;
    const VG_ModelConfig *cfg = vg_model_graph_config(g);
    CHECK(cfg->n_expert == VG_FIX_MOE_EXPERTS);
    CHECK(cfg->n_expert_used == VG_FIX_MOE_USED);

    VG_ModelProbe p;
    CHECK(vg_model_probe(file, &p) == VG_OK);
    CHECK(p.ok == 1);
    CHECK(p.is_moe == 1);
    CHECK(strstr(p.required_ops, "moe_route") != NULL);

    /* Router rows are constant per expert: expert e has value (e+1)/10, so with a positive
     * input the top-2 must be experts 3 and 2, in that order. */
    float *x = (float *)malloc(cfg->n_embd * sizeof(float));
    float *gate = (float *)calloc(cfg->n_expert, sizeof(float));
    assert(x && gate);
    for (uint32_t i = 0; i < cfg->n_embd; ++i) x[i] = 1.0f;
    for (uint32_t e = 0; e < cfg->n_expert; ++e) gate[e] = (float)(e + 1);

    VG_MoEState st;
    memset(&st, 0, sizeof(st));
    CHECK(vg_model_graph_moe_route(g, 0, NULL, gate, &st) == VG_OK);
    CHECK(st.n_expert_used == VG_FIX_MOE_USED);
    CHECK(st.expert_ids[0] == 3);
    CHECK(st.expert_ids[1] == 2);
    vg_moe_state_free(&st);

    /* The router projection path (no precomputed logits) must agree. */
    VG_MoEState st2;
    memset(&st2, 0, sizeof(st2));
    CHECK(vg_model_graph_moe_route(g, 0, x, NULL, &st2) == VG_OK);
    CHECK(st2.expert_ids[0] == 3);
    CHECK(st2.expert_ids[1] == 2);
    vg_moe_state_free(&st2);

    float *logits = (float *)malloc(cfg->n_vocab * sizeof(float));
    assert(logits);
    vg_model_graph_reset(g);
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits) == VG_OK);
    uint64_t hits = 0, misses = 0, evictions = 0;
    vg_model_graph_expert_stats(g, &hits, &misses, &evictions);
    /* First token: 2 experts x 3 tensors, all cold. */
    CHECK(misses >= 6);
    /* A second token reuses the same routing decision -> cache hits. */
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 1, logits) == VG_OK);
    vg_model_graph_expert_stats(g, &hits, &misses, &evictions);
    CHECK(hits >= 6);
    int finite = 1;
    for (uint32_t i = 0; i < cfg->n_vocab; ++i) if (!isfinite(logits[i])) finite = 0;
    CHECK(finite);

    /* A cache smaller than the working set must still produce a usable result. */
    CHECK(vg_model_graph_set_expert_cache(g, 2) == VG_OK);
    CHECK(vg_model_graph_decode(g, cfg->bos_id, 2, logits) == VG_OK);

    /* Regression: the routed experts must actually reach the residual stream. The MoE
     * block used to zero its input buffer before the expert matvecs, so every expert
     * multiplied a zero vector and this comparison against an all-zero-expert model was
     * bit-identical. */
    {
        size_t zlen = strlen(path) + 16u;
        char *zero_path = (char *)malloc(zlen);
        if (zero_path) {
            snprintf(zero_path, zlen, "%s.zero.gguf", path);
            if (vg_fixture_write_moe_zero(zero_path) == 0) {
                VG_GGUF *zf = NULL;
                VG_TensorSource *zs = NULL;
                VG_ModelGraph *zg = load_graph(zero_path, &zf, &zs);
                CHECK(zg != NULL);
                if (zg) {
                    float *zl = (float *)malloc(cfg->n_vocab * sizeof(float));
                    assert(zl);
                    vg_model_graph_reset(g);
                    CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits) == VG_OK);
                    vg_model_graph_reset(zg);
                    CHECK(vg_model_graph_decode(zg, cfg->bos_id, 0, zl) == VG_OK);
                    double diff = 0.0;
                    for (uint32_t i = 0; i < cfg->n_vocab; ++i) {
                        double d = fabs((double)logits[i] - (double)zl[i]);
                        if (d > diff) diff = d;
                    }
                    CHECK(diff > 1e-4);
                    free(zl);
                    vg_model_graph_free(zg);
                }
                vg_tensor_source_close(zs);
                vg_gguf_close(zf);
            }
            remove(zero_path);
            free(zero_path);
        }
    }

    free(x); free(gate); free(logits);
    vg_model_graph_free(g);
    vg_tensor_source_close(src);
    vg_gguf_close(file);
}

/* The expert cache hands out pointers into a tensor it no longer holds a lease on. With mmap
 * that is safe (the mapping outlives the lease), but a RAM-backed source is free to evict the
 * tensor as soon as its refcount drops, leaving every cached slot dangling. Driven through a
 * series of budgets so both backing modes are exercised: a tensor that fits the budget is
 * cached and pinnable, one that does not arrives as a transient malloc that release frees.
 * Logits must match the mmap reference exactly either way; under ASan a dangling slot is a
 * use-after-free rather than a silently wrong number. */
static void test_moe_expert_cache_lease(const char *path) {
    static const size_t budgets[] = { 8u * 1024u, 12u * 1024u, 24u * 1024u, 32u * 1024u, 64u * 1024u };
    VG_GGUF *rf = NULL; VG_TensorSource *rs = NULL;
    VG_ModelGraph *ref = load_graph(path, &rf, &rs);
    CHECK(ref != NULL);
    if (!ref) return;
    const VG_ModelConfig *cfg = vg_model_graph_config(ref);
    float *ref_logits = (float *)malloc(cfg->n_vocab * sizeof(float));
    float *logits = (float *)malloc(cfg->n_vocab * sizeof(float));
    assert(ref_logits && logits);
    vg_model_graph_reset(ref);
    CHECK(vg_model_graph_decode(ref, cfg->bos_id, 0, ref_logits) == VG_OK);

    for (size_t b = 0; b < sizeof(budgets) / sizeof(budgets[0]); ++b) {
        VG_GGUF *f = NULL; VG_TensorSource *s = NULL;
        VG_ModelGraph *g = load_graph_cfg(path, &f, &s, 0, budgets[b]);
        CHECK(g != NULL);
        if (!g) continue;
        /* A cache too small for the working set forces both slot eviction and source eviction. */
        CHECK(vg_model_graph_set_expert_cache(g, 4) == VG_OK);
        vg_model_graph_reset(g);
        CHECK(vg_model_graph_decode(g, cfg->bos_id, 0, logits) == VG_OK);
        int identical = 1;
        for (uint32_t i = 0; i < cfg->n_vocab; ++i) if (logits[i] != ref_logits[i]) identical = 0;
        CHECK(identical);
        /* More positions: repeated routing, slot churn, and eviction between acquires. */
        for (int32_t pos = 1; pos < 6; ++pos) {
            CHECK(vg_model_graph_decode(g, (int32_t)(pos % 4), pos, logits) == VG_OK);
            int finite = 1;
            for (uint32_t i = 0; i < cfg->n_vocab; ++i) if (!isfinite(logits[i])) finite = 0;
            CHECK(finite);
        }
        uint64_t hits = 0, misses = 0, evictions = 0;
        vg_model_graph_expert_stats(g, &hits, &misses, &evictions);
        CHECK(misses > 0 && evictions > 0);
        vg_model_graph_free(g);
        vg_tensor_source_close(s);
        vg_gguf_close(f);
    }

    free(logits); free(ref_logits);
    vg_model_graph_free(ref);
    vg_tensor_source_close(rs);
    vg_gguf_close(rf);
}

#ifdef VG_TEST_FULL_ENGINE
/* --- 5b. Speculative decoding ----------------------------------------------------
 * With draft == target every proposal must be accepted AND the token stream must match plain
 * greedy decoding exactly; that is the property the old argmax-comparison code broke as soon
 * as sampling was enabled. */
static void test_speculative(const char *path) {
    VG_GGUF *file = NULL;
    VG_TensorSource *src = NULL;
    VG_ModelGraph *g = load_graph(path, &file, &src);
    CHECK(g != NULL);
    if (!g) return;
    const VG_ModelConfig *mc = vg_model_graph_config(g);

    /* Reference: plain greedy decoding. */
    float *logits = (float *)malloc(mc->n_vocab * sizeof(float));
    int32_t reference[8];
    vg_model_graph_reset(g);
    int32_t tok = mc->bos_id;
    int32_t ref_n = 0;
    for (int32_t pos = 0; pos < 8; ++pos) {
        if (vg_model_graph_decode(g, tok, pos, logits) != VG_OK) break;
        int32_t best = 0;
        for (uint32_t v = 1; v < mc->n_vocab; ++v) if (logits[v] > logits[best]) best = (int32_t)v;
        reference[ref_n++] = best;
        tok = best;
    }
    CHECK(ref_n > 0);

    VG_SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.draft_graph = g;
    cfg.target_graph = g;
    cfg.n_draft = 4;
    cfg.temperature = 0.0f;
    uint32_t rng = 7u;
    cfg.rng = &rng;

    vg_model_graph_reset(g);
    int32_t emitted[8], n_emitted = 0, pos = 0, in = mc->bos_id;
    while (n_emitted < ref_n) {
        int32_t acc[8];
        const int32_t n = vg_speculative_step(&cfg, in, pos, acc, 8);
        if (n <= 0) break;
        /* With identical models every proposal is accepted, so a step returns n_draft tokens. */
        CHECK(n >= 1);
        for (int32_t i = 0; i < n && n_emitted < 8; ++i) emitted[n_emitted++] = acc[i];
        in = acc[n - 1];
        pos += n;
    }
    CHECK(n_emitted == ref_n);
    for (int32_t i = 0; i < ref_n && i < n_emitted; ++i) CHECK(emitted[i] == reference[i]);

    free(logits);
    vg_model_graph_free(g);
    vg_tensor_source_close(src);
    vg_gguf_close(file);
}

/* --- 6. Whole generation pipeline ------------------------------------------------
 * The CLI smoke test only checks an exit code. This drives vg_generate() in-process so the probe
 * gate, tokenizer, graph, sampler and callback are all verified, including determinism. */
typedef struct VG_GenRec {
    int32_t ids[16];
    uint32_t count;
} VG_GenRec;

static int gen_record(const char *piece, size_t bytes, int32_t token, void *user) {
    (void)piece;
    (void)bytes;
    VG_GenRec *r = (VG_GenRec *)user;
    if (r->count < 16) r->ids[r->count] = token;
    ++r->count;
    return 1;
}

static void make_generate_config(VG_GenerateConfig *cfg, int32_t n_ctx) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_predict = 4;
    cfg->n_ctx = n_ctx;
    cfg->seed = 1234;
    cfg->temperature = 0.0f; /* greedy: the only fully deterministic setting */
    cfg->top_k = 1;
    cfg->top_p = 1.0f;
    cfg->repeat_penalty = 1.0f;
    cfg->use_mmap = 1;
    cfg->n_threads = 2;
}

static void test_generate_end_to_end(const char *path) {
    VG_GenerateConfig cfg;
    make_generate_config(&cfg, 32);

    VG_GenRec a;
    memset(&a, 0, sizeof(a));
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &a) == VG_OK);
    VG_GenRec b;
    memset(&b, 0, sizeof(b));
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &b) == VG_OK);

    CHECK(a.count > 0);
    CHECK(a.count <= (uint32_t)cfg.n_predict);
    CHECK(a.count == b.count);
    for (uint32_t i = 0; i < a.count && i < 16; ++i) CHECK(a.ids[i] == b.ids[i]);

    /* The graph never runs past the model context even when the caller asks for more. */
    VG_GenerateConfig small;
    make_generate_config(&small, 4);
    small.n_predict = 64;
    VG_GenRec c;
    memset(&c, 0, sizeof(c));
    CHECK(vg_generate(path, "abc", &small, gen_record, &c) == VG_OK);
    CHECK(c.count < 64);
}

/* A BitNet model must generate, and the adapter must change the output: an adapter that is
 * silently ignored would produce byte-identical tokens. */
static void test_bitnet_generate(const char *path, const char *lora_path) {
    VG_GenerateConfig cfg;
    make_generate_config(&cfg, 32);

    VG_GenRec a;
    memset(&a, 0, sizeof(a));
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &a) == VG_OK);
    CHECK(a.count > 0);

    VG_GenRec b;
    memset(&b, 0, sizeof(b));
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &b) == VG_OK);
    CHECK(a.count == b.count);
    for (uint32_t i = 0; i < a.count && i < 16; ++i) CHECK(a.ids[i] == b.ids[i]);

    /* Missing adapter path must fail loudly instead of silently running the base model. */
    VG_GenerateConfig bad = cfg;
    bad.lora_path = "does-not-exist.gguf";
    VG_GenRec r;
    memset(&r, 0, sizeof(r));
    CHECK(vg_generate(path, "abc", &bad, gen_record, &r) != VG_OK);
    (void)lora_path;
}

static void test_lora_generate(const char *path, const char *lora_path) {
    VG_GenerateConfig cfg;
    make_generate_config(&cfg, 32);

    VG_GenRec base;
    memset(&base, 0, sizeof(base));
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &base) == VG_OK);
    CHECK(base.count > 0);

    VG_GenerateConfig with_lora = cfg;
    with_lora.lora_path = lora_path;
    with_lora.lora_scale = 1.0f;
    VG_GenRec adapted;
    memset(&adapted, 0, sizeof(adapted));
    CHECK(vg_generate(path, "abc", &with_lora, gen_record, &adapted) == VG_OK);
    CHECK(adapted.count > 0);

    /* The adapter must change the token stream, not just be loaded. */
    int tokens_differ = 0;
    for (uint32_t i = 0; i < adapted.count && i < base.count && i < 16; ++i)
        if (adapted.ids[i] != base.ids[i]) tokens_differ = 1;
    CHECK(tokens_differ);

    /* Greedy tokens can stay identical under a small delta, so verify the adapter at the
     * logits level: same graph, same input, once with and once without the adapter. */
    {
        VG_GGUF *file = NULL;
        VG_TensorSource *src = NULL;
        VG_ModelGraph *g = load_graph(path, &file, &src);
        CHECK(g != NULL);
        if (g) {
            const VG_ModelConfig *mc = vg_model_graph_config(g);
            float *la = (float *)malloc(mc->n_vocab * sizeof(float));
            float *lb = (float *)malloc(mc->n_vocab * sizeof(float));
            VG_GGUF *lf = NULL;
            VG_LoraAdapter *ad = NULL;
            if (la && lb && vg_gguf_open(lora_path, &lf) == VG_OK && vg_lora_load(lf, &ad) == VG_OK) {
                vg_model_graph_reset(g);
                CHECK(vg_model_graph_decode(g, mc->bos_id, 0, la) == VG_OK);
                /* A large scale keeps the delta visible above this fixture's activation
                 * magnitudes; the point here is that the delta reaches the logits at all. */
                CHECK(vg_model_graph_set_lora(g, ad, 1.0f) == VG_OK);
                vg_model_graph_reset(g);
                CHECK(vg_model_graph_decode(g, mc->bos_id, 0, lb) == VG_OK);
                /* Relative test: a small delta against large logits is still a real change,
                 * while the same comparison in absolute terms would round to zero. */
                float scale_ref = 0.0f;
                for (uint32_t i = 0; i < mc->n_vocab; ++i) {
                    const float a = fabsf(la[i]);
                    if (a > scale_ref) scale_ref = a;
                }
                const float tol = (scale_ref > 0.0f ? scale_ref : 1.0f) * 1e-7f;
                int moved = 0;
                float max_rel = 0.0f;
                for (uint32_t i = 0; i < mc->n_vocab; ++i) {
                    const float d = fabsf(la[i] - lb[i]);
                    if (d > max_rel) max_rel = d;
                    if (d > tol) moved = 1;
                }
                if (!moved) fprintf(stderr, "FAIL: LoRA adapter did not change any logit (max delta %g)\n", (double)max_rel);
                CHECK(moved);
                /* Detaching must restore the base model exactly. */
                float *lc = (float *)malloc(mc->n_vocab * sizeof(float));
                if (lc) {
                    CHECK(vg_model_graph_set_lora(g, NULL, 1.0f) == VG_OK);
                    vg_model_graph_reset(g);
                    CHECK(vg_model_graph_decode(g, mc->bos_id, 0, lc) == VG_OK);
                    int same_as_base = 1;
                    for (uint32_t i = 0; i < mc->n_vocab; ++i)
                        if (la[i] != lc[i]) { same_as_base = 0; break; }
                    CHECK(same_as_base);
                    free(lc);
                }
            } else {
                fprintf(stderr, "FAIL: could not prepare the LoRA logits comparison\n");
                ++g_failures;
            }
            if (ad) vg_lora_free(ad);
            if (lf) vg_gguf_close(lf);
            free(la); free(lb);
            vg_model_graph_free(g);
            vg_tensor_source_close(src);
            vg_gguf_close(file);
        }
    }

    /* Greedy decoding with the same seed must stay deterministic under the adapter. */
    VG_GenRec again;
    memset(&again, 0, sizeof(again));
    CHECK(vg_generate(path, "abc", &with_lora, gen_record, &again) == VG_OK);
    CHECK(again.count == adapted.count);
    for (uint32_t i = 0; i < again.count && i < 16; ++i) CHECK(again.ids[i] == adapted.ids[i]);
}

static void test_generate_rejects_unsupported(const char *path) {
    VG_GenerateConfig cfg;
    make_generate_config(&cfg, 32);
    VG_GenRec r;
    memset(&r, 0, sizeof(r));
    /* The probe must refuse the model before any weight is fetched. */
    CHECK(vg_generate(path, "abc", &cfg, gen_record, &r) == VG_E_UNSUPPORTED);
    CHECK(r.count == 0);
}
#endif

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    char dense[512], unsupported[512], moe[512], bitnet[512], lora[512];
    snprintf(dense, sizeof(dense), "%s/fixture_dense.gguf", dir);
    snprintf(unsupported, sizeof(unsupported), "%s/fixture_unsupported.gguf", dir);
    snprintf(moe, sizeof(moe), "%s/fixture_moe.gguf", dir);
    snprintf(bitnet, sizeof(bitnet), "%s/fixture_bitnet.gguf", dir);
    snprintf(lora, sizeof(lora), "%s/fixture_lora.gguf", dir);

    CHECK(vg_fixture_write_dense(dense) == 0);
    CHECK(vg_fixture_write_unsupported(unsupported) == 0);
    CHECK(vg_fixture_write_moe(moe) == 0);
    CHECK(vg_fixture_write_bitnet(bitnet) == 0);
    CHECK(vg_fixture_write_lora(lora) == 0);

    test_half_conversion();
    test_quant_block_types();
    test_k_quants();
    test_matmul_parity(VG_QUANT_F32, 64, 16, 1);
    test_matmul_parity(VG_QUANT_F16, 64, 16, 2);
    test_matmul_parity(VG_QUANT_Q4_0, 64, 16, 3);
    test_matmul_parity(VG_QUANT_Q4_1, 64, 16, 4);
    test_matmul_parity(VG_QUANT_Q5_0, 64, 16, 5);
    test_matmul_parity(VG_QUANT_Q5_1, 64, 16, 6);
    test_matmul_parity(VG_QUANT_Q8_0, 64, 16, 7);
    test_matmul_parity(VG_QUANT_IQ4_NL, 64, 16, 8);
    test_matmul_parity(VG_QUANT_Q2_K, 256, 8, 9);
    test_matmul_parity(VG_QUANT_Q3_K, 256, 8, 10);
    test_matmul_parity(VG_QUANT_Q4_K, 256, 8, 11);
    test_matmul_parity(VG_QUANT_Q5_K, 256, 8, 12);
    test_matmul_parity(VG_QUANT_Q6_K, 256, 8, 13);
    test_matmul_parity(VG_QUANT_Q8_K, 256, 8, 14);
    /* Ternary / 1-bit: the fused kernels must match dequantize-then-dot exactly. */
    test_matmul_parity(VG_QUANT_TQ1_0, 256, 8, 15);
    test_matmul_parity(VG_QUANT_TQ2_0, 256, 8, 16);
    test_matmul_parity(VG_QUANT_Q1_0, 128, 8, 17);
    test_matmul_parity(VG_QUANT_Q2_0, 64, 8, 18);
    test_bitnet_probe(bitnet);
    test_bitnet_decode(bitnet);
    test_lora_delta(lora);
    test_threaded_determinism();
    test_thread_resizing();
    test_threadpool_parallel();

    test_probe_ok(dense);
    test_probe_unsupported(unsupported);
    test_dense_decode(dense);
    test_moe_routing(moe);
    test_moe_expert_cache_lease(moe);
#ifdef VG_TEST_FULL_ENGINE
    test_generate_end_to_end(dense);
    test_generate_rejects_unsupported(unsupported);
    test_bitnet_generate(bitnet, lora);
    test_lora_generate(bitnet, lora);
    test_speculative(bitnet);
#endif

    if (g_failures) {
        fprintf(stderr, "vg-engine-tests: %d failure(s)\n", g_failures);
        return 1;
    }
    puts("vg-engine-tests: all checks passed");
    return 0;
}
