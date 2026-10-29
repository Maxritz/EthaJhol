#include "fixture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Minimal GGUF writer --------------------------------------------------------- */
typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
} Buf;

typedef struct {
    char name[64];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    unsigned char *payload;
    size_t bytes;
    size_t offset;
} FixTensor;

typedef struct {
    Buf meta;
    FixTensor t[64];
    uint32_t n_tensors;
    uint32_t n_kv;
    uint32_t max_tensors;
} Writer;

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    return q;
}

static void reserve(Buf *b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < b->len + extra) cap *= 2;
    b->data = (unsigned char *)xrealloc(b->data, cap);
    b->cap = cap;
}

static void put(Buf *b, const void *src, size_t n) {
    if (!n) return;
    reserve(b, n);
    memcpy(b->data + b->len, src, n);
    b->len += n;
}
static void put_u32(Buf *b, uint32_t v) { put(b, &v, 4); }
static void put_u64(Buf *b, uint64_t v) { put(b, &v, 8); }
static void put_f32(Buf *b, float v) { put(b, &v, 4); }
static void put_str(Buf *b, const char *s) { uint64_t n = strlen(s); put_u64(b, n); put(b, s, (size_t)n); }

static void kv_str(Writer *w, const char *key, const char *value) {
    put_str(&w->meta, key); put_u32(&w->meta, 8); put_str(&w->meta, value); ++w->n_kv;
}
static void kv_u32(Writer *w, const char *key, uint32_t value) {
    put_str(&w->meta, key); put_u32(&w->meta, 4); put_u32(&w->meta, value); ++w->n_kv;
}
static void kv_f32(Writer *w, const char *key, float value) {
    put_str(&w->meta, key); put_u32(&w->meta, 6); put_f32(&w->meta, value); ++w->n_kv;
}
static void kv_str_array(Writer *w, const char *key, const char *const *items, uint32_t count) {
    put_str(&w->meta, key); put_u32(&w->meta, 9); put_u32(&w->meta, 8); put_u64(&w->meta, count);
    for (uint32_t i = 0; i < count; ++i) put_str(&w->meta, items[i]);
    ++w->n_kv;
}

static size_t round32(size_t v) { return (v + 31u) / 32u * 32u; }

static int write_file(Writer *w, const char *path) {
    Buf out = {0};
    put_u32(&out, 0x46554747u);
    put_u32(&out, 3);
    put_u64(&out, w->n_tensors);
    put_u64(&out, w->n_kv);
    put(&out, w->meta.data, w->meta.len);

    for (uint32_t i = 0; i < w->n_tensors; ++i) {
        FixTensor *t = &w->t[i];
        put_str(&out, t->name);
        put_u32(&out, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; ++d) put_u64(&out, t->dims[d]);
        put_u32(&out, t->type);
        put_u64(&out, t->offset);
    }
    while (out.len % 32u) { const unsigned char z = 0; put(&out, &z, 1); }

    FILE *f = fopen(path, "wb");
    if (!f) { free(out.data); return -1; }
    if (fwrite(out.data, 1, out.len, f) != out.len) { fclose(f); free(out.data); return -1; }
    size_t written = out.len;
    size_t total = out.len;
    for (uint32_t i = 0; i < w->n_tensors; ++i) total += round32(w->t[i].bytes);
    for (uint32_t i = 0; i < w->n_tensors; ++i) {
        FixTensor *t = &w->t[i];
        if (t->bytes && fwrite(t->payload, 1, t->bytes, f) != t->bytes) { fclose(f); free(out.data); return -1; }
        written += t->bytes;
        while (written % 32u) { fputc(0, f); ++written; }
    }
    fclose(f);
    free(out.data);
    (void)total;
    return 0;
}

/* Computes each tensor offset (32-byte aligned) before the file is written. */
static void layout(Writer *w) {
    size_t cursor = 0;
    for (uint32_t i = 0; i < w->n_tensors; ++i) {
        w->t[i].offset = cursor;
        cursor = round32(cursor + w->t[i].bytes);
    }
}

/* --- Payload patterns ------------------------------------------------------------ */

/* Correct round-to-nearest-even float32 -> binary16. The fixtures used to derive the
 * half bits with a shift hack ((bits >> 13) & 0xFF etc.), which produced a NEGATIVE,
 * astronomically large scale (2^44 range) instead of the intended ~0.02: every synthetic
 * model then ran with activations in the thousands and a saturating SiLU. */
static uint16_t f16_bits(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;               /* underflow to zero (adequate here) */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);  /* overflow to infinity */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static void put_f16(unsigned char *dst, float v) {
    const uint16_t h = f16_bits(v);
    dst[0] = (unsigned char)(h & 0xFFu);
    dst[1] = (unsigned char)(h >> 8);
}

static void fill_q4_0(unsigned char *dst, size_t elements, uint32_t seed) {
    const size_t blocks = elements / 32u;
    uint32_t s = seed * 2654435761u + 977u;
    for (size_t b = 0; b < blocks; ++b) {
        unsigned char *blk = dst + b * 18u;
        const float d = 0.02f + 0.001f * (float)(s % 13u);
        put_f16(blk, d);
        for (int i = 0; i < 16; ++i) {
            s = s * 1664525u + 1013904223u;
            blk[2 + i] = (unsigned char)(s >> 24);
        }
    }
}

static void fill_f32(float *dst, size_t elements, uint32_t seed) {
    uint32_t s = seed * 22695477u + 7u;
    for (size_t i = 0; i < elements; ++i) {
        s = s * 1664525u + 1013904223u;
        dst[i] = ((float)((s >> 9) % 2001u) - 1000.0f) / 1000.0f * 0.5f;
    }
}

static void add_q4(Writer *w, const char *name, uint32_t in_dim, uint32_t out_dim, uint32_t seed) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 2;
    t->dims[0] = in_dim; t->dims[1] = out_dim;
    t->type = 2; /* Q4_0 */
    t->bytes = (size_t)out_dim * (in_dim / 32u) * 18u;
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    fill_q4_0(t->payload, (size_t)in_dim * out_dim, seed);
}

static void add_q4_experts(Writer *w, const char *name, uint32_t in_dim, uint32_t out_dim,
                           uint32_t n_expert, uint32_t seed) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 3;
    t->dims[0] = in_dim; t->dims[1] = out_dim; t->dims[2] = n_expert;
    t->type = 2;
    t->bytes = (size_t)n_expert * out_dim * (in_dim / 32u) * 18u;
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    for (uint32_t e = 0; e < n_expert; ++e)
        fill_q4_0(t->payload + (size_t)e * out_dim * (in_dim / 32u) * 18u,
                  (size_t)in_dim * out_dim, seed + e * 31u);
}

static void add_f32_2d(Writer *w, const char *name, uint32_t in_dim, uint32_t out_dim, uint32_t seed) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 2;
    t->dims[0] = in_dim; t->dims[1] = out_dim;
    t->type = 0; /* F32 */
    t->bytes = (size_t)in_dim * out_dim * sizeof(float);
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    fill_f32((float *)t->payload, (size_t)in_dim * out_dim, seed);
}

/* TQ2_0: 66-byte block (64 packed bytes + fp16 scale) holding 256 ternary weights.
 * Every byte is deterministic and the scale is chosen small so the model stays finite. */
static void fill_tq2_0(unsigned char *dst, size_t elements, uint32_t seed) {
    const size_t blocks = elements / 256u;
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t b = 0; b < blocks; ++b) {
        unsigned char *blk = dst + b * 66u;
        /* Kept small: 256 ternary values per row at a larger scale pushes activations into the
         * thousands, which hides a small adapter delta inside float rounding. */
        const float d = 0.0012f + 0.0002f * (float)(s % 7u);
        put_f16(blk + 64, d);
        for (int i = 0; i < 64; ++i) {
            s = s * 1664525u + 1013904223u;
            blk[i] = (unsigned char)(s >> 24);
        }
    }
}

static void add_tq2(Writer *w, const char *name, uint32_t in_dim, uint32_t out_dim, uint32_t seed) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 2;
    t->dims[0] = in_dim; t->dims[1] = out_dim;
    t->type = 35; /* TQ2_0 */
    t->bytes = (size_t)out_dim * (in_dim / 256u) * 66u;
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    fill_tq2_0(t->payload, (size_t)in_dim * out_dim, seed);
}

/* 1-D F32 vector with an explicit value (norm weights, per-tensor scales). */
static void add_scalar(Writer *w, const char *name, float value) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 1;
    t->dims[0] = 1;
    t->type = 0;
    t->bytes = sizeof(float);
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    memcpy(t->payload, &value, sizeof(float));
}

static void add_norm(Writer *w, const char *name, uint32_t dim) {
    FixTensor *t = &w->t[w->n_tensors++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->n_dims = 1;
    t->dims[0] = dim;
    t->type = 0;
    t->bytes = (size_t)dim * sizeof(float);
    t->payload = (unsigned char *)calloc(1, t->bytes);
    if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    float *p = (float *)t->payload;
    for (uint32_t i = 0; i < dim; ++i) p[i] = 1.0f;
}

static void common_meta(Writer *w, const char *arch, uint32_t layers, uint32_t embd,
                        uint32_t heads, uint32_t kv_heads, uint32_t ff, uint32_t vocab,
                        uint32_t ctx, uint32_t head_dim, uint32_t n_expert, uint32_t n_used) {
    char key[96];
    kv_str(w, "general.architecture", arch);
    kv_str(w, "general.name", "vg-fixture");
    kv_u32(w, "general.alignment", 32);
    snprintf(key, sizeof(key), "%s.context_length", arch); kv_u32(w, key, ctx);
    snprintf(key, sizeof(key), "%s.block_count", arch); kv_u32(w, key, layers);
    snprintf(key, sizeof(key), "%s.embedding_length", arch); kv_u32(w, key, embd);
    snprintf(key, sizeof(key), "%s.feed_forward_length", arch); kv_u32(w, key, ff);
    snprintf(key, sizeof(key), "%s.attention.head_count", arch); kv_u32(w, key, heads);
    snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch); kv_u32(w, key, kv_heads);
    snprintf(key, sizeof(key), "%s.attention.key_length", arch); kv_u32(w, key, head_dim);
    snprintf(key, sizeof(key), "%s.rope.dimension_count", arch); kv_u32(w, key, head_dim);
    snprintf(key, sizeof(key), "%s.rope.freq_base", arch); kv_f32(w, key, 10000.0f);
    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", arch); kv_f32(w, key, 1e-5f);
    snprintf(key, sizeof(key), "%s.vocab_size", arch); kv_u32(w, key, vocab);
    if (n_expert > 1) {
        snprintf(key, sizeof(key), "%s.expert_count", arch); kv_u32(w, key, n_expert);
        snprintf(key, sizeof(key), "%s.expert_used_count", arch); kv_u32(w, key, n_used);
    }
    kv_str(w, "tokenizer.ggml.model", "llama");
    kv_u32(w, "tokenizer.ggml.bos_token_id", 1);
    kv_u32(w, "tokenizer.ggml.eos_token_id", 2);
    static const char *tokens[32] = {
        "<unk>", "<s>", "</s>", "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u",
        "v", "w", "x", "y", "z", " ", ".", ","
    };
    kv_str_array(w, "tokenizer.ggml.tokens", tokens, vocab < 32 ? vocab : 32);
}

static void init_writer(Writer *w) { memset(w, 0, sizeof(*w)); w->max_tensors = 64; }

static void free_writer(Writer *w) {
    free(w->meta.data);
    for (uint32_t i = 0; i < w->n_tensors; ++i) free(w->t[i].payload);
}

/* --- Public fixtures ------------------------------------------------------------- */
int vg_fixture_write_dense(const char *path) {
    Writer w;
    init_writer(&w);
    const uint32_t D = VG_FIX_DENSE_EMBD, H = VG_FIX_DENSE_HEADS, KV = VG_FIX_DENSE_KV;
    const uint32_t FF = VG_FIX_DENSE_FF, V = VG_FIX_DENSE_VOCAB, HD = D / H;
    common_meta(&w, "llama", VG_FIX_DENSE_LAYERS, D, H, KV, FF, V, VG_FIX_DENSE_CTX, HD, 1, 1);
    add_f32_2d(&w, "token_embd.weight", D, V, 11);
    add_norm(&w, "output_norm.weight", D);
    add_f32_2d(&w, "output.weight", D, V, 12);
    for (uint32_t l = 0; l < VG_FIX_DENSE_LAYERS; ++l) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l); add_norm(&w, name, D);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l); add_q4(&w, name, D, H * HD, 100 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l); add_q4(&w, name, D, KV * HD, 101 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l); add_q4(&w, name, D, KV * HD, 102 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l); add_q4(&w, name, H * HD, D, 103 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l); add_norm(&w, name, D);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l); add_q4(&w, name, D, FF, 104 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l); add_q4(&w, name, D, FF, 105 + l * 10);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l); add_q4(&w, name, FF, D, 106 + l * 10);
    }
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}

int vg_fixture_write_unsupported(const char *path) {
    Writer w;
    init_writer(&w);
    /* in_dim must be a multiple of 256 so the file parses; the quant type is what fails. */
    const uint32_t D = 256, H = 4, KV = 2, FF = 256, V = 32, HD = D / H;
    common_meta(&w, "llama", 1, D, H, KV, FF, V, 128, HD, 1, 1);
    add_f32_2d(&w, "token_embd.weight", D, V, 21);
    add_norm(&w, "output_norm.weight", D);
    add_f32_2d(&w, "output.weight", D, V, 22);
    add_norm(&w, "blk.0.attn_norm.weight", D);
    /* Q5_K is implemented; use IQ2_XXS (66 bytes / 256 values) to exercise the rejection. */
    {
        FixTensor *t = &w.t[w.n_tensors++];
        snprintf(t->name, sizeof(t->name), "blk.0.attn_q.weight");
        t->n_dims = 2;
        t->dims[0] = D; t->dims[1] = H * HD;
        t->type = 16; /* IQ2_XXS */
        t->bytes = (size_t)(D / 256u) * (H * HD) * 66u;
        t->payload = (unsigned char *)calloc(1, t->bytes);
        if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
    }
    add_q4(&w, "blk.0.attn_k.weight", D, KV * HD, 31);
    add_q4(&w, "blk.0.attn_v.weight", D, KV * HD, 32);
    add_q4(&w, "blk.0.attn_output.weight", H * HD, D, 33);
    add_norm(&w, "blk.0.ffn_norm.weight", D);
    add_q4(&w, "blk.0.ffn_gate.weight", D, FF, 34);
    add_q4(&w, "blk.0.ffn_up.weight", D, FF, 35);
    add_q4(&w, "blk.0.ffn_down.weight", FF, D, 36);
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}

int vg_fixture_write_spm(const char *path) {
    Writer w;
    init_writer(&w);
    /* Ids match the VG_FIX_SPM_* enum in fixture.h. U+2581 is written as escapes so the file
     * does not depend on the source encoding. */
    static const char *tokens[24] = {
        "<unk>", "<s>", "</s>", "The",
        "\xE2\x96\x81" "capital", "capital", "<0x0A>", "\xE2\x96\x81" "of",
        "\xE2\x96\x81" "France", "\xE2\x96\x81" "is", "\xE2\x96\x81", "c",
        "a", "p", "i", "t",
        "l", "<0xE2>", "<0x82>", "<0xAC>",
        "T", "h", "e", "\xE2\x96\x81" "Paris"
    };
    /* Rank order matters: the chain c+a, ca+p, ... is the only way "capital" becomes one
     * token, so a tokenizer that ignores merges emits single characters instead. The marked
     * chain builds U+2581+capital the way real SentencePiece vocabularies do, which is also
     * what distinguishes a correct U+2581 encoder from one that assumes U+0120. */
    /* Marker merges rank first, exactly as in real tables: if the bare c+a merge won first,
     * the marker could never bind to the word and "\u2581capital" would split in two. */
    static const char *merges[15] = {
        "\xE2\x96\x81 c", "\xE2\x96\x81" "c a", "\xE2\x96\x81" "ca p",
        "\xE2\x96\x81" "cap i", "\xE2\x96\x81" "capi t", "\xE2\x96\x81" "capit a",
        "\xE2\x96\x81" "capita l",
        "c a", "ca p", "cap i", "capi t",
        "capit a", "capita l", "T h", "Th e"
    };
    kv_str(&w, "general.architecture", "llama");
    kv_str(&w, "general.name", "vg-spm-fixture");
    kv_str(&w, "tokenizer.ggml.model", "llama");
    kv_u32(&w, "tokenizer.ggml.bos_token_id", 1);
    kv_u32(&w, "tokenizer.ggml.eos_token_id", 2);
    kv_str_array(&w, "tokenizer.ggml.tokens", tokens, 24);
    kv_str_array(&w, "tokenizer.ggml.merges", merges, 15);
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}

/* BitNet b1.58 shape: ternary TQ2_0 projections, plus the two per-layer sub-norm RMSNorms
 * (attn over the context, ffn over the SwiGLU output) that the architecture requires. */
int vg_fixture_write_bitnet(const char *path) {
    Writer w;
    init_writer(&w);
    const uint32_t D = VG_FIX_BITNET_EMBD, H = VG_FIX_BITNET_HEADS, KV = VG_FIX_BITNET_KV;
    const uint32_t FF = VG_FIX_BITNET_FF, V = VG_FIX_BITNET_VOCAB, HD = D / H;
    common_meta(&w, "bitnet", VG_FIX_BITNET_LAYERS, D, H, KV, FF, V, VG_FIX_BITNET_CTX, HD, 1, 1);
    add_f32_2d(&w, "token_embd.weight", D, V, 201);
    add_norm(&w, "output_norm.weight", D);
    add_f32_2d(&w, "output.weight", D, V, 202);
    for (uint32_t l = 0; l < VG_FIX_BITNET_LAYERS; ++l) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l); add_norm(&w, name, D);
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l); add_tq2(&w, name, D, H * HD, 210 + l);
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l); add_tq2(&w, name, D, KV * HD, 211 + l);
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l); add_tq2(&w, name, D, KV * HD, 212 + l);
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l); add_tq2(&w, name, H * HD, D, 213 + l);
        snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", l); add_norm(&w, name, D);
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l); add_norm(&w, name, D);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l); add_tq2(&w, name, D, FF, 214 + l);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l); add_tq2(&w, name, D, FF, 215 + l);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l); add_tq2(&w, name, FF, D, 216 + l);
        snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", l); add_norm(&w, name, FF);
        if (l == 0) { snprintf(name, sizeof(name), "blk.%u.attn_q.scale", l); add_scalar(&w, name, 0.5f); }
    }
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}

/* LoRA adapter for the BitNet fixture: two targets, one with an explicit `.scaling`.
 * A is stored {in_dim, rank} and B {rank, out_dim}, matching ggml's convention. */
int vg_fixture_write_lora(const char *path) {
    Writer w;
    init_writer(&w);
    const uint32_t D = VG_FIX_BITNET_EMBD, FF = VG_FIX_BITNET_FF, R = 4;
    kv_str(&w, "general.architecture", "lora");
    kv_str(&w, "general.name", "vg-lora-fixture");
    add_f32_2d(&w, "blk.0.attn_q.weight.lora_a", D, R, 301);
    add_f32_2d(&w, "blk.0.attn_q.weight.lora_b", R, D, 302);
    add_scalar(&w, "blk.0.attn_q.weight.scaling", 0.5f);
    add_f32_2d(&w, "blk.0.ffn_down.weight.lora_a", FF, R, 303);
    add_f32_2d(&w, "blk.0.ffn_down.weight.lora_b", R, D, 304);
    /* A target with an A but no B must be dropped, not applied with garbage. */
    add_f32_2d(&w, "blk.0.attn_v.weight.lora_a", D, R, 305);
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}

int vg_fixture_write_moe(const char *path) {
    Writer w;
    init_writer(&w);
    const uint32_t D = VG_FIX_MOE_EMBD, H = VG_FIX_MOE_HEADS, KV = VG_FIX_MOE_KV;
    const uint32_t FF = VG_FIX_MOE_FF, V = VG_FIX_MOE_VOCAB, HD = D / H;
    const uint32_t E = VG_FIX_MOE_EXPERTS, U = VG_FIX_MOE_USED;
    common_meta(&w, "mixtral", VG_FIX_MOE_LAYERS, D, H, KV, FF, V, VG_FIX_DENSE_CTX, HD, E, U);
    add_f32_2d(&w, "token_embd.weight", D, V, 41);
    add_norm(&w, "output_norm.weight", D);
    add_f32_2d(&w, "output.weight", D, V, 42);
    add_norm(&w, "blk.0.attn_norm.weight", D);
    add_q4(&w, "blk.0.attn_q.weight", D, H * HD, 43);
    add_q4(&w, "blk.0.attn_k.weight", D, KV * HD, 44);
    add_q4(&w, "blk.0.attn_v.weight", D, KV * HD, 45);
    add_q4(&w, "blk.0.attn_output.weight", H * HD, D, 46);
    add_norm(&w, "blk.0.ffn_norm.weight", D);
    /* Router: row e has constant value (e+1)/10, so expert 3 always ranks first. */
    {
        FixTensor *t = &w.t[w.n_tensors++];
        snprintf(t->name, sizeof(t->name), "blk.0.ffn_gate_inp.weight");
        t->n_dims = 2;
        t->dims[0] = D; t->dims[1] = E;
        t->type = 0;
        t->bytes = (size_t)D * E * sizeof(float);
        t->payload = (unsigned char *)calloc(1, t->bytes);
        if (!t->payload) { fprintf(stderr, "fixture: out of memory\n"); exit(1); }
        float *p = (float *)t->payload;
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t i = 0; i < D; ++i) p[e * D + i] = (float)(e + 1) / 10.0f;
    }
    add_q4_experts(&w, "blk.0.ffn_gate_exps.weight", D, FF, E, 50);
    add_q4_experts(&w, "blk.0.ffn_up_exps.weight", D, FF, E, 60);
    add_q4_experts(&w, "blk.0.ffn_down_exps.weight", FF, D, E, 70);
    layout(&w);
    int rc = write_file(&w, path);
    free_writer(&w);
    return rc;
}
