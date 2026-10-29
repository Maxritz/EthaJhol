#include "vg/lora.h"
#include "vg/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Scalar stack buffer for the low-rank projection. Real adapters use rank 8..128; anything
 * larger falls back to a heap buffer so the decode path stays allocation-free in practice. */
#define VG_LORA_STACK_RANK 256

typedef struct VG_LoraPair {
    char *base;      /* weight tensor name this target modifies, e.g. "blk.0.attn_q.weight" */
    float *a;        /* [rank][in_dim]  */
    float *b;        /* [out_dim][rank] */
    uint32_t rank;
    uint32_t in_dim;
    uint32_t out_dim;
    float scaling;
} VG_LoraPair;

struct VG_LoraAdapter {
    VG_GGUF *file;
    VG_LoraPair *pairs;
    size_t count;
    size_t capacity;
    uint32_t rank;
};

static char *dup_str(const char *s) {
    if (!s) return NULL;
    const size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

/* Split ".../lora_a" style names into the base weight name. Returns 1 on a match and writes
 * the base name (without the trailing dot) into `base`. */
static int split_lora_name(const char *name, char *base, size_t cap, int *is_a) {
    static const char *const suffixes[4][2] = {
        { ".lora_a", ".lora_b" }, { ".loraA", ".loraB" }, { ".lora_a.weight", ".lora_b.weight" },
        { NULL, NULL }
    };
    for (size_t s = 0; suffixes[s][0]; ++s) {
        const size_t la = strlen(suffixes[s][0]), lb = strlen(suffixes[s][1]);
        if (strlen(name) > la && strcmp(name + strlen(name) - la, suffixes[s][0]) == 0) {
            const size_t n = strlen(name) - la;
            if (n + 1 > cap) return 0;
            memcpy(base, name, n); base[n] = 0; *is_a = 1; return 1;
        }
        if (strlen(name) > lb && strcmp(name + strlen(name) - lb, suffixes[s][1]) == 0) {
            const size_t n = strlen(name) - lb;
            if (n + 1 > cap) return 0;
            memcpy(base, name, n); base[n] = 0; *is_a = 0; return 1;
        }
    }
    return 0;
}

static VG_LoraPair *find_pair(VG_LoraAdapter *ad, const char *base) {
    for (size_t i = 0; i < ad->count; ++i)
        if (strcmp(ad->pairs[i].base, base) == 0) return &ad->pairs[i];
    return NULL;
}

/* Prefer an exact match; otherwise accept the name with a trailing ".weight" added or
 * removed so legacy adapters ("blk.0.attn_q.loraA") still bind to "blk.0.attn_q.weight". */
static VG_LoraPair *resolve(const VG_LoraAdapter *ad, const char *weight_name) {
    VG_LoraAdapter *mutable_ad = (VG_LoraAdapter *)ad;
    VG_LoraPair *p = find_pair(mutable_ad, weight_name);
    if (p) return p;
    const size_t n = strlen(weight_name);
    char buf[192];
    if (n > 7 && strcmp(weight_name + n - 7, ".weight") == 0) {
        const size_t m = n - 7;
        if (m + 1 <= sizeof(buf)) {
            memcpy(buf, weight_name, m); buf[m] = 0;
            p = find_pair(mutable_ad, buf);
            if (p) return p;
        }
    }
    if (n + 8 <= sizeof(buf)) {
        snprintf(buf, sizeof(buf), "%s.weight", weight_name);
        p = find_pair(mutable_ad, buf);
        if (p) return p;
    }
    return NULL;
}

/* Dequantize a whole tensor into float32. LoRA A/B are normally F32 or F16; any supported
 * type works so an adapter quantized in place still loads. */
static float *read_float_tensor(VG_GGUF *file, const VG_GGUF_Tensor *t) {
    const VG_QuantInfo *qi = vg_quant_info(t->ggml_type);
    if (!qi || !qi->supported_scalar) return NULL;
    uint64_t elements = 1;
    for (uint32_t i = 0; i < t->n_dims; ++i) elements *= t->dims[i];
    if (!elements || elements > (uint64_t)1 << 32) return NULL;
    void *raw = malloc(t->nbytes ? t->nbytes : 1);
    if (!raw) return NULL;
    float *out = (float *)malloc((size_t)elements * sizeof(float));
    if (!out) { free(raw); return NULL; }
    if (vg_gguf_read(file, t, 0, raw, t->nbytes) != VG_OK ||
        vg_dequantize_row(t->ggml_type, raw, t->nbytes, out, (size_t)elements) != VG_OK) {
        free(raw); free(out); return NULL;
    }
    free(raw);
    return out;
}

VG_Status vg_lora_load(VG_GGUF *file, VG_LoraAdapter **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_LoraAdapter *ad = (VG_LoraAdapter *)calloc(1, sizeof(*ad));
    if (!ad) return VG_E_NOMEM;
    ad->file = file;

    const uint64_t tc = vg_gguf_tensor_count(file);
    for (uint64_t i = 0; i < tc; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(file, i);
        if (!t || t->n_dims < 2) continue;
        char base[160];
        int is_a = 0;
        if (!split_lora_name(t->name, base, sizeof(base), &is_a)) continue;
        if (is_a) {
            /* A: ggml order {in_dim, rank}. */
            const uint32_t in_dim = (uint32_t)t->dims[0];
            const uint32_t rank = (uint32_t)t->dims[1];
            if (!in_dim || !rank) continue;
            if (ad->count == ad->capacity) {
                const size_t cap = ad->capacity ? ad->capacity * 2u : 16u;
                VG_LoraPair *np = (VG_LoraPair *)realloc(ad->pairs, cap * sizeof(*np));
                if (!np) break;
                ad->pairs = np; ad->capacity = cap;
            }
            VG_LoraPair *p = &ad->pairs[ad->count];
            memset(p, 0, sizeof(*p));
            p->base = dup_str(base);
            p->a = read_float_tensor(file, t);
            p->rank = rank; p->in_dim = in_dim; p->scaling = 1.0f;
            if (!p->base || !p->a) { free(p->base); free(p->a); continue; }
            ++ad->count;
            if (rank > ad->rank) ad->rank = rank;
        } else {
            VG_LoraPair *p = find_pair(ad, base);
            if (!p) continue;   /* B without A: ignore, nothing can be applied */
            const uint32_t rank = (uint32_t)t->dims[0];
            const uint32_t out_dim = (uint32_t)t->dims[1];
            if (rank != p->rank) continue;
            free(p->b);
            p->b = read_float_tensor(file, t);
            p->out_dim = out_dim;
            if (!p->b) { p->out_dim = 0; continue; }
        }
    }

    /* Per-target scaling scalars ("<base>.scaling"), one element each. */
    for (uint64_t i = 0; i < tc; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(file, i);
        if (!t) continue;
        const size_t n = strlen(t->name);
        if (n < 9 || strcmp(t->name + n - 8, ".scaling") != 0) continue;
        char base[160];
        const size_t m = n - 8;
        if (m + 1 > sizeof(base)) continue;
        memcpy(base, t->name, m); base[m] = 0;
        VG_LoraPair *p = find_pair(ad, base);
        if (!p) continue;
        float *s = read_float_tensor(file, t);
        if (s) { if (isfinite(*s) && *s != 0.0f) p->scaling = *s; free(s); }
    }

    /* Drop incomplete pairs (A without B cannot produce a delta). */
    size_t keep = 0;
    for (size_t i = 0; i < ad->count; ++i) {
        if (ad->pairs[i].b && ad->pairs[i].out_dim) {
            if (keep != i) ad->pairs[keep] = ad->pairs[i];
            ++keep;
        } else {
            free(ad->pairs[i].base);
            free(ad->pairs[i].a);
            free(ad->pairs[i].b);
        }
    }
    ad->count = keep;
    ad->capacity = keep;
    if (ad->count == 0) { free(ad->pairs); ad->pairs = NULL; }

    *out = ad;
    return VG_OK;
}

void vg_lora_free(VG_LoraAdapter *ad) {
    if (!ad) return;
    for (size_t i = 0; i < ad->count; ++i) {
        free(ad->pairs[i].base);
        free(ad->pairs[i].a);
        free(ad->pairs[i].b);
    }
    free(ad->pairs);
    free(ad);
}

uint32_t vg_lora_rank(const VG_LoraAdapter *ad) { return ad ? ad->rank : 0; }

size_t vg_lora_target_count(const VG_LoraAdapter *ad) { return ad ? ad->count : 0; }

const char *vg_lora_target_name(const VG_LoraAdapter *ad, size_t index) {
    if (!ad || index >= ad->count) return NULL;
    return ad->pairs[index].base;
}

uint32_t vg_lora_target_rank(const VG_LoraAdapter *ad, const char *weight_name) {
    if (!ad || !weight_name) return 0;
    VG_LoraPair *p = resolve(ad, weight_name);
    return p ? p->rank : 0;
}

int vg_lora_apply_delta(const VG_LoraAdapter *ad, const char *weight_name,
                        const float *x, float *y, uint32_t out_dim, uint32_t in_dim,
                        float scale) {
    if (!ad || !weight_name || !x || !y) return -1;
    VG_LoraPair *p = resolve(ad, weight_name);
    if (!p) return 0;
    if (p->in_dim != in_dim || p->out_dim != out_dim || !p->a || !p->b) return 0;

    float stack_buf[VG_LORA_STACK_RANK];
    float *tmp = p->rank <= VG_LORA_STACK_RANK ? stack_buf : (float *)malloc(p->rank * sizeof(float));
    if (!tmp) return -1;

    /* tmp = A * x  (A is [rank][in_dim]) */
    for (uint32_t r = 0; r < p->rank; ++r) {
        const float *ar = p->a + (size_t)r * p->in_dim;
        float acc = 0.0f;
        for (uint32_t j = 0; j < p->in_dim; ++j) acc += ar[j] * x[j];
        tmp[r] = acc;
    }
    /* y += s * scaling * B * tmp  (B is [out_dim][rank]) */
    const float s = scale * p->scaling;
    for (uint32_t i = 0; i < p->out_dim; ++i) {
        const float *br = p->b + (size_t)i * p->rank;
        float acc = 0.0f;
        for (uint32_t r = 0; r < p->rank; ++r) acc += br[r] * tmp[r];
        y[i] += s * acc;
    }
    if (tmp != stack_buf) free(tmp);
    return 1;
}

VG_Status vg_lora_apply(const VG_LoraAdapter *ad, const char *tensor_name,
                        float *weight, uint32_t out_dim, uint32_t in_dim, float scale) {
    if (!ad || !tensor_name || !weight) return VG_E_INVALID;
    VG_LoraPair *p = resolve(ad, tensor_name);
    if (!p) return VG_E_UNSUPPORTED;
    if (p->in_dim != in_dim || p->out_dim != out_dim) return VG_E_FORMAT;

    const float s = scale * p->scaling;
    for (uint32_t i = 0; i < out_dim; ++i) {
        const float *br = p->b + (size_t)i * p->rank;
        float *wr = weight + (size_t)i * in_dim;
        for (uint32_t j = 0; j < in_dim; ++j) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < p->rank; ++r) acc += br[r] * p->a[(size_t)r * in_dim + j];
            wr[j] += s * acc;
        }
    }
    return VG_OK;
}
