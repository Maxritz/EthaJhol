#include "vg/model_graph.h"
#include "vg/tensor_source.h"
#include "vg/quant.h"
#include "vg/cpu_ops.h"
#include "vg/lora.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Expert tensor kind; part of the cache key so gate/up/down never alias. */
enum { VG_EXPERT_GATE = 0, VG_EXPERT_UP = 1, VG_EXPERT_DOWN = 2 };

typedef struct VG_ExpertSlot {
    uint32_t layer;
    uint32_t expert;
    uint32_t kind;
    const void *data;
    size_t bytes;
    uint32_t ggml_type;
    uint32_t out_dim;
    uint32_t in_dim;
    uint64_t last_use;
    uint32_t hits;
} VG_ExpertSlot;

typedef struct VG_ExpertCache {
    VG_ExpertSlot *slots;
    uint32_t cap;
    uint32_t count;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
} VG_ExpertCache;

struct VG_ModelGraph {
    VG_GGUF *file;
    VG_TensorSource *source;
    VG_ModelConfig cfg;
    float *work_embd;
    float *work_ffn;
    float *work_qkv;
    uint32_t work_qkv_cap;
    float *work_att_out;
    uint32_t work_att_out_cap;
    float *work_head;
    float *work_res;
    /* Scratch reused by every layer and token: allocate once at load instead of per op. */
    float *work_scores;      /* [n_ctx] attention scores, one head at a time */
    uint32_t work_scores_cap;
    float *work_proj;        /* [n_embd] attention output projection / FFN down output */
    float *work_gate;        /* [n_ff] dense FFN gate */
    float *work_up;          /* [n_ff] dense FFN up */
    float *logits;
    float *rope_freqs;
    uint32_t n_rope_freqs;
    /* Per-position RoPE table, rebuilt only when the position changes. */
    float *rope_cos;
    float *rope_sin;
    int32_t rope_tab_pos;
    int rope_tab_valid;
    float **kv_k;
    float **kv_v;
    uint32_t *kv_dim;
    uint32_t kv_capacity;
    uint32_t kv_page_tokens;
    int debug;
    /* MoE */
    VG_ExpertCache cache;
    float *route_logits;
    float *route_probs;
    float *expert_gate;
    float *expert_up;
    float *expert_down;
    uint32_t route_cap;
    /* Optional LoRA adapter applied as y += scale * B * (A * x) inside matmul. */
    const VG_LoraAdapter *lora;
    float lora_scale;
#ifdef VG_HAS_VULKAN
    VG_VK *vk;
    struct VG_VKWeightSlot *vk_slots;
    uint32_t vk_slots_cap;
    uint32_t vk_slots_used;
    uint64_t vk_clock;
#endif
};

#ifdef VG_HAS_VULKAN
static void vk_cache_free(VG_ModelGraph *g);
#endif

static int debug_on(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("VG_DEBUG"); cached = (e && *e && strcmp(e, "0") != 0) ? 1 : 0; }
    return cached;
}

static const VG_GGUF_Tensor *find_tensor(const VG_ModelGraph *g, const char *name) {
    return vg_gguf_find_tensor(g->file, name);
}

static VG_Status acquire_weight(VG_ModelGraph *g, const char *name, VG_TensorLease *lease) {
    memset(lease, 0, sizeof(*lease));
    if (g->source) return vg_tensor_acquire(g->source, name, VG_TIER_STORAGE, lease);
    const VG_GGUF_Tensor *t = find_tensor(g, name);
    if (!t) return VG_E_INVALID;
    void *data = malloc(t->nbytes ? t->nbytes : 1);
    if (!data) return VG_E_NOMEM;
    VG_Status st = vg_gguf_read(g->file, t, 0, data, t->nbytes);
    if (st != VG_OK) { free(data); return st; }
    lease->tensor = t; lease->data = data; lease->size = t->nbytes; lease->tier = VG_TIER_HOST;
    lease->source = NULL; lease->ticket = 0;
    return VG_OK;
}

static void release_weight(VG_TensorLease *lease) {
    if (lease->source) vg_tensor_release(lease);
    else if (lease->data) free((void *)lease->data);
    memset(lease, 0, sizeof(*lease));
}

static VG_Status read_metadata_int(const VG_ModelGraph *g, const char *key, int32_t *out) {
    const char *v = vg_gguf_meta(g->file, key);
    if (!v) return VG_E_INVALID;
    const char *p = v;
    while (*p == '[' || *p == ' ' || *p == ',') ++p;
    char *end = NULL;
    long x = strtol(p, &end, 10);
    if (end == p) return VG_E_INVALID;
    *out = (int32_t)x;
    return VG_OK;
}

static VG_Status read_metadata_float(const VG_ModelGraph *g, const char *key, float *out) {
    const char *v = vg_gguf_meta(g->file, key);
    if (!v) return VG_E_INVALID;
    const char *p = v;
    while (*p == '[' || *p == ' ' || *p == ',') ++p;
    char *end = NULL;
    float val = strtof(p, &end);
    if (end == p || !isfinite(val)) return VG_E_INVALID;
    *out = val;
    return VG_OK;
}

/* --- Expert cache -------------------------------------------------------------- */
static void cache_init(VG_ExpertCache *c, uint32_t cap) {
    memset(c, 0, sizeof(*c));
    if (cap < 4) cap = 4;
    c->slots = (VG_ExpertSlot *)calloc(cap, sizeof(VG_ExpertSlot));
    c->cap = c->slots ? cap : 0;
}

static void cache_free(VG_ExpertCache *c) { free(c->slots); memset(c, 0, sizeof(*c)); }

static VG_ExpertSlot *cache_find(VG_ExpertCache *c, uint32_t layer, uint32_t expert, uint32_t kind) {
    for (uint32_t i = 0; i < c->count; ++i)
        if (c->slots[i].layer == layer && c->slots[i].expert == expert && c->slots[i].kind == kind)
            return &c->slots[i];
    return NULL;
}

static VG_ExpertSlot *cache_insert(VG_ExpertCache *c, VG_ExpertSlot value) {
    if (!c->slots || !c->cap) return NULL;
    uint32_t idx;
    if (c->count < c->cap) {
        idx = c->count++;
    } else {
        /* LRU victim */
        idx = 0;
        for (uint32_t i = 1; i < c->count; ++i)
            if (c->slots[i].last_use < c->slots[idx].last_use) idx = i;
        ++c->evictions;
    }
    value.last_use = ++c->clock;
    c->slots[idx] = value;
    return &c->slots[idx];
}

/* Bytes occupied by one expert inside a stacked (3-D) expert tensor. */
static size_t expert_stride(const VG_GGUF_Tensor *t, uint32_t out_dim, uint32_t in_dim) {
    const VG_QuantInfo *qi = vg_quant_info(t->ggml_type);
    if (!qi) return 0;
    uint64_t nels = (uint64_t)in_dim * out_dim;
    uint64_t blocks = nels / qi->block_size;
    return (size_t)(blocks * qi->bytes_per_block);
}

/* Resolve expert `e` of `blk.N.<kind>_exps.weight`, reusing the cache when the same expert
 * was used by an earlier token. Returns a pointer into the mmap'd tensor. */
static VG_Status acquire_expert(VG_ModelGraph *g, uint32_t layer, uint32_t kind,
                                uint32_t expert, uint32_t out_dim, uint32_t in_dim,
                                const void **data, size_t *bytes, uint32_t *ggml_type) {
    VG_ExpertSlot *slot = cache_find(&g->cache, layer, expert, kind);
    if (slot) {
        if (slot->out_dim == out_dim && slot->in_dim == in_dim) {
            ++slot->hits;
            slot->last_use = ++g->cache.clock;
            ++g->cache.hits;
            *data = slot->data;
            if (bytes) *bytes = slot->bytes;
            *ggml_type = slot->ggml_type;
            return VG_OK;
        }
    }
    static const char *const kinds[] = {"ffn_gate", "ffn_up", "ffn_down"};
    if (kind > VG_EXPERT_DOWN) return VG_E_INVALID;
    char name[160];
    snprintf(name, sizeof(name), "blk.%u.%s_exps.weight", layer, kinds[kind]);
    VG_TensorLease lease;
    VG_Status st = acquire_weight(g, name, &lease);
    if (st != VG_OK) return st;
    const size_t stride = expert_stride(lease.tensor, out_dim, in_dim);
    const uint64_t experts = lease.tensor->n_dims >= 3 ? lease.tensor->dims[2] : 1;
    if (expert >= experts || stride == 0 || (uint64_t)(expert + 1) * stride > lease.size) {
        release_weight(&lease);
        return VG_E_FORMAT;
    }
    VG_ExpertSlot value;
    memset(&value, 0, sizeof(value));
    value.layer = layer;
    value.expert = expert;
    value.kind = kind;
    value.data = (const unsigned char *)lease.data + (size_t)expert * stride;
    value.bytes = stride;
    value.ggml_type = lease.tensor->ggml_type;
    value.out_dim = out_dim;
    value.in_dim = in_dim;
    ++g->cache.misses;
    (void)cache_insert(&g->cache, value);
    *data = value.data;
    if (bytes) *bytes = value.bytes;
    *ggml_type = value.ggml_type;
    /* The mmap tier holds no reference; drop the lease we took for the lookup. */
    release_weight(&lease);
    return VG_OK;
}

/* --- Model loading -------------------------------------------------------------- */
VG_Status vg_model_graph_load(VG_GGUF *file, VG_TensorSource *source, VG_ModelGraph **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_ModelGraph *g = (VG_ModelGraph *)calloc(1, sizeof(*g));
    if (!g) return VG_E_NOMEM;
    g->file = file; g->source = source;
    g->debug = debug_on();

    const char *arch = vg_gguf_meta(file, "general.architecture");
    if (!arch) { free(g); return VG_E_FORMAT; }

    char key[128];
    int32_t i32; float f32;

    #define READ_INT(field, name) do { i32 = 0; snprintf(key, sizeof(key), "%s." #name, arch); if (read_metadata_int(g, key, &i32) == VG_OK) g->cfg.field = (uint32_t)i32; } while (0)
    #define READ_FLOAT(field, name) do { f32 = 0; snprintf(key, sizeof(key), "%s." #name, arch); if (read_metadata_float(g, key, &f32) == VG_OK) g->cfg.field = f32; } while (0)

    READ_INT(n_vocab, vocab_size);
    READ_INT(n_embd, embedding_length);
    READ_INT(n_head, attention.head_count);
    READ_INT(n_head_kv, attention.head_count_kv);
    READ_INT(n_layer, block_count);
    READ_INT(n_ff, feed_forward_length);
    READ_INT(n_rot, rope.dimension_count);
    READ_INT(head_dim, attention.key_length);

    const VG_GGUF_Tensor *te = find_tensor(g, "token_embd.weight");
    if (g->cfg.n_vocab == 0 && te && te->n_dims >= 2) g->cfg.n_vocab = (uint32_t)te->dims[1];
    if (g->cfg.n_embd == 0 && te && te->n_dims >= 1) g->cfg.n_embd = (uint32_t)te->dims[0];
    if (g->cfg.n_ff == 0) {
        const VG_GGUF_Tensor *fd = find_tensor(g, "blk.0.ffn_down.weight");
        if (!fd) fd = find_tensor(g, "blk.0.ffn_down_exps.weight");
        if (fd && fd->n_dims >= 1) g->cfg.n_ff = (uint32_t)fd->dims[0];
    }
    if (g->cfg.n_head_kv == 0) g->cfg.n_head_kv = g->cfg.n_head;
    if (g->cfg.head_dim == 0) {
        const VG_GGUF_Tensor *q = find_tensor(g, "blk.0.attn_q.weight");
        if (q && q->n_dims >= 2 && g->cfg.n_head) g->cfg.head_dim = (uint32_t)(q->dims[1] / g->cfg.n_head);
        if (g->cfg.head_dim == 0 && g->cfg.n_head) g->cfg.head_dim = g->cfg.n_embd / g->cfg.n_head;
    }
    if (g->cfg.n_rot == 0) g->cfg.n_rot = g->cfg.head_dim;
    if (g->cfg.n_rot > g->cfg.head_dim) g->cfg.n_rot = g->cfg.head_dim;
    READ_FLOAT(rms_eps, attention.layer_norm_rms_epsilon);
    if (!isfinite(g->cfg.rms_eps) || g->cfg.rms_eps <= 0.0f) g->cfg.rms_eps = 1e-5f;
    READ_FLOAT(freq_base, rope.freq_base);
    if (!isfinite(g->cfg.freq_base) || g->cfg.freq_base < 1.0f) g->cfg.freq_base = 10000.0f;

    /* Optional precomputed RoPE frequencies. */
    {
        const VG_GGUF_Tensor *rf = find_tensor(g, "rope_freqs.weight");
        if (rf && rf->n_dims >= 1 && rf->ggml_type == 0) {
            uint32_t n = (uint32_t)rf->dims[0];
            float *buf = (float *)malloc(n * sizeof(float));
            if (buf && vg_gguf_read(g->file, rf, 0, buf, n * sizeof(float)) == VG_OK) {
                int valid = 1;
                for (uint32_t i = 0; i < n; ++i)
                    if (!isfinite(buf[i]) || buf[i] <= 0.0f || buf[i] > 1.0e6f) { valid = 0; break; }
                if (valid) { g->rope_freqs = buf; g->n_rope_freqs = n; }
                else free(buf);
            } else {
                free(buf);
            }
        }
    }

    READ_INT(n_expert, expert_count);
    READ_INT(n_expert_used, expert_used_count);
    if (g->cfg.n_expert == 0) g->cfg.n_expert = 1;
    if (g->cfg.n_expert_used == 0) g->cfg.n_expert_used = 1;
    if (g->cfg.n_expert_used > g->cfg.n_expert) g->cfg.n_expert_used = g->cfg.n_expert;

    {
        int32_t n_ctx = 0;
        snprintf(key, sizeof(key), "%s.context_length", arch);
        if (read_metadata_int(g, key, &n_ctx) == VG_OK && n_ctx > 0) g->cfg.n_ctx = (uint32_t)n_ctx;
        else g->cfg.n_ctx = 2048;
    }
    {
        snprintf(key, sizeof(key), "%s.attention.sliding_window", arch);
        int32_t sw = 0;
        if (read_metadata_int(g, key, &sw) == VG_OK && sw > 0) g->cfg.sliding_window = (uint32_t)sw;
        else g->cfg.sliding_window = 0;
    }

    snprintf(key, sizeof(key), "tokenizer.ggml.bos_token_id"); read_metadata_int(g, key, &g->cfg.bos_id);
    snprintf(key, sizeof(key), "tokenizer.ggml.eos_token_id"); read_metadata_int(g, key, &g->cfg.eos_id);

    g->cfg.has_biases = find_tensor(g, "blk.0.attn_q.bias") != NULL;
    g->cfg.has_gate_up = find_tensor(g, "blk.0.ffn_gate.weight") != NULL;
    /* BitNet b1.58 carries two extra per-layer RMSNorm weights: one on the attention
     * context before the output projection and one on the SwiGLU output before ffn_down. */
    g->cfg.has_sub_norm = find_tensor(g, "blk.0.attn_sub_norm.weight") != NULL;

    if (g->debug) {
        fprintf(stderr, "[vg] config arch=%s vocab=%u embd=%u heads=%u/%u layers=%u ff=%u head_dim=%u rot=%u ctx=%u experts=%u/%u eps=%.2e theta=%.1f\n",
                arch, g->cfg.n_vocab, g->cfg.n_embd, g->cfg.n_head, g->cfg.n_head_kv, g->cfg.n_layer,
                g->cfg.n_ff, g->cfg.head_dim, g->cfg.n_rot, g->cfg.n_ctx, g->cfg.n_expert, g->cfg.n_expert_used,
                g->cfg.rms_eps, g->cfg.freq_base);
    }

    if (!g->cfg.n_vocab || !g->cfg.n_embd || !g->cfg.n_head || !g->cfg.n_layer || !g->cfg.head_dim) {
        free(g);
        return VG_E_FORMAT;
    }

    const uint32_t D = g->cfg.n_embd;
    const uint32_t FF = g->cfg.n_ff ? g->cfg.n_ff : 1;
    const uint32_t HD = g->cfg.head_dim;
    const uint32_t q_total = g->cfg.n_head * HD + 2 * g->cfg.n_head_kv * HD;

    g->work_embd = (float *)malloc(D * sizeof(float));
    g->work_res = (float *)malloc(D * sizeof(float));
    g->work_ffn = (float *)malloc(FF * sizeof(float));
    g->work_qkv = (float *)malloc(q_total * sizeof(float));
    g->work_qkv_cap = q_total;
    g->work_att_out = (float *)calloc(g->cfg.n_head * HD + 1, sizeof(float));
    g->work_att_out_cap = g->cfg.n_head * HD + 1;
    g->work_head = (float *)malloc(HD * sizeof(float));
    g->work_scores = (float *)malloc((size_t)g->cfg.n_ctx * sizeof(float));
    g->work_scores_cap = g->work_scores ? g->cfg.n_ctx : 0;
    g->work_proj = (float *)malloc(D * sizeof(float));
    g->work_gate = (float *)malloc(FF * sizeof(float));
    g->work_up = (float *)malloc(FF * sizeof(float));
    g->logits = (float *)malloc((size_t)g->cfg.n_vocab * sizeof(float));
    g->rope_cos = (float *)malloc((HD + 1) * sizeof(float));
    g->rope_sin = (float *)malloc((HD + 1) * sizeof(float));

    const uint32_t kv_page = 64;
    const uint32_t n_pages = (g->cfg.n_ctx + kv_page - 1) / kv_page;
    g->kv_page_tokens = kv_page;
    g->kv_capacity = n_pages;
    g->kv_k = (float **)calloc((size_t)g->cfg.n_layer * n_pages, sizeof(float *));
    g->kv_v = (float **)calloc((size_t)g->cfg.n_layer * n_pages, sizeof(float *));
    g->kv_dim = (uint32_t *)calloc(g->cfg.n_layer, sizeof(uint32_t));

    if (!g->work_embd || !g->work_res || !g->work_ffn || !g->work_qkv || !g->work_att_out ||
        !g->work_head || !g->work_scores || !g->work_proj || !g->work_gate || !g->work_up ||
        !g->logits || !g->rope_cos || !g->rope_sin ||
        !g->kv_k || !g->kv_v || !g->kv_dim) {
        vg_model_graph_free(g);
        return VG_E_NOMEM;
    }

    /* Expert cache: keep a few tokens' worth of routed experts hot. */
    {
        uint32_t cap = g->cfg.n_expert_used * 4u;
        if (cap < 8u) cap = 8u;
        if (cap > 256u) cap = 256u;
        cache_init(&g->cache, cap);
        if (g->cfg.n_expert > 1) {
            g->route_cap = g->cfg.n_expert;
            g->route_logits = (float *)calloc(g->cfg.n_expert, sizeof(float));
            g->route_probs = (float *)calloc(g->cfg.n_expert, sizeof(float));
            g->expert_gate = (float *)calloc(FF, sizeof(float));
            g->expert_up = (float *)calloc(FF, sizeof(float));
            g->expert_down = (float *)calloc(D, sizeof(float));
            if (!g->route_logits || !g->route_probs || !g->expert_gate || !g->expert_up || !g->expert_down) {
                vg_model_graph_free(g);
                return VG_E_NOMEM;
            }
        }
    }

    *out = g;
    return VG_OK;
    #undef READ_INT
    #undef READ_FLOAT
}

void vg_model_graph_free(VG_ModelGraph *g) {
    if (!g) return;
    free(g->work_embd); free(g->work_ffn); free(g->work_qkv);
    free(g->work_att_out); free(g->work_head); free(g->work_res);
    free(g->work_scores); free(g->work_proj); free(g->work_gate); free(g->work_up);
    free(g->logits); free(g->rope_freqs);
    free(g->rope_cos); free(g->rope_sin);
    if (g->kv_k) {
        for (uint32_t i = 0; i < g->cfg.n_layer * g->kv_capacity; ++i) { free(g->kv_k[i]); free(g->kv_v[i]); }
        free(g->kv_k); free(g->kv_v);
    }
    free(g->kv_dim);
    cache_free(&g->cache);
#ifdef VG_HAS_VULKAN
    vk_cache_free(g);
#endif
    free(g->route_logits); free(g->route_probs);
    free(g->expert_gate); free(g->expert_up); free(g->expert_down);
    free(g);
}

void vg_model_graph_reset(VG_ModelGraph *g) {
    if (!g) return;
    for (uint32_t i = 0; i < g->cfg.n_layer * g->kv_capacity; ++i) {
        if (!g->kv_k[i]) continue;
        uint32_t d = g->kv_dim[i / g->kv_capacity];
        free(g->kv_k[i]); g->kv_k[i] = NULL;
        free(g->kv_v[i]); g->kv_v[i] = NULL;
        (void)d;
    }
    g->rope_tab_valid = 0;
}

const VG_ModelConfig *vg_model_graph_config(const VG_ModelGraph *g) { return g ? &g->cfg : NULL; }
const float *vg_model_graph_logits(const VG_ModelGraph *g) { return g ? g->logits : NULL; }

VG_Status vg_model_graph_set_lora(VG_ModelGraph *g, const VG_LoraAdapter *adapter, float scale) {
    if (!g) return VG_E_INVALID;
    g->lora = adapter;
    g->lora_scale = (isfinite(scale) && scale != 0.0f) ? scale : 1.0f;
    return VG_OK;
}

void vg_model_graph_expert_stats(VG_ModelGraph *g, uint64_t *hits, uint64_t *misses, uint64_t *evictions) {
    if (!g) return;
    if (hits) *hits = g->cache.hits;
    if (misses) *misses = g->cache.misses;
    if (evictions) *evictions = g->cache.evictions;
}

VG_Status vg_model_graph_set_expert_cache(VG_ModelGraph *g, uint32_t capacity) {
    if (!g) return VG_E_INVALID;
    cache_free(&g->cache);
    cache_init(&g->cache, capacity ? capacity : 8u);
    return g->cache.slots || capacity == 0 ? VG_OK : VG_E_NOMEM;
}

/* --- Embedding ------------------------------------------------------------------ */
VG_Status vg_model_graph_embed(VG_ModelGraph *g, int32_t token, float *embd_out) {
    if (!g || !embd_out || token < 0 || (uint32_t)token >= g->cfg.n_vocab) return VG_E_INVALID;
    const VG_GGUF_Tensor *t = find_tensor(g, "token_embd.weight");
    if (!t) return VG_E_INVALID;
    const uint32_t embd = g->cfg.n_embd;
    const VG_QuantInfo *qi = vg_quant_info(t->ggml_type);
    if (!qi || !qi->supported_scalar) return VG_E_UNSUPPORTED;
    const uint64_t row_bytes = (uint64_t)(embd / qi->block_size) * qi->bytes_per_block;
    if (t->dims[0] != embd) return VG_E_FORMAT;
    const uint64_t offset = (uint64_t)token * row_bytes;
    if (offset + row_bytes > t->nbytes) return VG_E_RANGE;
    unsigned char *row = (unsigned char *)malloc((size_t)row_bytes);
    if (!row) return VG_E_NOMEM;
    VG_Status st = vg_gguf_read(g->file, t, offset, row, (size_t)row_bytes);
    if (st == VG_OK) st = vg_dequantize_row(t->ggml_type, row, (size_t)row_bytes, embd_out, embd);
    free(row);
    return st;
}

/* --- Helpers -------------------------------------------------------------------- */
static void rope_table(VG_ModelGraph *g, int32_t pos) {
    const uint32_t n_rot = g->cfg.n_rot;
    const uint32_t HD = g->cfg.head_dim;
    if (g->rope_tab_valid && g->rope_tab_pos == pos) return;
    if (g->rope_freqs && g->n_rope_freqs >= n_rot) {
        const uint32_t half = HD / 2u;
        const uint32_t iters = n_rot < half ? n_rot : half;
        for (uint32_t i = 0; i < iters; ++i) {
            const float freq = (float)pos * g->rope_freqs[i];
            g->rope_cos[i] = cosf(freq);
            g->rope_sin[i] = sinf(freq);
        }
    } else {
        vg_cpu_rope_table(g->rope_cos, g->rope_sin, n_rot, HD, 1.0f, g->cfg.freq_base, pos);
    }
    g->rope_tab_pos = pos;
    g->rope_tab_valid = 1;
}

#ifdef VG_HAS_VULKAN
/* Device-resident weight cache: a weight is uploaded once and reused by every token.
 * Entries are evicted LRU-style when the device budget would be exceeded. */
typedef struct VG_VKWeightSlot {
    const void *key;
    uint32_t ggml_type;
    uint32_t out_dim;
    uint32_t in_dim;
    uint64_t last_use;
    VG_VKWeight *weight;
} VG_VKWeightSlot;

static VG_VKWeight *vk_weight_get(VG_ModelGraph *g, const void *key, size_t bytes,
                                  uint32_t type, uint32_t out_dim, uint32_t in_dim) {
    if (!g->vk || !g->vk_slots || !g->vk_slots_cap || !vg_vk_supports_type(type)) return NULL;
    for (uint32_t i = 0; i < g->vk_slots_used; ++i) {
        VG_VKWeightSlot *s = &g->vk_slots[i];
        if (s->key == key) { s->last_use = ++g->vk_clock; return s->weight; }
    }
    VG_VKWeightSlot *slot = NULL;
    if (g->vk_slots_used < g->vk_slots_cap) {
        slot = &g->vk_slots[g->vk_slots_used++];
    } else {
        uint32_t victim = 0;
        for (uint32_t i = 1; i < g->vk_slots_used; ++i)
            if (g->vk_slots[i].last_use < g->vk_slots[victim].last_use) victim = i;
        slot = &g->vk_slots[victim];
        vg_vk_weight_release(g->vk, slot->weight);
        memset(slot, 0, sizeof(*slot));
    }
    VG_VKWeight *w = NULL;
    if (vg_vk_weight_upload(g->vk, key, bytes, type, out_dim, in_dim, &w) != VG_OK) {
        if (g->vk_slots_used && slot == &g->vk_slots[g->vk_slots_used - 1]) --g->vk_slots_used;
        return NULL;
    }
    slot->key = key; slot->ggml_type = type; slot->out_dim = out_dim; slot->in_dim = in_dim;
    slot->last_use = ++g->vk_clock; slot->weight = w;
    return w;
}
#endif

/* Per-tensor output scale used by BitNet GGUFs: "blk.N.attn_q.scale" (a single F32).
 * llama.cpp multiplies the matmul result by it, so skipping it silently biases the layer. */
static void apply_tensor_scale(VG_ModelGraph *g, const char *weight_name, float *y, uint32_t out_dim) {
    if (!weight_name) return;
    const size_t n = strlen(weight_name);
    if (n < 7 || strcmp(weight_name + n - 7, ".weight") != 0) return;
    char name[176];
    if (n - 7 + 7 > sizeof(name)) return;
    memcpy(name, weight_name, n - 7);
    memcpy(name + (n - 7), ".scale", 7);
    VG_TensorLease l;
    if (acquire_weight(g, name, &l) != VG_OK) return;
    /* The scale may be F32 or F16; reading four bytes from a two-byte tensor would run past
     * the lease, so go through the dequantizer for exactly one element. */
    float s = 1.0f;
    if (l.size > 0 && vg_dequantize_row(l.tensor->ggml_type, l.data, (size_t)l.size, &s, 1) == VG_OK) {
        if (isfinite(s) && s != 0.0f) for (uint32_t i = 0; i < out_dim; ++i) y[i] *= s;
    }
    release_weight(&l);
}

/* y = W * x with the fastest available backend for the tensor type, then the optional LoRA
 * delta for that weight. `weight_name` may be NULL for tensors an adapter cannot target. */
static VG_Status matmul_named(VG_ModelGraph *g, const char *weight_name, const void *w, size_t wbytes,
                             uint32_t type, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    VG_Status st;
#ifdef VG_HAS_VULKAN
    st = VG_E_UNSUPPORTED;
    if (g->vk && vg_vk_supports_type(type)) {
        VG_VKWeight *rw = vk_weight_get(g, w, wbytes, type, out_dim, in_dim);
        if (rw && vg_vk_matvec(g->vk, rw, x, y, out_dim, in_dim) == VG_OK) st = VG_OK;
    }
#else
    (void)g; (void)wbytes;
    st = VG_E_UNSUPPORTED;
#endif
    if (st != VG_OK) st = vg_cpu_matmul(type, w, x, y, out_dim, in_dim);
    if (st != VG_OK) return st;
    if (weight_name) {
        apply_tensor_scale(g, weight_name, y, out_dim);
        if (g->lora) (void)vg_lora_apply_delta(g->lora, weight_name, x, y, out_dim, in_dim, g->lora_scale);
    }
    return VG_OK;
}

static VG_Status matmul(VG_ModelGraph *g, const void *w, size_t wbytes, uint32_t type,
                        const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matmul_named(g, NULL, w, wbytes, type, x, y, out_dim, in_dim);
}

/* --- Attention ------------------------------------------------------------------ */
static VG_Status attention_layer(VG_ModelGraph *g, uint32_t layer, float *hidden, int32_t pos) {
    const uint32_t D = g->cfg.n_embd;
    const uint32_t H = g->cfg.n_head;
    const uint32_t HKV = g->cfg.n_head_kv ? g->cfg.n_head_kv : 1;
    const uint32_t HD = g->cfg.head_dim;
    const uint32_t n_rot = g->cfg.n_rot;
    char name[160];
    VG_TensorLease l;
    VG_Status st;

    memcpy(g->work_res, hidden, D * sizeof(float));

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    uint32_t Q_dim = H * HD;
    uint32_t KV_dim = HKV * HD;
    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
    st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    if (l.tensor->n_dims >= 2 && l.tensor->dims[1]) Q_dim = (uint32_t)l.tensor->dims[1];
    if (Q_dim % H) { release_weight(&l); return VG_E_FORMAT; }
    const uint32_t head_dim = Q_dim / H;
    const uint32_t q_rot = n_rot < head_dim ? n_rot : head_dim;

    if (Q_dim + 2 * KV_dim > g->work_qkv_cap) {
        float *tmp = (float *)realloc(g->work_qkv, (Q_dim + 2 * KV_dim) * sizeof(float));
        if (!tmp) { release_weight(&l); return VG_E_NOMEM; }
        g->work_qkv = tmp; g->work_qkv_cap = Q_dim + 2 * KV_dim;
    }
    float *q = g->work_qkv;
    float *k = q + Q_dim;
    float *v = k + KV_dim;

    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, q, Q_dim, D);
    release_weight(&l);
    if (st != VG_OK) return st;

    if (g->cfg.has_biases) {
        snprintf(name, sizeof(name), "blk.%u.attn_q.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < Q_dim; ++i) q[i] += bias[i];
            release_weight(&l);
        }
    }

    /* Q/K per-head RMSNorm (Qwen3-style) is applied before RoPE. */
    snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", layer);
    int has_q_norm = acquire_weight(g, name, &l) == VG_OK;
    const float *q_norm = has_q_norm ? (const float *)l.data : NULL;
    if (has_q_norm) for (uint32_t h = 0; h < H; ++h) {
        float *hq = q + h * head_dim;
        float ss = 0.0f;
        for (uint32_t i = 0; i < head_dim; ++i) ss += hq[i] * hq[i];
        const float inv = 1.0f / sqrtf(ss / (float)head_dim + 1e-6f);
        for (uint32_t i = 0; i < head_dim; ++i) hq[i] = hq[i] * inv * q_norm[i];
    }

    rope_table(g, pos);
    vg_cpu_rope_apply_all(q, H, head_dim, q_rot, g->rope_cos, g->rope_sin);
    if (has_q_norm) release_weight(&l);

    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", layer);
    st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    if (l.tensor->n_dims >= 2 && l.tensor->dims[1]) KV_dim = (uint32_t)l.tensor->dims[1];
    if (KV_dim % HKV) { release_weight(&l); return VG_E_FORMAT; }
    const uint32_t kv_dim = KV_dim / HKV;
    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, k, KV_dim, D);
    release_weight(&l);
    if (st != VG_OK) return st;

    if (g->cfg.has_biases) {
        snprintf(name, sizeof(name), "blk.%u.attn_k.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < KV_dim; ++i) k[i] += bias[i];
            release_weight(&l);
        }
    }

    snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", layer);
    int has_k_norm = acquire_weight(g, name, &l) == VG_OK;
    const float *k_norm = has_k_norm ? (const float *)l.data : NULL;
    if (has_k_norm) for (uint32_t h = 0; h < HKV; ++h) {
        float *hk = k + h * kv_dim;
        float ss = 0.0f;
        for (uint32_t i = 0; i < kv_dim; ++i) ss += hk[i] * hk[i];
        const float inv = 1.0f / sqrtf(ss / (float)kv_dim + 1e-6f);
        for (uint32_t i = 0; i < kv_dim; ++i) hk[i] = hk[i] * inv * k_norm[i];
    }
    vg_cpu_rope_apply_all(k, HKV, kv_dim, q_rot, g->rope_cos, g->rope_sin);
    if (has_k_norm) release_weight(&l);

    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", layer);
    st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, v, KV_dim, D);
    release_weight(&l);
    if (st != VG_OK) return st;
    if (g->cfg.has_biases) {
        snprintf(name, sizeof(name), "blk.%u.attn_v.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < KV_dim; ++i) v[i] += bias[i];
            release_weight(&l);
        }
    }

    /* --- KV storage (paged, optional sliding-window eviction) --- */
    const uint32_t page_tokens = g->kv_page_tokens;
    if ((uint32_t)pos >= g->cfg.n_ctx) return VG_E_RANGE;
    const uint32_t page_idx = (uint32_t)pos / page_tokens;
    if (page_idx >= g->kv_capacity) return VG_E_RANGE;
    const uint32_t lp = layer * g->kv_capacity + page_idx;
    if (!g->kv_k[lp]) {
        g->kv_k[lp] = (float *)calloc((size_t)page_tokens * KV_dim, sizeof(float));
        g->kv_v[lp] = (float *)calloc((size_t)page_tokens * KV_dim, sizeof(float));
        if (!g->kv_k[lp] || !g->kv_v[lp]) return VG_E_NOMEM;
        g->kv_dim[layer] = KV_dim;
    }
    const uint32_t slot = ((uint32_t)pos % page_tokens) * KV_dim;
    memcpy(g->kv_k[lp] + slot, k, KV_dim * sizeof(float));
    memcpy(g->kv_v[lp] + slot, v, KV_dim * sizeof(float));

    /* --- Attention over all past positions --- */
    const uint32_t attn_dim = H * head_dim;
    if (attn_dim > g->work_att_out_cap) {
        float *tmp = (float *)realloc(g->work_att_out, (attn_dim + 1) * sizeof(float));
        if (!tmp) return VG_E_NOMEM;
        g->work_att_out = tmp; g->work_att_out_cap = attn_dim + 1;
    }
    memset(g->work_att_out, 0, attn_dim * sizeof(float));

    const uint32_t group = HKV ? H / HKV : 1;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const uint32_t window = g->cfg.sliding_window;
    const uint32_t n_past = (uint32_t)pos + 1;
    const uint32_t t_begin = (window && n_past > window) ? n_past - window : 0;

    /* Reused scratch: one buffer per graph instead of one allocation per layer per token. */
    if (n_past > g->work_scores_cap) return VG_E_RANGE;
    float *scores = g->work_scores;
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t kv_h = group ? h / group : 0;
        const float *hq = q + h * head_dim;
        uint32_t n = 0;
        for (uint32_t t = t_begin; t < n_past; ++t) {
            const uint32_t pi = t / page_tokens;
            const uint32_t pg = layer * g->kv_capacity + pi;
            const float *hk = g->kv_k[pg];
            if (!hk) { scores[n++] = -INFINITY; continue; }
            const float *row = hk + ((t % page_tokens) * KV_dim + kv_h * kv_dim);
            scores[n++] = vg_cpu_dot(hq, row, head_dim) * scale;
        }
        if (n == 0) continue;
        float mx = scores[0];
        for (uint32_t i = 1; i < n; ++i) if (scores[i] > mx) mx = scores[i];
        if (!isfinite(mx)) continue;
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; ++i) { scores[i] = expf(scores[i] - mx); sum += scores[i]; }
        if (sum > 0.0f) { const float inv = 1.0f / sum; for (uint32_t i = 0; i < n; ++i) scores[i] *= inv; }
        float *out = g->work_att_out + h * head_dim;
        uint32_t idx = 0;
        for (uint32_t t = t_begin; t < n_past; ++t, ++idx) {
            const uint32_t pi = t / page_tokens;
            const uint32_t pg = layer * g->kv_capacity + pi;
            const float *hv = g->kv_v[pg];
            if (!hv) continue;
            const float *row = hv + ((t % page_tokens) * KV_dim + kv_h * kv_dim);
            const float w = scores[idx];
            for (uint32_t d = 0; d < head_dim; ++d) out[d] += w * row[d];
        }
    }

    /* BitNet b1.58: a per-layer RMSNorm on the attention context, applied BEFORE the output
     * projection (llama.cpp: build_norm(ctx) then build_lora_mm(wo)). */
    if (g->cfg.has_sub_norm) {
        snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            /* Only apply when the weight really covers the context width; a mismatch here
             * would read the norm weight out of bounds instead of failing. */
            if (l.tensor->n_dims == 1 && l.tensor->dims[0] == attn_dim)
                vg_cpu_rmsnorm(g->work_att_out, g->work_att_out, (const float *)l.data, attn_dim, g->cfg.rms_eps);
            else
                fprintf(stderr, "[vg] %s has %llu weights, expected %u; sub-norm skipped\n",
                        name, (unsigned long long)(l.tensor->n_dims ? l.tensor->dims[0] : 0), attn_dim);
            release_weight(&l);
        }
    }

    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    float *po = g->work_proj;
    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_att_out, po, D, attn_dim);
    release_weight(&l);
    if (st != VG_OK) return st;
    for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + po[i];
    return VG_OK;
}

/* --- MoE routing ---------------------------------------------------------------- */
VG_Status vg_model_graph_moe_route(VG_ModelGraph *graph, uint32_t layer_idx,
                                   const float *hidden, float *gate_logits,
                                   VG_MoEState *state) {
    if (!graph || !state) return VG_E_INVALID;
    const uint32_t n_expert = graph->cfg.n_expert > 0 ? graph->cfg.n_expert : 1;
    uint32_t n_used = graph->cfg.n_expert_used > 0 ? graph->cfg.n_expert_used : 1;
    if (n_used > n_expert) n_used = n_expert;
    state->n_expert = n_expert;
    state->n_expert_used = n_used;
    if (!state->expert_ids) {
        state->expert_ids = (uint32_t *)malloc(n_used * sizeof(uint32_t));
        if (!state->expert_ids) return VG_E_NOMEM;
    }
    if (n_used == 0) return VG_OK;

    /* Compute router logits when the caller did not supply them. */
    float *logits = gate_logits;
    float *owned = NULL;
    if (!logits && hidden) {
        char name[160];
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer_idx);
        VG_TensorLease l;
        VG_Status st = acquire_weight(graph, name, &l);
        if (st != VG_OK) return st;
        owned = (float *)malloc(n_expert * sizeof(float));
        if (!owned) { release_weight(&l); return VG_E_NOMEM; }
        st = matmul(graph, l.data, l.size, l.tensor->ggml_type, hidden, owned, n_expert, graph->cfg.n_embd);
        release_weight(&l);
        if (st != VG_OK) { free(owned); return st; }
        logits = owned;
    }
    if (!logits) {
        /* No router input available: deterministic round-robin keeps the graph runnable. */
        for (uint32_t i = 0; i < n_used; ++i) state->expert_ids[i] = i % n_expert;
        free(owned);
        return VG_OK;
    }

    /* Stable top-k by logit (ties resolved by lower expert id). */
    for (uint32_t i = 0; i < n_used; ++i) {
        uint32_t best = UINT32_MAX;
        float best_val = -FLT_MAX;
        for (uint32_t e = 0; e < n_expert; ++e) {
            int already = 0;
            for (uint32_t j = 0; j < i; ++j) if (state->expert_ids[j] == e) { already = 1; break; }
            if (already) continue;
            if (logits[e] > best_val) { best_val = logits[e]; best = e; }
        }
        if (best == UINT32_MAX) { state->expert_ids[i] = i % n_expert; continue; }
        state->expert_ids[i] = best;
    }
    free(owned);
    return VG_OK;
}

void vg_moe_state_free(VG_MoEState *state) {
    if (!state) return;
    free(state->expert_ids);
    state->expert_ids = NULL;
    state->n_expert = 0;
    state->n_expert_used = 0;
}

/* Softmax over the selected experts' logits, returning normalised routing weights. */
static void route_weights(const float *logits, const uint32_t *ids, uint32_t n_used, float *weights) {
    float mx = -FLT_MAX;
    for (uint32_t i = 0; i < n_used; ++i) if (logits[ids[i]] > mx) mx = logits[ids[i]];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_used; ++i) { weights[i] = expf(logits[ids[i]] - mx); sum += weights[i]; }
    if (sum > 0.0f) for (uint32_t i = 0; i < n_used; ++i) weights[i] /= sum;
}

static VG_Status moe_ffn_layer(VG_ModelGraph *g, uint32_t layer, float *hidden) {
    const uint32_t D = g->cfg.n_embd;
    const uint32_t FF = g->cfg.n_ff;
    char name[160];
    VG_TensorLease l;

    memcpy(g->work_res, hidden, D * sizeof(float));
    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    /* Router logits from the block input (llama.cpp convention: post-norm input). */
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
    VG_Status st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    st = matmul(g, l.data, l.size, l.tensor->ggml_type, g->work_embd, g->route_logits, g->cfg.n_expert, D);
    release_weight(&l);
    if (st != VG_OK) return st;

    /* Top-k ids live on the stack: the per-layer malloc/free pair was pure overhead. */
    if (g->cfg.n_expert_used > 16) return VG_E_UNSUPPORTED;
    uint32_t expert_ids[16];
    VG_MoEState state;
    memset(&state, 0, sizeof(state));
    state.expert_ids = expert_ids;
    st = vg_model_graph_moe_route(g, layer, NULL, g->route_logits, &state);
    if (st != VG_OK) return st;

    float weights[16];
    if (state.n_expert_used > 16) return VG_E_UNSUPPORTED;
    route_weights(g->route_logits, state.expert_ids, state.n_expert_used, weights);

    /* The block input is preserved in work_res; the caller buffer becomes the weighted sum. */
    memset(hidden, 0, D * sizeof(float));
    for (uint32_t i = 0; i < state.n_expert_used; ++i) {
        const uint32_t e = state.expert_ids[i];
        const void *wgate = NULL, *wup = NULL, *wdown = NULL;
        size_t bgate = 0, bup = 0, bdown = 0;
        uint32_t tgate = 0, tup = 0, tdown = 0;
        st = acquire_expert(g, layer, VG_EXPERT_GATE, e, FF, D, &wgate, &bgate, &tgate);
        if (st != VG_OK) return st;
        st = acquire_expert(g, layer, VG_EXPERT_UP, e, FF, D, &wup, &bup, &tup);
        if (st != VG_OK) return st;
        st = matmul(g, wgate, bgate, tgate, g->work_embd, g->expert_gate, FF, D);
        if (st != VG_OK) return st;
        st = matmul(g, wup, bup, tup, g->work_embd, g->expert_up, FF, D);
        if (st != VG_OK) return st;
        vg_cpu_swiglu(g->expert_gate, g->expert_gate, g->expert_up, FF);
        st = acquire_expert(g, layer, VG_EXPERT_DOWN, e, D, FF, &wdown, &bdown, &tdown);
        if (st != VG_OK) return st;
        st = matmul(g, wdown, bdown, tdown, g->expert_gate, g->expert_down, D, FF);
        if (st != VG_OK) return st;
        const float w = weights[i];
        for (uint32_t j = 0; j < D; ++j) hidden[j] += w * g->expert_down[j];
    }
    for (uint32_t j = 0; j < D; ++j) hidden[j] += g->work_res[j];
    return VG_OK;
}

/* --- Dense feed-forward --------------------------------------------------------- */
static VG_Status ffn_layer(VG_ModelGraph *g, uint32_t layer, float *hidden) {
    if (g->cfg.n_expert > 1) return moe_ffn_layer(g, layer, hidden);

    const uint32_t D = g->cfg.n_embd;
    const uint32_t FF = g->cfg.n_ff;
    char name[160];
    VG_TensorLease l;

    memcpy(g->work_res, hidden, D * sizeof(float));
    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    /* Graph-owned scratch: no allocation in the per-token FFN path. */
    float *down_out = g->work_proj;

    if (g->cfg.has_gate_up) {
        float *gate_out = g->work_gate;
        float *up_out = g->work_up;
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", layer);
        VG_Status st = acquire_weight(g, name, &l);
        if (st != VG_OK) return st;
        st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, gate_out, FF, D);
        release_weight(&l);
        if (st != VG_OK) return st;
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
        st = acquire_weight(g, name, &l);
        if (st != VG_OK) return st;
        st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, up_out, FF, D);
        release_weight(&l);
        if (st != VG_OK) return st;
        vg_cpu_swiglu(g->work_ffn, gate_out, up_out, FF);
    } else {
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
        VG_Status st = acquire_weight(g, name, &l);
        if (st != VG_OK) return st;
        st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, g->work_ffn, FF, D);
        release_weight(&l);
        if (st != VG_OK) return st;
        vg_cpu_gelu(g->work_ffn, FF);
    }

    /* BitNet b1.58: RMSNorm over the SwiGLU output (n_ff) before ffn_down. */
    if (g->cfg.has_sub_norm) {
        snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            if (l.tensor->n_dims == 1 && l.tensor->dims[0] == FF)
                vg_cpu_rmsnorm(g->work_ffn, g->work_ffn, (const float *)l.data, FF, g->cfg.rms_eps);
            else
                fprintf(stderr, "[vg] %s has %llu weights, expected %u; sub-norm skipped\n",
                        name, (unsigned long long)(l.tensor->n_dims ? l.tensor->dims[0] : 0), FF);
            release_weight(&l);
        }
    }

    snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", layer);
    VG_Status st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_ffn, down_out, D, FF);
    release_weight(&l);
    if (st != VG_OK) return st;
    for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + down_out[i];
    return VG_OK;
}

/* --- Output projection ---------------------------------------------------------- */
static VG_Status logits_from_head(VG_ModelGraph *g, const char *name) {
    VG_TensorLease l;
    VG_Status st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    const uint32_t D = g->cfg.n_embd;
    const VG_QuantInfo *qi = vg_quant_info(l.tensor->ggml_type);
    if (!qi || !qi->supported_scalar) { release_weight(&l); return VG_E_UNSUPPORTED; }
    const uint32_t rows = (uint32_t)(l.tensor->dims[1] < g->cfg.n_vocab ? l.tensor->dims[1] : g->cfg.n_vocab);
    st = matmul_named(g, name, l.data, l.size, l.tensor->ggml_type, g->work_embd, g->logits, rows, D);
    if (st == VG_OK && rows < g->cfg.n_vocab) {
        for (uint32_t i = rows; i < g->cfg.n_vocab; ++i) g->logits[i] = -FLT_MAX;
    }
    release_weight(&l);
    return st;
}

/* --- Decode --------------------------------------------------------------------- */
VG_Status vg_model_graph_decode(VG_ModelGraph *g, int32_t input_token, int32_t kv_pos, float *logits_out) {
    if (!g) return VG_E_INVALID;
    if (kv_pos < 0 || (uint32_t)kv_pos >= g->cfg.n_ctx) return VG_E_RANGE;
    const uint32_t D = g->cfg.n_embd;

    VG_Status st = vg_model_graph_embed(g, input_token, g->work_embd);
    if (st != VG_OK) return st;

    for (uint32_t l = 0; l < g->cfg.n_layer; ++l) {
        /* No explicit prefetch: weights are leased from the read-only mapping, so the kernel
         * already streams them, and copying them into the host cache would double the IO. */
        st = attention_layer(g, l, g->work_embd, kv_pos);
        if (st != VG_OK) return st;
        st = ffn_layer(g, l, g->work_embd);
        if (st != VG_OK) return st;
    }

    VG_TensorLease wl;
    if (acquire_weight(g, "output_norm.weight", &wl) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, g->work_embd, (const float *)wl.data, D, g->cfg.rms_eps);
        release_weight(&wl);
    }

    if (find_tensor(g, "output.weight")) {
        st = logits_from_head(g, "output.weight");
    } else {
        st = logits_from_head(g, "token_embd.weight");
    }
    if (st != VG_OK) return st;

    if (logits_out) memcpy(logits_out, g->logits, (size_t)g->cfg.n_vocab * sizeof(float));
    return VG_OK;
}

#ifdef VG_HAS_VULKAN
void vg_model_graph_set_vulkan(VG_ModelGraph *g, VG_VK *vk) {
    if (!g) return;
    g->vk = vk;
    if (!g->vk_slots) {
        const uint32_t cap = g->cfg.n_expert > 1 ? 16u : 32u;
        g->vk_slots = (VG_VKWeightSlot *)calloc(cap, sizeof(VG_VKWeightSlot));
        g->vk_slots_cap = g->vk_slots ? cap : 0;
        g->vk_slots_used = 0;
        g->vk_clock = 0;
    }
}
static void vk_cache_free(VG_ModelGraph *g) {
    if (!g->vk_slots) return;
    for (uint32_t i = 0; i < g->vk_slots_used; ++i) vg_vk_weight_release(g->vk, g->vk_slots[i].weight);
    free(g->vk_slots);
    g->vk_slots = NULL;
    g->vk_slots_cap = g->vk_slots_used = 0;
}
#endif
