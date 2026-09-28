#include "vg/model_graph.h"
#include "vg/tensor_source.h"
#include "vg/trace.h"
#include "vg/quant.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

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
    float *logits;
    float *rope_freqs;  /* Optional: precomputed RoPE frequencies */
    uint32_t n_rope_freqs;
    float **kv_k;
    float **kv_v;
    uint32_t *kv_dim;   /* actual per-layer KV dim from tensor shapes */
    uint32_t kv_capacity;
    uint32_t kv_used;
#ifdef VG_HAS_VULKAN
    VG_VK *vk;
    VG_VKBuffer *hidden_dev;   /* persistence for the fused dense path */
    VG_VKDenseLayer *gpu_layers;
    uint32_t gpu_layers_cap;
    int gpu_dense;
    int gpu_dense_checked;
#endif
};

static const VG_GGUF_Tensor *find_tensor(const VG_ModelGraph *g, const char *name) {
    return vg_gguf_find_tensor(g->file, name);
}

static VG_Status acquire_weight(VG_ModelGraph *g, const char *name, VG_TensorLease *lease);

#ifdef VG_HAS_VULKAN
static VG_Status vk_matmul(VG_ModelGraph *g, uint32_t ggml_type, const void *wdata, size_t wbytes,
                             const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
static VG_Status vk_matmul_slice(VG_ModelGraph *g, uint32_t ggml_type, const void *key,
                                 uint32_t tag, const void *wdata, size_t wbytes,
                                 const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
static VG_Status vk_q8_prepare(VG_ModelGraph *g, const void *key, uint32_t tag,
                               const void *wdata, uint32_t out_dim, uint32_t in_dim,
                               VG_VKBuffer **wb, VG_VKBuffer **sb);
static int dense_gpu_layer_forward(VG_ModelGraph *g, uint32_t layer, uint32_t pos,
                                   int first, int last, float *hidden, VG_VKDenseLayer *out);
#endif

#ifdef VG_HAS_VULKAN
static int vg_layer_tail_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("VG_LAYER_TAIL"); v = (e && e[0] == '0') ? 0 : 1; } return v; }
/* True when the MoE attention-output + FFN head can be fused (COMPASS 3.1):
 * the fused head folds the separate o_proj submit and router submit into one
 * command buffer. Both attention_layer (skips its o_proj) and ffn_layer (runs
 * the fused head) consult this so they stay consistent. */
static int moe_fused_head_ok(const VG_ModelGraph *g) { return g->vk && vg_layer_tail_on() && find_tensor(g, "blk.0.ffn_gate_inp.weight") != NULL; }
/* Strict eligibility for the fused single-submit dense layer (VG_GPU_ATTN,
 * default on). Requires the shapes and quant types the device path implements;
 * anything else falls back to the per-op CPU/GPU path. */
static int gpu_dense_ok(VG_ModelGraph *g) {
    if (!g->vk) return 0;
    const char *e = getenv("VG_GPU_ATTN"); if (e && e[0] == '0') return 0;
    if (!g->cfg.has_gate_up || g->cfg.has_biases) return 0;
    if (find_tensor(g, "blk.0.attn_q_norm.weight") || find_tensor(g, "blk.0.attn_k_norm.weight")) return 0;
    if (!find_tensor(g, "blk.0.ffn_gate.weight")) return 0;
    if (!find_tensor(g, "blk.0.attn_norm.weight") || !find_tensor(g, "blk.0.ffn_norm.weight")) return 0;
    if (g->cfg.n_head * g->cfg.head_dim != g->cfg.n_embd) return 0;
    uint32_t HKV = g->cfg.n_head_kv ? g->cfg.n_head_kv : 1;
    if (!HKV || (g->cfg.n_head % HKV)) return 0;
    if ((g->cfg.n_embd % 32u) || (g->cfg.n_ff % 32u) || ((g->cfg.n_head * g->cfg.head_dim) % 32u)) return 0;
    static const char *wn[] = {"blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight",
                               "blk.0.attn_output.weight","blk.0.ffn_gate.weight","blk.0.ffn_up.weight","blk.0.ffn_down.weight"};
    for (unsigned i = 0; i < sizeof(wn)/sizeof(wn[0]); ++i) { const VG_GGUF_Tensor *t = find_tensor(g, wn[i]); if (!t || t->ggml_type != 8) return 0; }
    return 1;
}
#else
static int vg_layer_tail_on(void) { return 0; }
static int moe_fused_head_ok(const VG_ModelGraph *g) { (void)g; return 0; }
#endif
static VG_Status acquire_weight(VG_ModelGraph *g, const char *name, VG_TensorLease *lease) {
    if (g->source) return vg_tensor_acquire(g->source, name, VG_TIER_STORAGE, lease);
    const VG_GGUF_Tensor *t = find_tensor(g, name); if (!t) return VG_E_INVALID;
    void *data = malloc(t->nbytes ? t->nbytes : 1); if (!data) return VG_E_NOMEM;
    VG_Status st = vg_gguf_read(g->file, t, 0, data, t->nbytes);
    if (st != VG_OK) { free(data); return st; }
    lease->tensor = t; lease->data = data; lease->size = t->nbytes; lease->tier = VG_TIER_HOST;
    lease->source = NULL; lease->ticket = 0;
    return VG_OK;
}

static void release_weight(VG_TensorLease *lease) {
    if (lease->source) vg_tensor_release(lease);
    else if (lease->data) free((void *)lease->data);
}

static VG_Status read_metadata_int(const VG_ModelGraph *g, const char *key, int32_t *out) {
    const char *v = vg_gguf_meta(g->file, key);
    if (!v) return VG_E_INVALID;
    const char *p = v;
    while (*p == '[' || *p == ' ' || *p == ',') ++p;
    *out = (int32_t)strtol(p, NULL, 10);
    return VG_OK;
}

static VG_Status read_metadata_float(const VG_ModelGraph *g, const char *key, float *out) {
    const char *v = vg_gguf_meta(g->file, key);
    if (!v) return VG_E_INVALID;
    const char *p = v;
    while (*p == '[' || *p == ' ' || *p == ',') ++p;
    float val = strtof(p, NULL);
    if (!isfinite(val)) return VG_E_INVALID;
    *out = val;
    return VG_OK;
}

VG_Status vg_model_graph_load(VG_GGUF *file, VG_TensorSource *source, VG_ModelGraph **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_ModelGraph *g = (VG_ModelGraph *)calloc(1, sizeof(*g));
    if (!g) return VG_E_NOMEM;
    g->file = file; g->source = source;

    const char *arch = vg_gguf_meta(file, "general.architecture");
    if (!arch) { free(g); return VG_E_FORMAT; }

    char key[128];
    int32_t i32; float f32;

    #define READ_INT(field, name) do { i32 = 0; snprintf(key, sizeof(key), "%s." #name, arch); read_metadata_int(g, key, &i32); g->cfg.field = (uint32_t)i32; } while(0)
    #define READ_FLOAT(field, name) do { f32 = 0; snprintf(key, sizeof(key), "%s." #name, arch); if (read_metadata_float(g, key, &f32) == VG_OK) g->cfg.field = f32; } while(0)

    READ_INT(n_vocab, vocab_size);
    READ_INT(n_embd, embedding_length);
    READ_INT(n_head, attention.head_count);
    READ_INT(n_head_kv, attention.head_count_kv);
    READ_INT(n_layer, block_count);
    READ_INT(n_ff, feed_forward_length);
    READ_INT(n_rot, rope.dimension_count);

    if (g->cfg.n_vocab == 0) {
        const VG_GGUF_Tensor *te = find_tensor(g, "token_embd.weight");
        if (te && te->n_dims >= 2) g->cfg.n_vocab = (uint32_t)te->dims[1];
    }
    if (g->cfg.n_embd == 0) {
        const VG_GGUF_Tensor *te = find_tensor(g, "token_embd.weight");
        if (te && te->n_dims >= 1) g->cfg.n_embd = (uint32_t)te->dims[0];
    }
    if (g->cfg.n_ff == 0) {
        const VG_GGUF_Tensor *fd = find_tensor(g, "blk.0.ffn_down.weight");
        if (fd && fd->n_dims >= 1) g->cfg.n_ff = (uint32_t)fd->dims[0];
    }

    if (g->cfg.n_head_kv == 0) g->cfg.n_head_kv = g->cfg.n_head;
    if (g->cfg.head_dim == 0) {
        /* Try to derive from attn_q.weight dims: dims[1] = n_head * head_dim (out dim) */
        const VG_GGUF_Tensor *q = find_tensor(g, "blk.0.attn_q.weight");
        if (q && q->n_dims >= 2 && g->cfg.n_head) {
            g->cfg.head_dim = (uint32_t)(q->dims[1] / g->cfg.n_head);
        }
        if (g->cfg.head_dim == 0) g->cfg.head_dim = g->cfg.n_embd / (g->cfg.n_head ? g->cfg.n_head : 1);
    }
    /* n_rot from metadata may be total (key_length); clamp to head_dim */
    if (g->cfg.n_rot == 0) g->cfg.n_rot = g->cfg.head_dim;
    if (g->cfg.n_rot > g->cfg.head_dim) g->cfg.n_rot = g->cfg.head_dim;
    READ_FLOAT(rms_eps, attention.layer_norm_rms_epsilon);
    if (!isfinite(g->cfg.rms_eps) || g->cfg.rms_eps <= 0.0f) g->cfg.rms_eps = 1e-5f;
    READ_FLOAT(freq_base, rope.freq_base);
    /* freq_base of 1e-23 or non-finite indicates corrupt/missing metadata; use gemma4 default */
    if (!isfinite(g->cfg.freq_base) || g->cfg.freq_base < 1.0f) g->cfg.freq_base = 100000.0f;

    /* Load optional rope_freqs.weight tensor (precomputed RoPE frequencies) */
    {
        const VG_GGUF_Tensor *rf = find_tensor(g, "rope_freqs.weight");
        if (rf && rf->n_dims >= 1 && rf->ggml_type == 0) {
            g->n_rope_freqs = (uint32_t)rf->dims[0];
            g->rope_freqs = (float *)malloc(g->n_rope_freqs * sizeof(float));
            if (g->rope_freqs) {
                VG_Status st = vg_gguf_read(g->file, rf, 0, g->rope_freqs, g->n_rope_freqs * sizeof(float));
                if (st != VG_OK) {
                    free(g->rope_freqs); g->rope_freqs = NULL; g->n_rope_freqs = 0;
                } else {
                    /* Fall back to theta formula if rope_freqs have invalid/unpopulated values */
                    int valid = 1;
                    for (uint32_t i = 0; i < g->n_rope_freqs; ++i) {
                        if (!isfinite(g->rope_freqs[i]) || g->rope_freqs[i] <= 0.0f || g->rope_freqs[i] > 1.0e6f) {
                            valid = 0; break;
                        }
                    }
                    if (!valid) {
                        fprintf(stderr, "[vg] warning: rope_freqs metadata produced invalid values, using theta formula\n");
                        free(g->rope_freqs); g->rope_freqs = NULL; g->n_rope_freqs = 0;
                    }
                }
            }
        }
    }


    READ_INT(n_expert, expert_count);
    READ_INT(n_expert_used, expert_used_count);
    if (g->cfg.n_expert_used == 0 && g->cfg.n_expert > 0) g->cfg.n_expert_used = 1;
    if (g->cfg.n_expert == 0) g->cfg.n_expert = 1;

     {
        int32_t n_ctx = 0;
        snprintf(key, sizeof(key), "%s.context_length", arch);
        read_metadata_int(g, key, &n_ctx);
        g->cfg.n_ctx = (n_ctx > 0) ? (uint32_t)n_ctx : 2048;
    }

    {
        int32_t sw = 0;
        read_metadata_int(g, "gemma4.attention.sliding_window", &sw);
        if (sw <= 0) read_metadata_int(g, "llama.attention.sliding_window", &sw);
        if (sw <= 0) read_metadata_int(g, "attention.sliding_window", &sw);
        g->cfg.sliding_window = (sw > 0) ? (uint32_t)sw : 512;
    }

    fprintf(stderr, "[vg] %s: n_vocab=%u n_embd=%u n_head=%u n_head_kv=%u n_layer=%u n_ff=%u head_dim=%u n_rot=%u n_ctx=%u rms_eps=%.2e freq_base=%.1f\n",
            arch, g->cfg.n_vocab, g->cfg.n_embd, g->cfg.n_head, g->cfg.n_head_kv,
            g->cfg.n_layer, g->cfg.n_ff, g->cfg.head_dim, g->cfg.n_rot, g->cfg.n_ctx, g->cfg.rms_eps, g->cfg.freq_base);

    snprintf(key, sizeof(key), "%s.bos_token_id", "tokenizer.ggml");
    read_metadata_int(g, key, &g->cfg.bos_id);
    snprintf(key, sizeof(key), "%s.eos_token_id", "tokenizer.ggml");
    read_metadata_int(g, key, &g->cfg.eos_id);

    {
        char qkv_bias[128]; snprintf(qkv_bias, sizeof(qkv_bias), "blk.0.attn_q.bias");
        g->cfg.has_biases = (find_tensor(g, qkv_bias) != NULL);
        char gate[128]; snprintf(gate, sizeof(gate), "blk.0.ffn_gate.weight");
        g->cfg.has_gate_up = (find_tensor(g, gate) != NULL);
    }

    size_t embd_bytes = g->cfg.n_embd * sizeof(float);
    size_t ff_bytes = g->cfg.n_ff * sizeof(float);
    size_t logits_bytes = g->cfg.n_vocab * sizeof(float);
    uint32_t kv_heads = g->cfg.n_head_kv ? g->cfg.n_head_kv : 1;
    uint32_t h_dim = g->cfg.head_dim ? g->cfg.head_dim : 1;
    uint32_t q_total = g->cfg.n_head * h_dim + 2 * kv_heads * h_dim;
    size_t qkv_bytes = q_total * sizeof(float);
    g->work_embd = (float *)malloc(embd_bytes);
    g->work_ffn = (float *)malloc(ff_bytes);
    g->work_qkv = (float *)malloc(qkv_bytes);
    g->work_qkv_cap = q_total;
    g->work_att_out = (float *)malloc((g->cfg.n_head * h_dim + 1) * sizeof(float));
    g->work_att_out_cap = g->cfg.n_head * h_dim + 1;
    g->work_head = (float *)malloc(g->cfg.head_dim * sizeof(float));
    g->work_res = (float *)malloc(embd_bytes);
    g->logits = (float *)malloc(logits_bytes);
    if (!g->work_embd || !g->work_ffn || !g->work_qkv || !g->work_att_out || !g->work_head || !g->work_res || !g->logits) {
        free(g->work_embd); free(g->work_ffn); free(g->work_qkv);
        free(g->work_att_out); free(g->work_head); free(g->work_res);
        free(g->logits);
        free(g); return VG_E_NOMEM;
    }

    {
        uint32_t kv_page = 64;
        uint32_t n_pages = (g->cfg.n_ctx + kv_page - 1) / kv_page;
        g->kv_k = (float **)calloc(g->cfg.n_layer * n_pages, sizeof(float *));
        g->kv_v = (float **)calloc(g->cfg.n_layer * n_pages, sizeof(float *));
        g->kv_dim = (uint32_t *)calloc(g->cfg.n_layer, sizeof(uint32_t));
        if (!g->kv_k || !g->kv_v || !g->kv_dim) {
            free(g->kv_k); free(g->kv_v); free(g->kv_dim);
            free(g->work_embd); free(g->work_ffn); free(g->work_qkv);
            free(g->work_att_out); free(g->work_head); free(g->work_res);
            free(g->logits);
            free(g); return VG_E_NOMEM;
        }
    g->kv_capacity = n_pages;
    g->kv_used = 0;
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
    free(g->logits);
    free(g->rope_freqs);
    if (g->kv_k) {
        uint32_t total_pages = g->cfg.n_layer * g->kv_capacity;
        for (uint32_t i = 0; i < total_pages; ++i) {
            free(g->kv_k[i]); free(g->kv_v[i]);
        }
        free(g->kv_k); free(g->kv_v);
    }
    free(g->kv_dim);
#ifdef VG_HAS_VULKAN
    free(g->gpu_layers);
#endif
    free(g);
}

void vg_model_graph_reset(VG_ModelGraph *g) {
    if (!g) return;
    uint32_t page_tokens = 64;
    for (uint32_t i = 0; i < g->cfg.n_layer * g->kv_capacity; ++i) {
        if (g->kv_k[i]) {
            uint32_t d = g->kv_dim[i / g->kv_capacity];
            memset(g->kv_k[i], 0, page_tokens * d * sizeof(float));
            memset(g->kv_v[i], 0, page_tokens * d * sizeof(float));
        }
    }
    g->kv_used = 0;
}

const VG_ModelConfig *vg_model_graph_config(const VG_ModelGraph *g) { return g ? &g->cfg : NULL; }

VG_Status vg_model_graph_embed(VG_ModelGraph *g, int32_t token, float *embd_out) {
    if (!g) return VG_E_INVALID;
    /* Reject token ids outside the vocabulary before touching the embedding
     * tensor. Qwen3 ships no bos_token_id, so a caller that prepends BOS blindly
     * would otherwise read the embedding table at a negative offset. */
    if (token < 0 || (uint32_t)token >= g->cfg.n_vocab) return VG_E_RANGE;
    if (!g || token < 0 || (uint32_t)token >= g->cfg.n_vocab) return VG_E_INVALID;
    const VG_GGUF_Tensor *t = find_tensor(g, "token_embd.weight");
    if (!t) return VG_E_INVALID;
    uint32_t embd = g->cfg.n_embd;
    uint32_t ggml_type = t->ggml_type;
    if (ggml_type == 0) {
        uint32_t row_bytes = embd * sizeof(float);
        uint64_t offset = (uint64_t)token * row_bytes;
        void *row = malloc(row_bytes);
        if (!row) return VG_E_NOMEM;
        VG_Status st = vg_gguf_read(g->file, t, offset, row, row_bytes);
        if (st != VG_OK) { free(row); return st; }
        memcpy(embd_out, row, row_bytes);
        free(row);
    } else {
        const VG_QuantInfo *qi = vg_quant_info(ggml_type);
        if (!qi || qi->block_size == 0) return VG_E_UNSUPPORTED;
        uint32_t blocks_per_row = embd / qi->block_size;
        uint64_t row_bytes = (uint64_t)blocks_per_row * qi->bytes_per_block;
        uint64_t offset = (uint64_t)token * row_bytes;
        void *row = malloc((size_t)row_bytes);
        if (!row) return VG_E_NOMEM;
        VG_Status st = vg_gguf_read(g->file, t, offset, row, (size_t)row_bytes);
        if (st != VG_OK) { free(row); return st; }
        st = vg_dequantize_row(ggml_type, row, (size_t)row_bytes, embd_out, embd);
        free(row);
        if (st != VG_OK) return st;
    }
    return VG_OK;
}

static float vec_rms(const float *x, uint32_t n, float eps) {
    float ss = 0; for (uint32_t i = 0; i < n; ++i) ss += x[i]*x[i];
    return 1.0f / sqrtf(ss / (float)n + eps);
}

static void attention_layer(VG_ModelGraph *g, uint32_t layer, float *hidden, float *cur_pos_f) {
    uint32_t D = g->cfg.n_embd;
    uint32_t H = g->cfg.n_head;
    uint32_t HKV = g->cfg.n_head_kv ? g->cfg.n_head_kv : 1;
    uint32_t pos = (uint32_t)*cur_pos_f;
    char name[128];
    VG_TensorLease l;
    VG_TensorLease nl;

    memcpy(g->work_res, hidden, D * sizeof(float));

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }
    if (layer == 0 && pos <= 1) {
        fprintf(stderr, "[vg] DUMP an0");
        for (uint32_t di = 0; di < D; ++di) fprintf(stderr, " %.7g", g->work_embd[di]);
        fprintf(stderr, "\n");
    }

    uint32_t Q_dim = H * g->cfg.head_dim;
    uint32_t HD = g->cfg.head_dim;
    uint32_t KV_dim = HKV * HD;
    uint32_t n_rot = g->cfg.n_rot ? (g->cfg.n_rot < HD ? g->cfg.n_rot : HD) : HD;
    float *q_proj = NULL, *k_full = NULL, *v_full = NULL;

    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
    if (acquire_weight(g, name, &l) != VG_OK) {
        memcpy(hidden, g->work_res, D * sizeof(float));
        return;
    }
    if (l.tensor->n_dims >= 2 && l.tensor->dims[1] >= 1) Q_dim = (uint32_t)l.tensor->dims[1];
    HD = H ? (Q_dim / H) : g->cfg.head_dim;
    n_rot = g->cfg.n_rot ? (g->cfg.n_rot < HD ? g->cfg.n_rot : HD) : HD;

    uint32_t need_qkv = Q_dim + 2 * KV_dim;
    if (need_qkv > g->work_qkv_cap) {
        float *tmp = (float *)realloc(g->work_qkv, need_qkv * sizeof(float));
        if (tmp) { g->work_qkv = tmp; g->work_qkv_cap = need_qkv; }
    }
    /* COMPASS 3.1: batch q/k/v projection into ONE submit. */
    int qkv_gpu = 0;
#ifdef VG_HAS_VULKAN
    if (g->vk) {
        VG_TensorLease lk, lv; memset(&lk, 0, sizeof(lk)); memset(&lv, 0, sizeof(lv));
        char kn[128], vn[128];
        snprintf(kn, sizeof(kn), "blk.%u.attn_k.weight", layer);
        snprintf(vn, sizeof(vn), "blk.%u.attn_v.weight", layer);
        if (l.tensor->ggml_type == 8 && acquire_weight(g, kn, &lk) == VG_OK && acquire_weight(g, vn, &lv) == VG_OK &&
            lk.tensor->ggml_type == 8 && lv.tensor->ggml_type == 8) {
            uint32_t kd = (lk.tensor->n_dims >= 2 && lk.tensor->dims[1] >= 1) ? (uint32_t)lk.tensor->dims[1] : KV_dim;
            uint32_t vd = (lv.tensor->n_dims >= 2 && lv.tensor->dims[1] >= 1) ? (uint32_t)lv.tensor->dims[1] : KV_dim;
            if (kd == KV_dim && vd == KV_dim) {
                if (!q_proj) q_proj = (float *)malloc(Q_dim * sizeof(float));
                if (!k_full) k_full = (float *)malloc(KV_dim * sizeof(float));
                if (!v_full) v_full = (float *)malloc(KV_dim * sizeof(float));
                if (q_proj && k_full && v_full) {
                    VG_VKBuffer *wb = NULL, *sb = NULL;
                    VG_VKMatvecReq rq[3]; int ok = 1;
                    if (vk_q8_prepare(g, l.tensor, 0u, l.data, Q_dim, D, &wb, &sb) != VG_OK) ok = 0;
                    else { rq[0].weights = wb; rq[0].scales = sb; rq[0].x = g->work_embd; rq[0].y = q_proj; rq[0].in_dim = D; rq[0].out_dim = Q_dim; }
                    if (ok && vk_q8_prepare(g, lk.tensor, 0u, lk.data, KV_dim, D, &wb, &sb) != VG_OK) ok = 0;
                    else if (ok) { rq[1].weights = wb; rq[1].scales = sb; rq[1].x = g->work_embd; rq[1].y = k_full; rq[1].in_dim = D; rq[1].out_dim = KV_dim; }
                    if (ok && vk_q8_prepare(g, lv.tensor, 0u, lv.data, KV_dim, D, &wb, &sb) != VG_OK) ok = 0;
                    else if (ok) { rq[2].weights = wb; rq[2].scales = sb; rq[2].x = g->work_embd; rq[2].y = v_full; rq[2].in_dim = D; rq[2].out_dim = KV_dim; }
                    if (ok && vg_vk_matvec_q8_0_batch(g->vk, rq, 3) == VG_OK) qkv_gpu = 1;
                }
            }
        }
        release_weight(&lk); release_weight(&lv);
    }
#endif

    if (!q_proj) q_proj = (float *)malloc(Q_dim * sizeof(float));
    if (q_proj) {
        if (!qkv_gpu) {
#ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, q_proj, Q_dim, D) == VG_OK) { } else
#endif
        { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, q_proj, Q_dim, D); }
        }
        if (layer == 0 && pos <= 1) {
            fprintf(stderr, "[vg] DUMP q0");
            for (uint32_t di = 0; di < Q_dim; ++di) fprintf(stderr, " %.7g", q_proj[di]);
            fprintf(stderr, "\n");
        }

        for (uint32_t h = 0; h < H; ++h) {
            float *hq = q_proj + h * HD;
            char qn_name[128]; snprintf(qn_name, sizeof(qn_name), "blk.%u.attn_q_norm.weight", layer);
            if (acquire_weight(g, qn_name, &nl) == VG_OK) {
                const float *qn = (const float *)nl.data;
                float inv = vec_rms(hq, HD, 1e-6f);
                for (uint32_t i = 0; i < HD; ++i) hq[i] = hq[i] * inv * qn[i];
                release_weight(&nl);
            }
            uint32_t n_pairs = n_rot / 2;
            if (n_pairs > HD / 2) n_pairs = HD / 2;
            /* rope_freqs stores one scaled inverse frequency per rotary PAIR.
             * ggml GGML_ROPE_TYPE_NORMAL pairs ADJACENT elements (2k, 2k+1). */
            if (g->rope_freqs && n_pairs && g->n_rope_freqs >= n_pairs) {
                for (uint32_t i = 0; i < n_pairs; ++i) {
                    float inv = powf(g->cfg.freq_base, -2.0f * (float)i / (float)(n_rot ? n_rot : HD));
                    float freq = (float)pos * inv / g->rope_freqs[i];
                    float c = cosf(freq), s = sinf(freq);
                    float a = hq[2 * i], b = hq[2 * i + 1];
                    hq[2 * i] = a * c - b * s; hq[2 * i + 1] = a * s + b * c;
                }
            } else {
                vg_cpu_rope(hq, hq, HD, n_rot, 1.0f, g->cfg.freq_base, (int32_t)pos);
            }
        }
        memcpy(g->work_qkv, q_proj, Q_dim * sizeof(float));
        free(q_proj);
    }
    release_weight(&l);

    if (g->cfg.has_biases) {
        snprintf(name, sizeof(name), "blk.%u.attn_q.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < Q_dim; ++i) g->work_qkv[i] += bias[i];
            release_weight(&l);
        }
    }

    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        uint32_t kdim = (l.tensor->n_dims >= 2 && l.tensor->dims[1] >= 1) ? (uint32_t)l.tensor->dims[1] : KV_dim;
        if (kdim != KV_dim) { need_qkv = Q_dim + 2 * kdim; if (need_qkv > g->work_qkv_cap) { float *t=realloc(g->work_qkv,need_qkv*sizeof(float)); if(t){g->work_qkv=t;g->work_qkv_cap=need_qkv;} } KV_dim = kdim; }
        if (!k_full) k_full = (float *)malloc(KV_dim * sizeof(float));
        if (k_full) {
            if (!qkv_gpu) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, k_full, KV_dim, D) == VG_OK) { } else
#endif
            { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, k_full, KV_dim, D); }
            }
            if (layer == 0 && pos == 0) fprintf(stderr, "[vg] Kcur-0 = [%.4f, %.4f, %.4f]\n", k_full[0], k_full[1], k_full[2]);
            uint32_t k_hdim = HKV ? (KV_dim / HKV) : HD;
            for (uint32_t h = 0; h < HKV; ++h) {
                float *hk = k_full + h * k_hdim;
                char kn_name[128]; snprintf(kn_name, sizeof(kn_name), "blk.%u.attn_k_norm.weight", layer);
                if (acquire_weight(g, kn_name, &nl) == VG_OK) {
                    const float *kn = (const float *)nl.data;
                    float inv = vec_rms(hk, k_hdim, 1e-6f);
                    for (uint32_t i = 0; i < k_hdim; ++i) hk[i] = hk[i] * inv * kn[i];
                    release_weight(&nl);
                }
                uint32_t n_pairs = n_rot / 2;
                if (n_pairs > k_hdim / 2) n_pairs = k_hdim / 2;
                if (g->rope_freqs && n_pairs && g->n_rope_freqs >= n_pairs) {
                    for (uint32_t i = 0; i < n_pairs; ++i) {
                        float inv = powf(g->cfg.freq_base, -2.0f * (float)i / (float)(n_rot ? n_rot : k_hdim));
                        float freq = (float)pos * inv / g->rope_freqs[i];
                        float c = cosf(freq), s = sinf(freq);
                        float a = hk[2 * i], b = hk[2 * i + 1];
                        hk[2 * i] = a * c - b * s; hk[2 * i + 1] = a * s + b * c;
                    }
                } else {
                    vg_cpu_rope(hk, hk, k_hdim, n_rot, 1.0f, g->cfg.freq_base, (int32_t)pos);
                }
            }
        }
        release_weight(&l);
    }

    if (g->cfg.has_biases && k_full) {
        snprintf(name, sizeof(name), "blk.%u.attn_k.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < KV_dim; ++i) k_full[i] += bias[i];
            release_weight(&l);
        }
    }

    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        uint32_t vdim = (l.tensor->n_dims >= 2 && l.tensor->dims[1] >= 1) ? (uint32_t)l.tensor->dims[1] : KV_dim;
        if (vdim != KV_dim) { need_qkv = Q_dim + 2 * vdim; if (need_qkv > g->work_qkv_cap) { float *t=realloc(g->work_qkv,need_qkv*sizeof(float)); if(t){g->work_qkv=t;g->work_qkv_cap=need_qkv;} } KV_dim = vdim; }
        if (!v_full) v_full = (float *)malloc(KV_dim * sizeof(float));
        if (v_full) {
            if (!qkv_gpu) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, v_full, KV_dim, D) == VG_OK) { } else
#endif
            { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, v_full, KV_dim, D); }
            }
            if (layer == 0 && pos == 0) fprintf(stderr, "[vg] Vcur-0 = [%.4f, %.4f, %.4f]\n", v_full[0], v_full[1], v_full[2]);
        }
        release_weight(&l);
    }

    if (g->cfg.has_biases && v_full) {
        snprintf(name, sizeof(name), "blk.%u.attn_v.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < KV_dim; ++i) v_full[i] += bias[i];
            release_weight(&l);
        }
    }

    uint32_t page_tokens = 64;
    uint32_t kv_capacity = g->kv_capacity;
    uint32_t slide_window = g->cfg.sliding_window;
    if (pos >= slide_window && slide_window > 0) {
        uint32_t evict_page = (pos - slide_window) / page_tokens;
        if (evict_page < kv_capacity) {
            uint32_t lp = layer * kv_capacity + evict_page;
            if (g->kv_k[lp]) { free(g->kv_k[lp]); g->kv_k[lp] = NULL; }
            if (g->kv_v[lp]) { free(g->kv_v[lp]); g->kv_v[lp] = NULL; }
        }
    }
    if (pos < g->cfg.n_ctx) {
        uint32_t page_idx = pos / page_tokens;
        uint32_t slot_off = (pos % page_tokens) * KV_dim;
        uint32_t lp = layer * kv_capacity + page_idx;
        if (page_idx < kv_capacity) {
            if (!g->kv_k[lp]) {
                g->kv_k[lp] = (float *)calloc(page_tokens * KV_dim, sizeof(float));
                g->kv_v[lp] = (float *)calloc(page_tokens * KV_dim, sizeof(float));
                g->kv_dim[layer] = KV_dim;
            }
            if (g->kv_k[lp] && k_full) memcpy(g->kv_k[lp] + slot_off, k_full, KV_dim * sizeof(float));
            if (g->kv_v[lp] && v_full) memcpy(g->kv_v[lp] + slot_off, v_full, KV_dim * sizeof(float));
        }
    }

    uint32_t attn_dim = H * HD;
    if (attn_dim > g->work_att_out_cap) {
        float *tmp = (float *)realloc(g->work_att_out, (attn_dim + 1) * sizeof(float));
        if (tmp) { g->work_att_out = tmp; g->work_att_out_cap = attn_dim + 1; }
    }
    memset(g->work_att_out, 0, attn_dim * sizeof(float));
    float scale = 1.0f / sqrtf((float)HD);
    uint32_t n_past = pos + 1;
    uint32_t kv_hdim = HKV ? (KV_dim / HKV) : HD;

    for (uint32_t h = 0; h < H; ++h) {
        uint32_t gk = H / HKV; gk = gk ? gk : 1;
        uint32_t kv_h = h / gk;
        float *hq = g->work_qkv + h * HD;
        float *scores = (float *)malloc(n_past * sizeof(float));
        if (!scores) continue;

        for (uint32_t t = 0; t < n_past; ++t) {
            uint32_t pi = t / page_tokens;
            uint32_t so = (t % page_tokens) * KV_dim + kv_h * kv_hdim;
            uint32_t lpg = layer * kv_capacity + pi;
            float *hk = (g->kv_k[lpg]) ? (g->kv_k[lpg] + so) : NULL;
            float dot = 0.0f;
            if (hk) { for (uint32_t d = 0; d < HD; ++d) dot += hq[d] * hk[d]; }
            scores[t] = dot * scale;
        }

        float mx = scores[0];
        for (uint32_t t = 1; t < n_past; ++t) if (scores[t] > mx) mx = scores[t];
        float s = 0.0f;
        for (uint32_t t = 0; t < n_past; ++t) { scores[t] = expf(scores[t] - mx); s += scores[t]; }
        if (s > 0.0f) for (uint32_t t = 0; t < n_past; ++t) scores[t] /= s;

        for (uint32_t t = 0; t < n_past; ++t) {
            uint32_t pi = t / page_tokens;
            uint32_t so = (t % page_tokens) * KV_dim + kv_h * kv_hdim;
            uint32_t lpg = layer * kv_capacity + pi;
            float *hv = (g->kv_v[lpg]) ? (g->kv_v[lpg] + so) : NULL;
            float w = scores[t];
            if (hv) { for (uint32_t d = 0; d < kv_hdim; ++d) g->work_att_out[h * kv_hdim + d] += w * hv[d]; }
        }
        free(scores);
    }

    int fused_moe_head = moe_fused_head_ok(g);
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    if (!fused_moe_head && (!vg_layer_tail_on() || find_tensor(g, "blk.0.ffn_gate_inp.weight") != NULL) && acquire_weight(g, name, &l) == VG_OK) {
        float *po = (float *)malloc(D * sizeof(float));
        if (po) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_att_out, po, D, attn_dim) == VG_OK) { } else
#endif
            { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_att_out, po, D, attn_dim); }
            for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + po[i];
            free(po);
        }
        release_weight(&l);
    } else if (!fused_moe_head) {
        for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i];
    }
    /* fused_moe_head: leave g->work_att_out (attention output) and g->work_res
     * (pre-attention hidden) untouched; ffn_layer's fused head consumes them. */

    free(k_full); free(v_full);
}

static void ffn_layer(VG_ModelGraph *g, uint32_t layer, float *hidden) {
    uint32_t D = g->cfg.n_embd;
    uint32_t FF = g->cfg.n_ff;
    char name[128];
    VG_TensorLease l;

    /* MoE fused head (COMPASS 3.1): attention_layer skipped its o_proj and left
     * attention in work_att_out and the pre-attention hidden in work_res.
     * Compute h1 = resid + o_proj*attn, xn = rmsnorm(h1), and the router logits
     * in ONE submit, then run only the experts in a second submit. Replaces the
     * former "o_proj submit + router submit" pair. */
    float *moe_logits = NULL;
    if (moe_fused_head_ok(g)) {
        uint32_t NE = g->cfg.n_expert > 0 ? g->cfg.n_expert : 1;
        float *hl = (float *)malloc((size_t)NE * sizeof(float));
        int head_ok = 0;
        if (hl) {
#ifdef VG_HAS_VULKAN
            VG_TensorLease lo, ln, lr;
            memset(&lo,0,sizeof(lo)); memset(&ln,0,sizeof(ln)); memset(&lr,0,sizeof(lr));
            char n1[128], n2[128], n3[128];
            snprintf(n1,sizeof(n1),"blk.%u.attn_output.weight",layer);
            snprintf(n2,sizeof(n2),"blk.%u.ffn_norm.weight",layer);
            snprintf(n3,sizeof(n3),"blk.%u.ffn_gate_inp.weight",layer);
            uint32_t adim = g->cfg.n_head*g->cfg.head_dim;
            if (adim == D && acquire_weight(g,n1,&lo)==VG_OK && acquire_weight(g,n2,&ln)==VG_OK &&
                acquire_weight(g,n3,&lr)==VG_OK && lo.tensor->ggml_type==8 && lr.tensor->ggml_type==8) {
                VG_VKBuffer *ow=NULL,*os=NULL,*nw=NULL,*rw=NULL,*rs=NULL;
                int okp = vk_q8_prepare(g, lo.tensor, 0x800u, lo.data, D, D, &ow, &os)==VG_OK;
                if (okp && vg_vk_cache_get(g->vk, ln.tensor, 0x9000u, &nw)!=VG_OK)
                    vg_vk_cache_acquire(g->vk, ln.tensor, 0x9000u, ln.data, (size_t)D*sizeof(float), &nw);
                if (okp && nw && vk_q8_prepare(g, lr.tensor, 0x10000u, lr.data, NE, D, &rw, &rs)==VG_OK &&
                    vg_vk_moe_head(g->vk, ow, os, g->work_att_out, g->work_res, nw, rw, rs, hidden, hl, D, NE, g->cfg.rms_eps)==VG_OK)
                    head_ok = 1;
            }
            release_weight(&lo); release_weight(&ln); release_weight(&lr);
#endif
        }
        if (head_ok) {
            moe_logits = hl;
        } else {
            free(hl);
            /* Fallback: reconstruct h1 = resid + o_proj*attn on the host. */
            float *po = (float *)malloc((size_t)D*sizeof(float));
            if (po) {
                char n1[128]; snprintf(n1,sizeof(n1),"blk.%u.attn_output.weight",layer);
                if (acquire_weight(g,n1,&l)==VG_OK) {
                    uint32_t adim = g->cfg.n_head*g->cfg.head_dim;
#ifdef VG_HAS_VULKAN
                    if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_att_out, po, D, adim)==VG_OK) { } else
#endif
                    { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_att_out, po, D, adim); }
                    release_weight(&l);
                } else { memset(po, 0, (size_t)D*sizeof(float)); }
                for (uint32_t i=0;i<D;++i) hidden[i] = g->work_res[i] + po[i];
                free(po);
            } else {
                for (uint32_t i=0;i<D;++i) hidden[i] = g->work_res[i];
            }
        }
    }

    /* Save original for residual (hidden may alias work_embd) */
    memcpy(g->work_res, hidden, D * sizeof(float));
#ifdef VG_HAS_VULKAN
    if (vg_layer_tail_on() && g->vk) {
        VG_TensorLease lo, ln, lg, lu, ld;
        memset(&lo,0,sizeof(lo)); memset(&ln,0,sizeof(ln)); memset(&lg,0,sizeof(lg)); memset(&lu,0,sizeof(lu)); memset(&ld,0,sizeof(ld));
        char tn1[128], tn2[128], tn3[128], tn4[128], tn5[128];
        snprintf(tn1,sizeof(tn1),"blk.%u.attn_output.weight",layer);
        snprintf(tn2,sizeof(tn2),"blk.%u.ffn_norm.weight",layer);
        snprintf(tn3,sizeof(tn3),"blk.%u.ffn_gate.weight",layer);
        snprintf(tn4,sizeof(tn4),"blk.%u.ffn_up.weight",layer);
        snprintf(tn5,sizeof(tn5),"blk.%u.ffn_down.weight",layer);
        int tok = acquire_weight(g, tn1, &lo)==VG_OK && acquire_weight(g, tn2, &ln)==VG_OK &&
                  acquire_weight(g, tn3, &lg)==VG_OK && acquire_weight(g, tn4, &lu)==VG_OK &&
                  acquire_weight(g, tn5, &ld)==VG_OK;
        int did = 0;
        if (tok && lo.tensor->ggml_type==8 && lg.tensor->ggml_type==8 && lu.tensor->ggml_type==8 && ld.tensor->ggml_type==8 &&
            g->cfg.n_head*g->cfg.head_dim == D) {
            VG_VKBuffer *ow=NULL,*os=NULL,*gw2=NULL,*gs2=NULL,*uw2=NULL,*us2=NULL,*dw3=NULL,*ds3=NULL,*nw=NULL;
            int okp = vk_q8_prepare(g, lo.tensor, 0x800u, lo.data, D, D, &ow, &os)==VG_OK &&
                      vk_q8_prepare(g, lg.tensor, 0x800u, lg.data, FF, D, &gw2, &gs2)==VG_OK &&
                      vk_q8_prepare(g, lu.tensor, 0x800u, lu.data, FF, D, &uw2, &us2)==VG_OK &&
                      vk_q8_prepare(g, ld.tensor, 0x800u, ld.data, D, FF, &dw3, &ds3)==VG_OK;
            if (okp) {
                if (vg_vk_cache_get(g->vk, ln.tensor, 0x9000u, &nw)!=VG_OK)
                    vg_vk_cache_acquire(g->vk, ln.tensor, 0x9000u, ln.data, (size_t)D*sizeof(float), &nw);
                if (nw && vg_vk_layer_tail(g->vk, ow, os, g->work_att_out, g->work_res, nw, gw2, gs2, uw2, us2, dw3, ds3, hidden, D, FF, g->cfg.rms_eps)==VG_OK) did = 1;
            }
        }
        if (!did && tok) {
            float *po = (float *)malloc((size_t)D*sizeof(float));
            if (po) {
                uint32_t adim = g->cfg.n_head*g->cfg.head_dim;
                if (!(g->vk && vk_matmul(g, lo.tensor->ggml_type, lo.data, lo.size, g->work_att_out, po, D, adim)==VG_OK))
                    vg_cpu_matmul(lo.tensor->ggml_type, lo.data, g->work_att_out, po, D, adim);
                for (uint32_t i=0;i<D;++i) hidden[i] = g->work_res[i] + po[i];
                free(po);
            }
        }
        release_weight(&lo); release_weight(&ln); release_weight(&lg); release_weight(&lu); release_weight(&ld);
        if (did) return;
    }
#endif

    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    if (find_tensor(g, "blk.0.ffn_gate_inp.weight") != NULL) {
        /* MoE FFN: router -> top-k experts -> weighted sum. Expert tensors are
         * [D, FF, n_expert] with each expert's slice stored contiguously. */
        uint32_t NE = g->cfg.n_expert > 0 ? g->cfg.n_expert : 1;
        uint32_t NU = g->cfg.n_expert_used > 0 ? g->cfg.n_expert_used : 1;
        if (NU > NE) NU = NE;
        if (NU > 8) NU = 8;
        float *logits = (float *)malloc((size_t)NE * sizeof(float));
        float *gate_out = (float *)malloc((size_t)FF * sizeof(float));
        float *up_out = (float *)malloc((size_t)FF * sizeof(float));
        float *down_out = (float *)malloc((size_t)D * sizeof(float));
        float *acc = (float *)calloc((size_t)D, sizeof(float));
        if (logits && gate_out && up_out && down_out && acc) {
            if (moe_logits) {
                /* Router logits already produced by the fused head (COMPASS 3.1). */
                memcpy(logits, moe_logits, (size_t)NE * sizeof(float));
            } else {
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
            if (acquire_weight(g, name, &l) == VG_OK) {
                /* Stable key = tensor descriptor (l.tensor); tag 0x10000 = router.
                 * The router is shared across experts so it caches once. */
#ifdef VG_HAS_VULKAN
                if (g->vk && vk_matmul_slice(g, l.tensor->ggml_type, l.tensor, 0x10000u,
                                             l.data, l.size, g->work_embd, logits, NE, D) == VG_OK) { } else
#endif
                { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, logits, NE, D); }
                release_weight(&l);
            } else {
                for (uint32_t e = 0; e < NE; ++e) logits[e] = 0.0f;
            }
            }
            uint32_t ids[8] = {0};
            float wts[8] = {0};
            for (uint32_t k = 0; k < NU; ++k) {
                uint32_t best = 0; float bv = -FLT_MAX;
                for (uint32_t e = 0; e < NE; ++e) {
                    int used = 0;
                    for (uint32_t j = 0; j < k; ++j) if (ids[j] == e) used = 1;
                    if (!used && logits[e] > bv) { bv = logits[e]; best = e; }
                }
                ids[k] = best; wts[k] = bv;
            }
            float mx = wts[0];
            for (uint32_t k = 1; k < NU; ++k) if (wts[k] > mx) mx = wts[k];
            float wsum = 0.0f;
            for (uint32_t k = 0; k < NU; ++k) { wts[k] = expf(wts[k] - mx); wsum += wts[k]; }
            if (wsum > 0.0f) for (uint32_t k = 0; k < NU; ++k) wts[k] /= wsum;
            /* COMPASS 3.1: batch every routed expert's matvecs into 2 submits
             * (gate+up for all experts, then down for all experts). */
            VG_TensorLease gl2[8], ul2[8], dl2[8];
            const char *gp2[8], *up2[8], *dp2[8];
            uint64_t sg2[8], su2[8], sd2[8];
            float *go2[8], *uo2[8], *wo2[8], *do2[8];
            for (uint32_t k = 0; k < 8; ++k) {
                memset(&gl2[k], 0, sizeof(gl2[k])); memset(&ul2[k], 0, sizeof(ul2[k])); memset(&dl2[k], 0, sizeof(dl2[k]));
                gp2[k] = up2[k] = dp2[k] = NULL; sg2[k] = su2[k] = sd2[k] = 0;
                go2[k] = uo2[k] = wo2[k] = do2[k] = NULL;
            }
            int have_all = 1;
            for (uint32_t k = 0; k < NU; ++k) {
                uint32_t e = ids[k];
                snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", layer);
                if (acquire_weight(g, name, &gl2[k]) != VG_OK) { have_all = 0; break; }
                snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", layer);
                if (acquire_weight(g, name, &ul2[k]) != VG_OK) { have_all = 0; break; }
                snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", layer);
                if (acquire_weight(g, name, &dl2[k]) != VG_OK) { have_all = 0; break; }
                sg2[k] = gl2[k].tensor->nbytes / (NE ? NE : 1);
                su2[k] = ul2[k].tensor->nbytes / (NE ? NE : 1);
                sd2[k] = dl2[k].tensor->nbytes / (NE ? NE : 1);
                gp2[k] = (const char *)gl2[k].data + e * sg2[k];
                up2[k] = (const char *)ul2[k].data + e * su2[k];
                dp2[k] = (const char *)dl2[k].data + e * sd2[k];
                go2[k] = (float *)malloc((size_t)FF * sizeof(float));
                uo2[k] = (float *)malloc((size_t)FF * sizeof(float));
                wo2[k] = (float *)malloc((size_t)FF * sizeof(float));
                do2[k] = (float *)malloc((size_t)D * sizeof(float));
                if (!go2[k] || !uo2[k] || !wo2[k] || !do2[k]) { have_all = 0; break; }
            }
            int done_gpu = 0;
#ifdef VG_HAS_VULKAN
            if (have_all && g->vk) {
                int ok = 1;
                VG_VKBuffer *EW[2][6]; memset(EW, 0, sizeof(EW));
                for (uint32_t k = 0; k < NU && ok; ++k) {
                    uint32_t et = ids[k] << 16u;
                    if (k >= 2 || gl2[k].tensor->ggml_type != 8 || ul2[k].tensor->ggml_type != 8 || dl2[k].tensor->ggml_type != 8) { ok = 0; break; }
                    if (vk_q8_prepare(g, gl2[k].tensor, et | 0u, gp2[k], FF, D, &EW[k][0], &EW[k][1]) != VG_OK) { ok = 0; break; }
                    if (vk_q8_prepare(g, ul2[k].tensor, et | 2u, up2[k], FF, D, &EW[k][2], &EW[k][3]) != VG_OK) { ok = 0; break; }
                    if (vk_q8_prepare(g, dl2[k].tensor, et | 4u, dp2[k], D, FF, &EW[k][4], &EW[k][5]) != VG_OK) { ok = 0; break; }
                }
                if (ok && NU == 2) {
                    if (vg_vk_ffn_fused_x2(g->vk, EW[0][0],EW[0][1],EW[0][2],EW[0][3],EW[0][4],EW[0][5],
                                                  EW[1][0],EW[1][1],EW[1][2],EW[1][3],EW[1][4],EW[1][5],
                                                  g->work_embd, do2[0], do2[1], D, FF) == VG_OK) done_gpu = 1;
                } else if (ok) {
                    for (uint32_t k = 0; k < NU; ++k)
                        if (vg_vk_ffn_fused(g->vk, EW[k][0],EW[k][1],EW[k][2],EW[k][3],EW[k][4],EW[k][5], g->work_embd, do2[k], D, FF) != VG_OK) { ok = 0; break; }
                    if (ok) done_gpu = 1;
                }
            }
#endif
            if (!done_gpu && have_all) {
                for (uint32_t k = 0; k < NU; ++k) {
                    vg_cpu_matmul(gl2[k].tensor->ggml_type, gp2[k], g->work_embd, go2[k], FF, D);
                    vg_cpu_matmul(ul2[k].tensor->ggml_type, up2[k], g->work_embd, uo2[k], FF, D);
                    vg_cpu_swiglu(wo2[k], go2[k], uo2[k], FF);
                    vg_cpu_matmul(dl2[k].tensor->ggml_type, dp2[k], wo2[k], do2[k], D, FF);
                }
            }
            if (have_all) {
                for (uint32_t k = 0; k < NU; ++k)
                    for (uint32_t i = 0; i < D; ++i) acc[i] += wts[k] * do2[k][i];
            }
            for (uint32_t k = 0; k < 8; ++k) {
                release_weight(&gl2[k]); release_weight(&ul2[k]); release_weight(&dl2[k]);
                free(go2[k]); free(uo2[k]); free(wo2[k]); free(do2[k]);
            }
            for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + acc[i];
        }
        free(logits); free(gate_out); free(up_out); free(down_out); free(acc);
        free(moe_logits);
        return;
    }

    if (g->cfg.has_gate_up) {
#ifdef VG_HAS_VULKAN
        if (g->vk) {
            VG_TensorLease fg, fu, fd; memset(&fg,0,sizeof(fg)); memset(&fu,0,sizeof(fu)); memset(&fd,0,sizeof(fd));
            char fgn[128], fun[128], fdn[128];
            snprintf(fgn,sizeof(fgn),"blk.%u.ffn_gate.weight",layer);
            snprintf(fun,sizeof(fun),"blk.%u.ffn_up.weight",layer);
            snprintf(fdn,sizeof(fdn),"blk.%u.ffn_down.weight",layer);
            if (acquire_weight(g, fgn, &fg) == VG_OK && acquire_weight(g, fun, &fu) == VG_OK && acquire_weight(g, fdn, &fd) == VG_OK &&
                fg.tensor->ggml_type == 8 && fu.tensor->ggml_type == 8 && fd.tensor->ggml_type == 8) {
                VG_VKBuffer *gw=NULL,*gs=NULL,*uw=NULL,*us=NULL,*dw=NULL,*ds=NULL;
                if (vk_q8_prepare(g, fg.tensor, 0u, fg.data, FF, D, &gw, &gs) == VG_OK &&
                    vk_q8_prepare(g, fu.tensor, 0u, fu.data, FF, D, &uw, &us) == VG_OK &&
                    vk_q8_prepare(g, fd.tensor, 0u, fd.data, D, FF, &dw, &ds) == VG_OK) {
                    float *fout = (float *)malloc((size_t)D * sizeof(float)); VG_Status fst = VG_E_UNSUPPORTED;
                    fst = fout ? vg_vk_ffn_fused(g->vk, gw, gs, uw, us, dw, ds, g->work_embd, fout, D, FF) : VG_E_NOMEM; if (fst == VG_OK) {
                        for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + fout[i];
                        free(fout); release_weight(&fg); release_weight(&fu); release_weight(&fd); return;
                    }
                    free(fout);
                }
            }
            release_weight(&fg); release_weight(&fu); release_weight(&fd);
        }
#endif

        float *gate_out = (float *)malloc(FF * sizeof(float));
        float *up_out = (float *)malloc(FF * sizeof(float));
        VG_TensorLease lg, lu; memset(&lg, 0, sizeof(lg)); memset(&lu, 0, sizeof(lu));
        char gn[128], un[128];
        snprintf(gn, sizeof(gn), "blk.%u.ffn_gate.weight", layer);
        snprintf(un, sizeof(un), "blk.%u.ffn_up.weight", layer);
        if (gate_out && up_out && acquire_weight(g, gn, &lg) == VG_OK && acquire_weight(g, un, &lu) == VG_OK) {
            int gpu_done = 0; /* COMPASS 3.1: gate+up in ONE submit */
#ifdef VG_HAS_VULKAN
            if (g->vk && lg.tensor->ggml_type == 8 && lu.tensor->ggml_type == 8) {
                VG_VKBuffer *wb = NULL, *sb = NULL;
                VG_VKMatvecReq rq[2];
                if (vk_q8_prepare(g, lg.tensor, 0u, lg.data, FF, D, &wb, &sb) == VG_OK) {
                    rq[0].weights = wb; rq[0].scales = sb; rq[0].x = g->work_embd; rq[0].y = gate_out; rq[0].in_dim = D; rq[0].out_dim = FF;
                    if (vk_q8_prepare(g, lu.tensor, 0u, lu.data, FF, D, &wb, &sb) == VG_OK) {
                        rq[1].weights = wb; rq[1].scales = sb; rq[1].x = g->work_embd; rq[1].y = up_out; rq[1].in_dim = D; rq[1].out_dim = FF;
                        if (vg_vk_matvec_q8_0_batch(g->vk, rq, 2) == VG_OK) gpu_done = 1;
                    }
                }
            }
#endif
            if (!gpu_done) {
                vg_cpu_matmul(lg.tensor->ggml_type, lg.data, g->work_embd, gate_out, FF, D);
                vg_cpu_matmul(lu.tensor->ggml_type, lu.data, g->work_embd, up_out, FF, D);
            }
            vg_cpu_swiglu(g->work_ffn, gate_out, up_out, FF);
        }
        release_weight(&lg); release_weight(&lu);
        free(gate_out); free(up_out);
    } else {
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, g->work_ffn, FF, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, g->work_ffn, FF, D);
            }
            release_weight(&l);
        }
        vg_cpu_gelu(g->work_ffn, FF);
    }

    snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        float *down_out = (float *)malloc(D * sizeof(float));
        if (down_out) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_ffn, down_out, D, FF) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_ffn, down_out, D, FF);
            }
            for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + down_out[i];
            free(down_out);
        }
        release_weight(&l);
    }
}

VG_Status vg_model_graph_decode(VG_ModelGraph *g, int32_t input_token, int32_t kv_pos, float *logits_out) {
    if (!g) return VG_E_INVALID;
    if ((uint32_t)kv_pos >= g->cfg.n_ctx) return VG_E_RANGE;
    uint32_t D = g->cfg.n_embd;
    int gpu_logits_done_local = 0;
    float pos_f = (float)kv_pos;

    VG_Status st = vg_model_graph_embed(g, input_token, g->work_embd);
    if (st != VG_OK) return st;

#ifdef VG_HAS_VULKAN
    if (!g->gpu_dense_checked) { g->gpu_dense_checked = 1; g->gpu_dense = gpu_dense_ok(g); }
    if (g->gpu_dense) {
        uint32_t nl = g->cfg.n_layer;
        if (g->gpu_layers_cap < nl) {
            VG_VKDenseLayer *tmp = (VG_VKDenseLayer *)realloc(g->gpu_layers, (size_t)nl * sizeof(VG_VKDenseLayer));
            if (tmp) { g->gpu_layers = tmp; g->gpu_layers_cap = nl; }
            else g->gpu_dense = 0;
        }
        int all_ok = g->gpu_dense ? 1 : 0;
        for (uint32_t l = 0; all_ok && l < nl; ++l)
            if (!dense_gpu_layer_forward(g, l, (uint32_t)kv_pos, l == 0, l + 1 == nl, g->work_embd, &g->gpu_layers[l])) all_ok = 0;

        /* Prepare the optional device-resident final RMSNorm + Q8 lm_head.
         * Failure here must not disable the already-valid fused transformer
         * path: it simply falls back to the existing host finalisation. */
        VG_VKFinalHead final_head;
        memset(&final_head, 0, sizeof(final_head));
        VG_TensorLease loutn, lout;
        memset(&loutn, 0, sizeof(loutn));
        memset(&lout, 0, sizeof(lout));
        VG_VKBuffer *final_norm = NULL, *final_lmw = NULL, *final_lms = NULL;
        const VG_GGUF_Tensor *lt_final = find_tensor(g, "output.weight");
        if (!lt_final) lt_final = find_tensor(g, "token_embd.weight");
        int final_ok = 0;
        if (all_ok && lt_final && lt_final->ggml_type == 8 &&
            acquire_weight(g, "output_norm.weight", &loutn) == VG_OK &&
            acquire_weight(g, lt_final->name, &lout) == VG_OK) {
            int prep_ok =
                (vg_vk_cache_get(g->vk, loutn.tensor, 0x9200u, &final_norm) == VG_OK ||
                 vg_vk_cache_acquire(g->vk, loutn.tensor, 0x9200u, loutn.data,
                                     (size_t)D * sizeof(float), &final_norm) == VG_OK) &&
                (vk_q8_prepare(g, lout.tensor, 0x30000u, lout.data,
                               g->cfg.n_vocab, D, &final_lmw, &final_lms) == VG_OK);
            if (prep_ok) {
                final_head.norm_w = final_norm;
                final_head.lm_w = final_lmw;
                final_head.lm_s = final_lms;
                final_head.vocab = g->cfg.n_vocab;
                final_head.D = D;
                final_head.eps = g->cfg.rms_eps;
                final_head.logits_out = g->logits;
                final_ok = 1;
            }
        }

        int gpu_done = 0;
        if (final_ok && vg_vk_dense_forward_final(g->vk, g->gpu_layers, nl, &final_head) == VG_OK) {
            gpu_done = 1;
            gpu_logits_done_local = 1;
        }
        else if (all_ok && vg_vk_dense_forward(g->vk, g->gpu_layers, nl) == VG_OK)
            gpu_done = 1;

        release_weight(&loutn);
        release_weight(&lout);

        if (gpu_done) {
            /* Final-head success skips the host output norm/lm_head below;
             * ordinary dense GPU success retains the existing host finalisation. */
            if (gpu_logits_done_local) goto after_layers;
            goto after_layers;
        }
        g->gpu_dense = 0;   /* permanent fallback for this process */
    }

#endif

    for (uint32_t l = 0; l < g->cfg.n_layer; ++l) {
        if (g->source) {
            char prefetch_name[128];
            if (l + 1 < g->cfg.n_layer) {
                snprintf(prefetch_name, sizeof(prefetch_name), "blk.%u.attn_norm.weight", l + 1);
                vg_tensor_prefetch_after(g->source, prefetch_name, 4);
            }
        }
        uint64_t ta0 = vg_trace_now_ns();
        attention_layer(g, l, g->work_embd, &pos_f);
        uint64_t ta1 = vg_trace_now_ns();
        ffn_layer(g, l, g->work_embd);
        uint64_t ta2 = vg_trace_now_ns();
        {
            static uint64_t a_acc = 0, f_acc = 0; static uint64_t a_n = 0;
            a_acc += ta1 - ta0; f_acc += ta2 - ta1; ++a_n;
            if ((a_n % ((uint64_t)g->cfg.n_layer * 4u)) == 0)
                fprintf(stderr, "[vg] layer avg att=%.2fms ffn=%.2fms\n", (double)a_acc / 1e6 / (double)a_n, (double)f_acc / 1e6 / (double)a_n);
        }
    }
#ifdef VG_HAS_VULKAN
after_layers: ;
#endif
    char name[128]; VG_TensorLease wl;
#ifdef VG_HAS_VULKAN
    if (gpu_logits_done_local) goto logits_ready;
#endif
    snprintf(name, sizeof(name), "output_norm.weight");
    if (acquire_weight(g, name, &wl) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, g->work_embd, (const float *)wl.data, D, g->cfg.rms_eps);
        release_weight(&wl);
    }

    /* Compute logits via output projection (lm_head). */
    const VG_GGUF_Tensor *lt = find_tensor(g, "output.weight");
    if (!lt) { lt = find_tensor(g, "token_embd.weight"); }
    if (lt) {
        uint32_t ggml_type = lt->ggml_type;
        const VG_QuantInfo *qi = vg_quant_info(ggml_type);
        int logits_done = 0;
#ifdef VG_HAS_VULKAN
        /* GPU lm_head: 128K x D projection is ~400ms per token on the scalar
         * CPU path; the Q8_0 dp4a kernel does it in ~1ms. */
        if (g->vk && g->source && ggml_type == 8) {
            VG_TensorLease ol;
            if (vg_tensor_acquire(g->source, lt->name, VG_TIER_STORAGE, &ol) == VG_OK) {
                /* Key the q8 repack on the stable tensor descriptor, not the
                 * transient lease pointer, so the ~260MB lm_head stays resident. */
                if (vk_matmul_slice(g, ggml_type, ol.tensor, 0x30000u, ol.data, ol.size, g->work_embd, g->logits, g->cfg.n_vocab, D) == VG_OK) logits_done = 1;
                vg_tensor_release(&ol);
            }
        }
#endif
        if (logits_done) {
            /* done on GPU */
        } else if (ggml_type == 0) {
            uint32_t row_bytes = D * sizeof(float);
            float *row_buf = (float *)malloc(row_bytes);
            if (row_buf) {
                for (uint32_t v = 0; v < g->cfg.n_vocab; ++v) {
                    uint64_t offset = (uint64_t)v * row_bytes;
                    if (vg_gguf_read(g->file, lt, offset, row_buf, row_bytes) == VG_OK) {
                        float acc = 0; for (uint32_t i = 0; i < D; ++i) acc += row_buf[i] * g->work_embd[i];
                        g->logits[v] = acc;
                    } else {
                        g->logits[v] = -FLT_MAX;
                    }
                }
                free(row_buf);
            }
        } else if (qi && qi->block_size > 0) {
            uint32_t blocks_per_row = D / qi->block_size;
            uint64_t row_bytes = (uint64_t)blocks_per_row * qi->bytes_per_block;
            size_t deq_buf_size = D * sizeof(float);
            uint32_t batch = 256;
            float *deq_buf = (float *)malloc(deq_buf_size + batch * row_bytes);
            if (deq_buf) {
                void *quant_buf = (char *)deq_buf + deq_buf_size;
                for (uint32_t v = 0; v < g->cfg.n_vocab; v += batch) {
                    uint32_t cnt = g->cfg.n_vocab - v < batch ? g->cfg.n_vocab - v : batch;
                    uint64_t offset = (uint64_t)v * row_bytes;
                    if (vg_gguf_read(g->file, lt, offset, quant_buf, cnt * row_bytes) == VG_OK) {
                        for (uint32_t i = 0; i < cnt; ++i) {
                            vg_dequantize_row(ggml_type, (const char *)quant_buf + i * row_bytes, row_bytes, deq_buf, D);
                            float acc = 0; for (uint32_t j = 0; j < D; ++j) acc += deq_buf[j] * g->work_embd[j];
                            g->logits[v + i] = acc;
                        }
                    } else {
                        for (uint32_t i = 0; i < cnt; ++i) g->logits[v + i] = -FLT_MAX;
                    }
                }
                free(deq_buf);
            }
        }
    } else {
        memset(g->logits, 0, g->cfg.n_vocab * sizeof(float));
    }

#ifdef VG_HAS_VULKAN
logits_ready:
#endif
    if (logits_out) memcpy(logits_out, g->logits, g->cfg.n_vocab * sizeof(float));
    return VG_OK;
}

const float *vg_model_graph_logits(const VG_ModelGraph *g) { return g ? g->logits : NULL; }

#ifdef VG_HAS_VULKAN
void vg_model_graph_set_vulkan(VG_ModelGraph *g, VG_VK *vk) { if (g) g->vk = vk; }

static VG_Status vk_matmul(VG_ModelGraph *g, uint32_t ggml_type, const void *wdata, size_t wbytes,
                            const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    if (!g->vk) return VG_E_UNSUPPORTED;
    if (!wdata || !x || !y || !out_dim || !in_dim) return VG_E_INVALID;

    if (ggml_type == 8) {
        /* Q8_0 fast path: repack into aligned int8 blocks + fp16 scales ONCE,
         * cache the result in device-local VRAM (COMPASS 2.1), dispatch the
         * dp4a kernel with batched submit (3.1). */
        if (in_dim % 32u || out_dim * (in_dim / 32u) == 0) return VG_E_UNSUPPORTED;
        uint32_t nb = out_dim * (in_dim / 32u);
        VG_VKBuffer *wb = NULL, *sb = NULL;
        VG_Status st;
        if (vg_vk_cache_get(g->vk, wdata, 0, &wb) != VG_OK) {
            uint32_t *qs = (uint32_t *)malloc((size_t)nb * 32u);
            if (!qs) return VG_E_NOMEM;
            const unsigned char *p = (const unsigned char *)wdata;
            for (uint32_t b = 0; b < nb; ++b) memcpy((unsigned char *)qs + (size_t)b * 32u, p + (size_t)b * 34u + 2u, 32u);
            st = vg_vk_cache_acquire(g->vk, wdata, 0, qs, (size_t)nb * 32u, &wb);
            free(qs);
            if (st != VG_OK) return st;
        }
        if (vg_vk_cache_get(g->vk, wdata, 1, &sb) != VG_OK) {
            uint32_t n_scales = (nb + 1u) / 2u;
            uint32_t *scales = (uint32_t *)malloc((size_t)n_scales * 4u);
            if (!scales) return VG_E_NOMEM;
            const unsigned char *p = (const unsigned char *)wdata;
            for (uint32_t i = 0; i < nb; i += 2u) {
                uint16_t s0 = *(const uint16_t *)(p + (size_t)i * 34u);
                uint16_t s1 = (i + 1u < nb) ? *(const uint16_t *)(p + (size_t)(i + 1u) * 34u) : 0;
                scales[i / 2u] = (uint32_t)s0 | ((uint32_t)s1 << 16);
            }
            st = vg_vk_cache_acquire(g->vk, wdata, 1, scales, (size_t)n_scales * 4u, &sb);
            free(scales);
            if (st != VG_OK) return st;
        }
        VG_VKMatvecReq r;
        r.weights = wb; r.scales = sb; r.x = x; r.y = y; r.in_dim = in_dim; r.out_dim = out_dim;
        return vg_vk_matvec_q8_0_batch(g->vk, &r, 1);
    }

    if (ggml_type == 0 || ggml_type == 1) {
        VG_VKBuffer *wb = NULL, *sb = NULL;
        VG_Status st;
        if (vg_vk_cache_get(g->vk, wdata, 0, &wb) != VG_OK) {
            st = vg_vk_cache_acquire(g->vk, wdata, 0, wdata, wbytes, &wb);
            if (st != VG_OK) return st;
        }
        if (vg_vk_cache_get(g->vk, wdata, 2, &sb) != VG_OK) {
            float one = 1.0f;
            st = vg_vk_cache_acquire(g->vk, wdata, 2, &one, sizeof(one), &sb);
            if (st != VG_OK) return st;
        }
        VG_VKMatvecReq r;
        r.weights = wb; r.scales = sb; r.x = x; r.y = y; r.in_dim = in_dim; r.out_dim = out_dim;
        if (ggml_type == 0) return vg_vk_matvec_f32_batch(g->vk, &r, 1);
        return vg_vk_matvec_f16_batch(g->vk, &r, 1);
    }

    if (ggml_type == 12) {
        /* Q4_K: upload the raw 144-byte blocks once (keyed by the stable mmap
         * data pointer) and run the on-disk-layout matvec kernel. */
        if (in_dim % 256u) return VG_E_UNSUPPORTED;
        VG_VKBuffer *wb = NULL;
        if (vg_vk_cache_get(g->vk, wdata, 0, &wb) != VG_OK) {
            VG_Status st = vg_vk_cache_acquire(g->vk, wdata, 0, wdata, wbytes, &wb);
            if (st != VG_OK) return st;
        }
        return vg_vk_matvec_q4_k(g->vk, wb, x, y, out_dim, in_dim, out_dim);
    }

    return VG_E_UNSUPPORTED;
}

/* Stable-key variant: the cache key is `key` (a stable pointer, e.g. the tensor
 * descriptor) and `tag` encodes (expert_index<<16)|tensor_kind. The actual
 * matmul bytes come from `wdata`/`wbytes` (a slice of a larger expert tensor).
 * This keeps hot experts resident in VRAM across tokens even though the lease
 * pointer is transient. */
static VG_Status vk_matmul_slice(VG_ModelGraph *g, uint32_t ggml_type, const void *key,
                                 uint32_t tag, const void *wdata, size_t wbytes,
                                 const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    if (!g->vk) return VG_E_UNSUPPORTED;
    if (!key || !wdata || !x || !y || !out_dim || !in_dim) return VG_E_INVALID;
    if (ggml_type == 8) {
        if (in_dim % 32u || out_dim * (in_dim / 32u) == 0) return VG_E_UNSUPPORTED;
        uint32_t nb = out_dim * (in_dim / 32u);
        VG_VKBuffer *wb = NULL, *sb = NULL;
        VG_Status st;
        if (vg_vk_cache_get(g->vk, key, tag, &wb) != VG_OK) {
            uint32_t *qs = (uint32_t *)malloc((size_t)nb * 32u);
            if (!qs) return VG_E_NOMEM;
            const unsigned char *p = (const unsigned char *)wdata;
            for (uint32_t b = 0; b < nb; ++b) memcpy((unsigned char *)qs + (size_t)b * 32u, p + (size_t)b * 34u + 2u, 32u);
            st = vg_vk_cache_acquire(g->vk, key, tag, qs, (size_t)nb * 32u, &wb);
            free(qs);
            if (st != VG_OK) return st;
        }
        if (vg_vk_cache_get(g->vk, key, tag | 1u, &sb) != VG_OK) {
            uint32_t n_scales = (nb + 1u) / 2u;
            uint32_t *scales = (uint32_t *)malloc((size_t)n_scales * 4u);
            if (!scales) return VG_E_NOMEM;
            const unsigned char *p = (const unsigned char *)wdata;
            for (uint32_t i = 0; i < nb; i += 2u) {
                uint16_t s0 = *(const uint16_t *)(p + (size_t)i * 34u);
                uint16_t s1 = (i + 1u < nb) ? *(const uint16_t *)(p + (size_t)(i + 1u) * 34u) : 0;
                scales[i / 2u] = (uint32_t)s0 | ((uint32_t)s1 << 16);
            }
            st = vg_vk_cache_acquire(g->vk, key, tag | 1u, scales, (size_t)n_scales * 4u, &sb);
            free(scales);
            if (st != VG_OK) return st;
        }
        VG_VKMatvecReq r;
        r.weights = wb; r.scales = sb; r.x = x; r.y = y; r.in_dim = in_dim; r.out_dim = out_dim;
        return vg_vk_matvec_q8_0_batch(g->vk, &r, 1);
    }
    if (ggml_type == 0 || ggml_type == 1) {
        VG_VKBuffer *wb = NULL, *sb = NULL;
        VG_Status st;
        if (vg_vk_cache_get(g->vk, key, tag, &wb) != VG_OK) {
            st = vg_vk_cache_acquire(g->vk, key, tag, wdata, wbytes, &wb);
            if (st != VG_OK) return st;
        }
        if (vg_vk_cache_get(g->vk, key, tag | 1u, &sb) != VG_OK) {
            float one = 1.0f;
            st = vg_vk_cache_acquire(g->vk, key, tag | 1u, &one, sizeof(one), &sb);
            if (st != VG_OK) return st;
        }
        VG_VKMatvecReq r;
        r.weights = wb; r.scales = sb; r.x = x; r.y = y; r.in_dim = in_dim; r.out_dim = out_dim;
        if (ggml_type == 0) return vg_vk_matvec_f32_batch(g->vk, &r, 1);
        return vg_vk_matvec_f16_batch(g->vk, &r, 1);
    }
    if (ggml_type == 12) {
        if (in_dim % 256u) return VG_E_UNSUPPORTED;
        VG_VKBuffer *wb = NULL;
        if (vg_vk_cache_get(g->vk, key, tag, &wb) != VG_OK) {
            VG_Status st = vg_vk_cache_acquire(g->vk, key, tag, wdata, wbytes, &wb);
            if (st != VG_OK) return st;
        }
        return vg_vk_matvec_q4_k(g->vk, wb, x, y, out_dim, in_dim, out_dim);
    }
    return VG_E_UNSUPPORTED;
}

/* Resolve (repack+cache) a Q8_0 weight slice into device buffers without
 * dispatching, so several matvecs can share ONE batched submit (COMPASS 3.1). */
static VG_Status vk_q8_prepare(VG_ModelGraph *g, const void *key, uint32_t tag,
                               const void *wdata, uint32_t out_dim, uint32_t in_dim,
                               VG_VKBuffer **wb, VG_VKBuffer **sb) {
    if (!g->vk || !key || !wdata) return VG_E_INVALID;
    if (in_dim % 32u || out_dim * (in_dim / 32u) == 0) return VG_E_UNSUPPORTED;
    uint32_t nb = out_dim * (in_dim / 32u);
    if (vg_vk_cache_get(g->vk, key, tag, wb) != VG_OK) {
        uint32_t *qs = (uint32_t *)malloc((size_t)nb * 32u);
        if (!qs) return VG_E_NOMEM;
        const unsigned char *p = (const unsigned char *)wdata;
        for (uint32_t b = 0; b < nb; ++b) memcpy((unsigned char *)qs + (size_t)b * 32u, p + (size_t)b * 34u + 2u, 32u);
        VG_Status st = vg_vk_cache_acquire(g->vk, key, tag, qs, (size_t)nb * 32u, wb);
        free(qs);
        if (st != VG_OK) return st;
    }
    if (vg_vk_cache_get(g->vk, key, tag | 1u, sb) != VG_OK) {
        uint32_t n_scales = (nb + 1u) / 2u;
        uint32_t *scales = (uint32_t *)malloc((size_t)n_scales * 4u);
        if (!scales) return VG_E_NOMEM;
        const unsigned char *p = (const unsigned char *)wdata;
        for (uint32_t i = 0; i < nb; i += 2u) {
            uint16_t s0 = *(const uint16_t *)(p + (size_t)i * 34u);
            uint16_t s1 = (i + 1u < nb) ? *(const uint16_t *)(p + (size_t)(i + 1u) * 34u) : 0;
            scales[i / 2u] = (uint32_t)s0 | ((uint32_t)s1 << 16);
        }
        VG_Status st = vg_vk_cache_acquire(g->vk, key, tag | 1u, scales, (size_t)n_scales * 4u, sb);
        free(scales);
        if (st != VG_OK) return st;
    }
    return VG_OK;
}

/* Prepare one dense layer's weights into a VG_VKDenseLayer descriptor. The
 * actual recording/submit happens once for all layers in vg_vk_dense_forward.
 * Returns 1 on success, 0 to trigger fallback. */
static int dense_gpu_layer_forward(VG_ModelGraph *g, uint32_t layer, uint32_t pos,
                                   int first, int last, float *hidden, VG_VKDenseLayer *out) {
    uint32_t D = g->cfg.n_embd, FF = g->cfg.n_ff;
    uint32_t H = g->cfg.n_head, HKV = g->cfg.n_head_kv ? g->cfg.n_head_kv : 1;
    uint32_t HD = g->cfg.head_dim, Q = H * HD, KV = HKV * HD;
    char n[128];
    VG_TensorLease lq, lk, lv, lo, lgn, lun, ldn, lan, lfn;
    memset(&lq,0,sizeof(lq)); memset(&lk,0,sizeof(lk)); memset(&lv,0,sizeof(lv)); memset(&lo,0,sizeof(lo));
    memset(&lgn,0,sizeof(lgn)); memset(&lun,0,sizeof(lun)); memset(&ldn,0,sizeof(ldn));
    memset(&lan,0,sizeof(lan)); memset(&lfn,0,sizeof(lfn));
    VG_VKBuffer *qw=NULL,*qs=NULL,*kw=NULL,*ks=NULL,*vw=NULL,*vs=NULL,*ow=NULL,*os=NULL;
    VG_VKBuffer *gw=NULL,*gs=NULL,*uw=NULL,*us=NULL,*dw=NULL,*ds=NULL;
    VG_VKBuffer *anw=NULL,*fnw=NULL,*rf=NULL,*kc=NULL,*vc=NULL;
    int ok = 1;
    snprintf(n,sizeof(n),"blk.%u.attn_q.weight",layer);
    if (ok && (acquire_weight(g,n,&lq)!=VG_OK || vk_q8_prepare(g,lq.tensor,0u,lq.data,Q,D,&qw,&qs)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.attn_k.weight",layer);
    if (ok && (acquire_weight(g,n,&lk)!=VG_OK || vk_q8_prepare(g,lk.tensor,0u,lk.data,KV,D,&kw,&ks)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.attn_v.weight",layer);
    if (ok && (acquire_weight(g,n,&lv)!=VG_OK || vk_q8_prepare(g,lv.tensor,0u,lv.data,KV,D,&vw,&vs)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.attn_output.weight",layer);
    if (ok && (acquire_weight(g,n,&lo)!=VG_OK || vk_q8_prepare(g,lo.tensor,0x800u,lo.data,D,Q,&ow,&os)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.ffn_gate.weight",layer);
    if (ok && (acquire_weight(g,n,&lgn)!=VG_OK || vk_q8_prepare(g,lgn.tensor,0x800u,lgn.data,FF,D,&gw,&gs)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.ffn_up.weight",layer);
    if (ok && (acquire_weight(g,n,&lun)!=VG_OK || vk_q8_prepare(g,lun.tensor,0x800u,lun.data,FF,D,&uw,&us)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.ffn_down.weight",layer);
    if (ok && (acquire_weight(g,n,&ldn)!=VG_OK || vk_q8_prepare(g,ldn.tensor,0x800u,ldn.data,D,FF,&dw,&ds)!=VG_OK)) ok=0;
    snprintf(n,sizeof(n),"blk.%u.attn_norm.weight",layer);
    if (ok && (acquire_weight(g,n,&lan)!=VG_OK ||
        (vg_vk_cache_get(g->vk,lan.tensor,0x9100u,&anw)!=VG_OK &&
         vg_vk_cache_acquire(g->vk,lan.tensor,0x9100u,lan.data,(size_t)D*sizeof(float),&anw)!=VG_OK))) ok=0;
    snprintf(n,sizeof(n),"blk.%u.ffn_norm.weight",layer);
    if (ok && (acquire_weight(g,n,&lfn)!=VG_OK ||
        (vg_vk_cache_get(g->vk,lfn.tensor,0x9000u,&fnw)!=VG_OK &&
         vg_vk_cache_acquire(g->vk,lfn.tensor,0x9000u,lfn.data,(size_t)D*sizeof(float),&fnw)!=VG_OK))) ok=0;
    if (ok && g->rope_freqs && g->n_rope_freqs) {
        if (vg_vk_cache_get(g->vk,g->rope_freqs,0x2000u,&rf)!=VG_OK &&
            vg_vk_cache_acquire(g->vk,g->rope_freqs,0x2000u,g->rope_freqs,(size_t)g->n_rope_freqs*sizeof(float),&rf)!=VG_OK) rf=NULL;
    }
    if (ok) {
        /* Cap the per-layer device KV window: allocating the full metadata n_ctx
         * (e.g. 32k) for every layer can exceed VRAM and thrash the weight cache.
         * VG_GPU_CTX overrides; beyond the window we fall back for that token. */
        uint32_t cap = g->cfg.n_ctx;
        const char *ce = getenv("VG_GPU_CTX");
        uint32_t env_cap = (ce && atoi(ce) > 0) ? (uint32_t)atoi(ce) : 4096u;
        if (cap > env_cap) cap = env_cap;
        if (pos >= cap) { ok = 0; }
        else {
            uint32_t slot = 1000u + layer*2u;
            if (vg_vk_dev_scratch_acquire(g->vk,slot,(size_t)cap*KV*sizeof(float),&kc)!=VG_OK) ok=0;
            else if (vg_vk_dev_scratch_acquire(g->vk,slot+1u,(size_t)cap*KV*sizeof(float),&vc)!=VG_OK) ok=0;
        }
    }
    if (ok && !g->hidden_dev) { if (vg_vk_dev_scratch_acquire(g->vk,2000u,(size_t)D*sizeof(float),&g->hidden_dev)!=VG_OK) ok=0; }
    int rc = 0;
    if (ok && anw && fnw && kc && vc && g->hidden_dev) {
        VG_VKDenseLayer d;
        memset(&d,0,sizeof(d));
        d.attn_norm_w=anw; d.q_w=qw; d.q_s=qs; d.k_w=kw; d.k_s=ks; d.v_w=vw; d.v_s=vs;
        d.rope_freqs=rf; d.k_cache=kc; d.v_cache=vc; d.o_w=ow; d.o_s=os; d.ffn_norm_w=fnw;
        d.gate_w=gw; d.gate_s=gs; d.up_w=uw; d.up_s=us; d.down_w=dw; d.down_s=ds;
        d.hidden_dev=g->hidden_dev; d.hidden_in=hidden; d.hidden_out=hidden;
        d.first=first; d.last=last;
        d.D=D; d.H=H; d.HKV=HKV; d.HD=HD; d.KV_dim=KV; d.n_ctx=g->cfg.n_ctx; d.n_rot=g->cfg.n_rot;
        d.FF=FF; d.Q_dim=Q; d.pos=pos; d.has_freqs=(rf!=NULL); d.freq_base=g->cfg.freq_base; d.eps=g->cfg.rms_eps;
        *out = d; rc = 1;
    }
    release_weight(&lq); release_weight(&lk); release_weight(&lv); release_weight(&lo);
    release_weight(&lgn); release_weight(&lun); release_weight(&ldn); release_weight(&lan); release_weight(&lfn);
    return rc;
}
#endif

VG_Status vg_model_graph_moe_route(VG_ModelGraph *graph, uint32_t layer_idx,
                                   const float *hidden, float *gate_logits,
                                   VG_MoEState *state) {
    (void)layer_idx; (void)hidden;
    if (!graph || !state) return VG_E_INVALID;
    uint32_t n_expert = graph->cfg.n_expert > 0 ? graph->cfg.n_expert : 1;
    uint32_t n_used = graph->cfg.n_expert_used > 0 ? graph->cfg.n_expert_used : 1;
    state->n_expert = n_expert;
    state->n_expert_used = n_used;
    if (!state->expert_ids) {
        state->expert_ids = (uint32_t *)malloc(n_used * sizeof(uint32_t));
        if (!state->expert_ids) return VG_E_NOMEM;
    }
    if (!gate_logits) {
        for (uint32_t i = 0; i < n_used; ++i) state->expert_ids[i] = i % n_expert;
        return VG_OK;
    }
    uint32_t *order = (uint32_t *)malloc(n_expert * sizeof(uint32_t));
    if (!order) return VG_E_NOMEM;
    for (uint32_t i = 0; i < n_expert; ++i) order[i] = i;
    for (uint32_t i = 1; i < n_expert; ++i) {
        uint32_t j = i;
        while (j > 0 && gate_logits[order[j]] > gate_logits[order[j - 1]]) {
            uint32_t tmp = order[j]; order[j] = order[j - 1]; order[j - 1] = tmp;
            j--;
        }
    }
    for (uint32_t i = 0; i < n_used; ++i) state->expert_ids[i] = order[i];
    free(order);
    return VG_OK;
}

void vg_moe_state_free(VG_MoEState *state) {
    if (!state) return;
    free(state->expert_ids);
    state->expert_ids = NULL;
    state->n_expert = 0;
    state->n_expert_used = 0;
}

static void moe_dispatch_weights(VG_ModelGraph *g, uint32_t layer_idx,
                                 const float *hidden, uint32_t in_dim,
                                 float *output, uint32_t out_dim) __attribute__((unused));
static void moe_dispatch_weights(VG_ModelGraph *g, uint32_t layer_idx,
                                 const float *hidden, uint32_t in_dim,
                                 float *output, uint32_t out_dim) {
    VG_MoEState state; memset(&state, 0, sizeof(state));
    vg_model_graph_moe_route(g, layer_idx, hidden, NULL, &state);
    uint32_t D = in_dim; uint32_t FF = out_dim;
    char name[128];
    float *tmp_out = (float *)calloc(FF, sizeof(float));
    if (!tmp_out) { vg_moe_state_free(&state); return; }
    for (uint32_t e = 0; e < state.n_expert_used; ++e) {
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight.%u", layer_idx, state.expert_ids[e]);
        VG_TensorLease l; memset(&l, 0, sizeof(l));
        if (acquire_weight(g, name, &l) == VG_OK) {
            float *gate_out = (float *)calloc(FF, sizeof(float));
            float *up_out = (float *)calloc(FF, sizeof(float));
            if (gate_out && up_out) {
                #ifdef VG_HAS_VULKAN
                if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, hidden, gate_out, FF, D) != VG_OK)
                #endif
                vg_cpu_matmul(l.tensor->ggml_type, l.data, hidden, gate_out, FF, D);
                release_weight(&l);
                snprintf(name, sizeof(name), "blk.%u.ffn_up.weight.%u", layer_idx, state.expert_ids[e]);
                memset(&l, 0, sizeof(l));
                if (acquire_weight(g, name, &l) == VG_OK) {
                    #ifdef VG_HAS_VULKAN
                    if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, hidden, up_out, FF, D) != VG_OK)
                    #endif
                    vg_cpu_matmul(l.tensor->ggml_type, l.data, hidden, up_out, FF, D);
                    release_weight(&l);
                }
                for (uint32_t i = 0; i < FF; ++i) tmp_out[i] += gate_out[i] * up_out[i];
            }
            free(gate_out); free(up_out);
        }
    }
    vg_moe_state_free(&state);
    snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", layer_idx);
    VG_TensorLease l; memset(&l, 0, sizeof(l));
    if (acquire_weight(g, name, &l) == VG_OK) {
        #ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, tmp_out, output, D, FF) != VG_OK)
        #endif
        vg_cpu_matmul(l.tensor->ggml_type, l.data, tmp_out, output, D, FF);
        release_weight(&l);
    }
    free(tmp_out);
}
