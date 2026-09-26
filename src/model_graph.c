#include "vg/model_graph.h"
#include "vg/tensor_source.h"
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
#endif
};

static const VG_GGUF_Tensor *find_tensor(const VG_ModelGraph *g, const char *name) {
    return vg_gguf_find_tensor(g->file, name);
}

static VG_Status acquire_weight(VG_ModelGraph *g, const char *name, VG_TensorLease *lease);

#ifdef VG_HAS_VULKAN
static VG_Status vk_matmul(VG_ModelGraph *g, uint32_t ggml_type, const void *wdata, size_t wbytes,
                            const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
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
                        fprintf(stderr, "[dbg] rope_freqs has invalid values, using theta formula\n");
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

    fprintf(stderr, "[dbg] model config: arch=%s n_vocab=%u n_embd=%u n_head=%u n_head_kv=%u n_layer=%u n_ff=%u head_dim=%u n_rot=%u n_ctx=%u rms_eps=%.2e freq_base=%.1f\n",
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
        uint32_t KV_dim = g->cfg.n_head_kv * g->cfg.head_dim;
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
    if (layer < 2 && pos < 2) fprintf(stderr, "[dbg] attn_layer layer=%u D=%u H=%u HKV=%u pos=%u\n", layer, D, H, HKV, pos);

    memcpy(g->work_res, hidden, D * sizeof(float));

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
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
    q_proj = (float *)malloc(Q_dim * sizeof(float));
    if (q_proj) {
        if (layer < 2 && pos < 2) fprintf(stderr, "[dbg] layer %u Q out=%u in=%u type=%u HD=%u n_rot=%u\n", layer, Q_dim, D, l.tensor->ggml_type, HD, n_rot);
#ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, q_proj, Q_dim, D) == VG_OK) { } else
#endif
        { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, q_proj, Q_dim, D); }

        for (uint32_t h = 0; h < H; ++h) {
            float *hq = q_proj + h * HD;
            char qn_name[128]; snprintf(qn_name, sizeof(qn_name), "blk.%u.attn_q_norm.weight", layer);
            if (acquire_weight(g, qn_name, &nl) == VG_OK) {
                const float *qn = (const float *)nl.data;
                float inv = vec_rms(hq, HD, 1e-6f);
                for (uint32_t i = 0; i < HD; ++i) hq[i] = hq[i] * inv * qn[i];
                release_weight(&nl);
            }
            if (g->rope_freqs && g->n_rope_freqs >= n_rot) {
                uint32_t half = HD / 2;
                uint32_t iters = n_rot < half ? n_rot : half;
                for (uint32_t i = 0; i < iters; ++i) {
                    float freq = pos * g->rope_freqs[i];
                    float c = cosf(freq), s = sinf(freq);
                    float a = hq[i], b = hq[i + half];
                    hq[i] = a * c - b * s; hq[i + half] = a * s + b * c;
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
        k_full = (float *)malloc(KV_dim * sizeof(float));
        if (k_full) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, k_full, KV_dim, D) == VG_OK) { } else
#endif
            { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, k_full, KV_dim, D); }
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
                if (g->rope_freqs && g->n_rope_freqs >= n_rot) {
                    uint32_t half = k_hdim / 2;
                    uint32_t iters = n_rot < half ? n_rot : half;
                    for (uint32_t i = 0; i < iters; ++i) {
                        float freq = pos * g->rope_freqs[i];
                        float c = cosf(freq), s = sinf(freq);
                        float a = hk[i], b = hk[i + half];
                        hk[i] = a * c - b * s; hk[i + half] = a * s + b * c;
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
        v_full = (float *)malloc(KV_dim * sizeof(float));
        if (v_full) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, v_full, KV_dim, D) == VG_OK) { } else
#endif
            { vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, v_full, KV_dim, D); }
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

    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
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
    } else {
        for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i];
    }

    free(k_full); free(v_full);
}

static void ffn_layer(VG_ModelGraph *g, uint32_t layer, float *hidden) {
    uint32_t D = g->cfg.n_embd;
    uint32_t FF = g->cfg.n_ff;
    char name[128];
    VG_TensorLease l;

    /* Save original for residual (hidden may alias work_embd) */
    memcpy(g->work_res, hidden, D * sizeof(float));

    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    if (g->cfg.has_gate_up) {
        float *gate_out = (float *)malloc(FF * sizeof(float));
        float *up_out = (float *)malloc(FF * sizeof(float));
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", layer);
        if (gate_out && acquire_weight(g, name, &l) == VG_OK) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, gate_out, FF, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, gate_out, FF, D);
            }
            release_weight(&l);
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", layer);
        if (up_out && acquire_weight(g, name, &l) == VG_OK) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, up_out, FF, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, up_out, FF, D);
            }
            release_weight(&l);
        }
        if (gate_out && up_out) {
            vg_cpu_swiglu(g->work_ffn, gate_out, up_out, FF);
        }
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
    float pos_f = (float)kv_pos;

    VG_Status st = vg_model_graph_embed(g, input_token, g->work_embd);
    if (st != VG_OK) return st;

    for (uint32_t l = 0; l < g->cfg.n_layer; ++l) {
        if (g->source) {
            char prefetch_name[128];
            if (l + 1 < g->cfg.n_layer) {
                snprintf(prefetch_name, sizeof(prefetch_name), "blk.%u.attn_norm.weight", l + 1);
                vg_tensor_prefetch_after(g->source, prefetch_name, 4);
            }
        }
        attention_layer(g, l, g->work_embd, &pos_f);
        ffn_layer(g, l, g->work_embd);
    }

    char name[128]; VG_TensorLease wl;
    snprintf(name, sizeof(name), "output_norm.weight");
    if (acquire_weight(g, name, &wl) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, g->work_embd, (const float *)wl.data, D, g->cfg.rms_eps);
        release_weight(&wl);
    }

    /* Compute logits via output projection (lm_head). */
    const VG_GGUF_Tensor *lt = find_tensor(g, "output.weight");
    const char *lm_head_name = "output.weight";
    if (!lt) { lt = find_tensor(g, "token_embd.weight"); lm_head_name = "token_embd.weight"; }
    if (lt) {
        uint32_t ggml_type = lt->ggml_type;
        const VG_QuantInfo *qi = vg_quant_info(ggml_type);
        if (ggml_type == 0) {
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

    if (logits_out) memcpy(logits_out, g->logits, g->cfg.n_vocab * sizeof(float));
    return VG_OK;
}

const float *vg_model_graph_logits(const VG_ModelGraph *g) { return g ? g->logits : NULL; }

#ifdef VG_HAS_VULKAN
void vg_model_graph_set_vulkan(VG_ModelGraph *g, VG_VK *vk) { if (g) g->vk = vk; }

static VG_Status vk_matmul(VG_ModelGraph *g, uint32_t ggml_type, const void *wdata, size_t wbytes,
                            const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    if (!g->vk) return VG_E_UNSUPPORTED;
    VG_VKBuffer *wb = NULL, *sb = NULL, *xb = NULL;
    VG_Status st;

    /* Upload weights */
    st = vg_vk_buffer_upload(g->vk, wdata, wbytes, &wb);
    if (st != VG_OK) return st;

    /* Create scales buffer (identity for non-quantized, or extract from quantized blocks) */
    if (ggml_type == 0 || ggml_type == 1) {
        /* F32/F16: no per-block scales, upload dummy */
        float one = 1.0f;
        st = vg_vk_buffer_upload(g->vk, &one, sizeof(float), &sb);
    } else if (ggml_type == 8) {
        /* Q8_0: 34-byte blocks = uint16_t d (fp16) + int8_t qs[32] */
        uint32_t bs = 32;
        uint32_t n_blocks = out_dim * (in_dim / bs);
        /* Pack fp16 scales: 2 per uint32 = n_blocks/2 uint32s, plus 1 if odd */
        uint32_t n_scales = (n_blocks + 1) / 2;
        uint32_t *scales = (uint32_t *)malloc(n_scales * sizeof(uint32_t));
        if (!scales) { vg_vk_buffer_release(g->vk, wb); return VG_E_NOMEM; }
        const unsigned char *p = (const unsigned char *)wdata;
        for (uint32_t i = 0; i < n_blocks; i += 2) {
            uint16_t s0 = *(const uint16_t *)(p + (uint64_t)i * 34);
            uint16_t s1 = (i + 1 < n_blocks) ? *(const uint16_t *)(p + (uint64_t)(i + 1) * 34) : 0;
            scales[i / 2] = (uint32_t)s0 | ((uint32_t)s1 << 16);
        }
        st = vg_vk_buffer_upload(g->vk, scales, n_scales * sizeof(uint32_t), &sb);
        free(scales);
    } else {
        /* Q4_0 and others: not yet supported on GPU */
        vg_vk_buffer_release(g->vk, wb);
        return VG_E_UNSUPPORTED;
    }
    if (st != VG_OK) { vg_vk_buffer_release(g->vk, wb); return st; }

    if (ggml_type == 0) {
        st = vg_vk_matvec_f32(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
    } else if (ggml_type == 1) {
        st = vg_vk_matvec_f16(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
    } else if (ggml_type == 2) {
        st = vg_vk_matvec_q4_0(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
    } else if (ggml_type == 8) {
        st = vg_vk_matvec_q8_0(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
    } else {
        st = VG_E_UNSUPPORTED;
    }

    vg_vk_buffer_release(g->vk, wb);
    vg_vk_buffer_release(g->vk, sb);
    return st;
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
