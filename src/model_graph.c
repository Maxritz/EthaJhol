#include "vg/model_graph.h"
#include "vg/tensor_source.h"
#include "vg/quant.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

struct VG_ModelGraph {
    VG_GGUF *file;
    VG_TensorSource *source;
    VG_ModelConfig cfg;
    float *work_embd;
    float *work_ffn;
    float *work_qkv;
    float *work_att_out;
    float *work_head;
    float *work_res;
    float *logits;
    float **kv_k;
    float **kv_v;
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
    if (g->source) return vg_tensor_acquire(g->source, name, VG_TIER_HOST, lease);
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
    *out = (int32_t)strtol(v, NULL, 10);
    return VG_OK;
}

static VG_Status read_metadata_float(const VG_ModelGraph *g, const char *key, float *out) {
    const char *v = vg_gguf_meta(g->file, key);
    if (!v) return VG_E_INVALID;
    *out = strtof(v, NULL);
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
    if (g->cfg.head_dim == 0) g->cfg.head_dim = g->cfg.n_embd / g->cfg.n_head;
    if (g->cfg.n_rot == 0) g->cfg.n_rot = g->cfg.head_dim;
    READ_FLOAT(rms_eps, attention.layer_norm_rms_epsilon);
    if (g->cfg.rms_eps == 0.0f) g->cfg.rms_eps = 1e-5f;
    READ_FLOAT(freq_base, rope.freq_base);
    if (g->cfg.freq_base == 0.0f) g->cfg.freq_base = 10000.0f;

    {
        int32_t n_ctx = 0;
        snprintf(key, sizeof(key), "%s.context_length", arch);
        read_metadata_int(g, key, &n_ctx);
        g->cfg.n_ctx = (n_ctx > 0) ? (uint32_t)n_ctx : 2048;
    }

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
    g->work_embd = (float *)malloc(embd_bytes);
    g->work_ffn = (float *)malloc(ff_bytes);
    g->work_qkv = (float *)malloc(embd_bytes);
    g->work_att_out = (float *)malloc(embd_bytes);
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
        g->kv_k = (float **)calloc(g->cfg.n_layer, sizeof(float *));
        g->kv_v = (float **)calloc(g->cfg.n_layer, sizeof(float *));
        if (!g->kv_k || !g->kv_v) {
            free(g->kv_k); free(g->kv_v);
            free(g->work_embd); free(g->work_ffn); free(g->work_qkv);
            free(g->work_att_out); free(g->work_head); free(g->work_res);
            free(g->logits);
            free(g); return VG_E_NOMEM;
        }
        for (uint32_t i = 0; i < g->cfg.n_layer; ++i) {
            g->kv_k[i] = (float *)calloc(g->cfg.n_ctx * KV_dim, sizeof(float));
            g->kv_v[i] = (float *)calloc(g->cfg.n_ctx * KV_dim, sizeof(float));
            if (!g->kv_k[i] || !g->kv_v[i]) {
                for (uint32_t j = 0; j <= i; ++j) { free(g->kv_k[j]); free(g->kv_v[j]); }
                free(g->kv_k); free(g->kv_v);
                free(g->work_embd); free(g->work_ffn); free(g->work_qkv);
                free(g->work_att_out); free(g->work_head); free(g->work_res);
                free(g->logits);
                free(g); return VG_E_NOMEM;
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
    free(g->logits);
    if (g->kv_k) { for (uint32_t i = 0; i < g->cfg.n_layer; ++i) free(g->kv_k[i]); free(g->kv_k); }
    if (g->kv_v) { for (uint32_t i = 0; i < g->cfg.n_layer; ++i) free(g->kv_v[i]); free(g->kv_v); }
    free(g);
}

void vg_model_graph_reset(VG_ModelGraph *g) {
    if (!g) return;
    uint32_t KV_dim = g->cfg.n_head_kv * g->cfg.head_dim;
    for (uint32_t i = 0; i < g->cfg.n_layer; ++i) {
        memset(g->kv_k[i], 0, g->cfg.n_ctx * KV_dim * sizeof(float));
        memset(g->kv_v[i], 0, g->cfg.n_ctx * KV_dim * sizeof(float));
    }
}

const VG_ModelConfig *vg_model_graph_config(const VG_ModelGraph *g) { return g ? &g->cfg : NULL; }

VG_Status vg_model_graph_embed(VG_ModelGraph *g, int32_t token, float *embd_out) {
    if (!g || token < 0 || (uint32_t)token >= g->cfg.n_vocab) return VG_E_INVALID;
    char name[128]; snprintf(name, sizeof(name), "token_embd.weight");
    VG_TensorLease l; VG_Status st = acquire_weight(g, name, &l);
    if (st != VG_OK) return st;
    uint32_t embd = g->cfg.n_embd;
    uint32_t ggml_type = l.tensor->ggml_type;
    if (ggml_type == 0) {
        const float *row = (const float *)((const char *)l.data + (uint64_t)token * embd * sizeof(float));
        memcpy(embd_out, row, embd * sizeof(float));
    } else {
        const VG_QuantInfo *qi = vg_quant_info(ggml_type);
        if (!qi || qi->block_size == 0) { release_weight(&l); return VG_E_UNSUPPORTED; }
        uint32_t blocks_per_row = embd / qi->block_size;
        uint64_t row_bytes = (uint64_t)blocks_per_row * qi->bytes_per_block;
        const void *row_data = (const char *)l.data + (uint64_t)token * row_bytes;
        st = vg_dequantize_row(ggml_type, row_data, row_bytes, embd_out, embd);
        if (st != VG_OK) { release_weight(&l); return st; }
    }
    release_weight(&l);
    return VG_OK;
}

static void attention_layer(VG_ModelGraph *g, uint32_t layer, float *hidden, float *cur_pos_f) {
    uint32_t D = g->cfg.n_embd;
    uint32_t H = g->cfg.n_head;
    uint32_t HKV = g->cfg.n_head_kv;
    uint32_t HD = g->cfg.head_dim;
    uint32_t KV_dim = HKV * HD;
    uint32_t n_rot = g->cfg.n_rot;
    uint32_t pos = (uint32_t)*cur_pos_f;
    char name[128];
    VG_TensorLease l;

    /* Save original hidden for residual (hidden may alias work_embd) */
    memcpy(g->work_res, hidden, D * sizeof(float));

    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        vg_cpu_rmsnorm(g->work_embd, hidden, (const float *)l.data, D, g->cfg.rms_eps);
        release_weight(&l);
    } else {
        memcpy(g->work_embd, hidden, D * sizeof(float));
    }

    /* Q projection */
    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        float *q_proj = (float *)malloc(D * sizeof(float));
        if (q_proj) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, q_proj, D, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, q_proj, D, D);
            }
            for (uint32_t h = 0; h < H; ++h) {
                float *hq = q_proj + h * HD;
                vg_cpu_rope(hq, hq, HD, n_rot, 1.0f, g->cfg.freq_base, (int32_t)pos);
            }
            memcpy(g->work_qkv, q_proj, D * sizeof(float));
            free(q_proj);
        }
        release_weight(&l);
    }
    if (g->cfg.has_biases) {
        snprintf(name, sizeof(name), "blk.%u.attn_q.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < D; ++i) g->work_qkv[i] += bias[i];
            release_weight(&l);
        }
    }

    /* K projection */
    float *k_full = (float *)malloc(KV_dim * sizeof(float));
    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", layer);
    if (k_full && acquire_weight(g, name, &l) == VG_OK) {
#ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, k_full, KV_dim, D) == VG_OK) {
            /* Vulkan dispatch succeeded */
        } else
#endif
        {
            vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, k_full, KV_dim, D);
        }
        release_weight(&l);
        for (uint32_t h = 0; h < HKV; ++h) {
            float *hk = k_full + h * HD;
            vg_cpu_rope(hk, hk, HD, n_rot, 1.0f, g->cfg.freq_base, (int32_t)pos);
        }
    }
    if (g->cfg.has_biases && k_full) {
        snprintf(name, sizeof(name), "blk.%u.attn_k.bias", layer);
        if (acquire_weight(g, name, &l) == VG_OK) {
            const float *bias = (const float *)l.data;
            for (uint32_t i = 0; i < KV_dim; ++i) k_full[i] += bias[i];
            release_weight(&l);
        }
    }

    /* V projection */
    float *v_full = (float *)malloc(KV_dim * sizeof(float));
    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", layer);
    if (v_full && acquire_weight(g, name, &l) == VG_OK) {
#ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_embd, v_full, KV_dim, D) == VG_OK) {
            /* Vulkan dispatch succeeded */
        } else
#endif
        {
            vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_embd, v_full, KV_dim, D);
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

    /* Store K/V in cache */
    if (k_full && v_full && pos < g->cfg.n_ctx) {
        memcpy(g->kv_k[layer] + pos * KV_dim, k_full, KV_dim * sizeof(float));
        memcpy(g->kv_v[layer] + pos * KV_dim, v_full, KV_dim * sizeof(float));
    }

    /* Attention: softmax(Q @ K_cache^T / sqrt(HD)) @ V_cache */
    memset(g->work_att_out, 0, D * sizeof(float));
    if (k_full && v_full) {
        float scale = 1.0f / sqrtf((float)HD);
        uint32_t n_past = pos + 1;

        for (uint32_t h = 0; h < H; ++h) {
            uint32_t kv_h = h / (H / HKV);
            float *hq = g->work_qkv + h * HD;

            float *scores = (float *)malloc(n_past * sizeof(float));
            if (!scores) continue;

            for (uint32_t t = 0; t < n_past; ++t) {
                float *hk = g->kv_k[layer] + t * KV_dim + kv_h * HD;
                float dot = 0.0f;
                for (uint32_t d = 0; d < HD; ++d) dot += hq[d] * hk[d];
                scores[t] = dot * scale;
            }

            /* Softmax */
            float mx = scores[0];
            for (uint32_t t = 1; t < n_past; ++t) if (scores[t] > mx) mx = scores[t];
            float s = 0.0f;
            for (uint32_t t = 0; t < n_past; ++t) { scores[t] = expf(scores[t] - mx); s += scores[t]; }
            for (uint32_t t = 0; t < n_past; ++t) scores[t] /= s;

            /* Weighted sum of V */
            for (uint32_t t = 0; t < n_past; ++t) {
                float *hv = g->kv_v[layer] + t * KV_dim + kv_h * HD;
                float w = scores[t];
                for (uint32_t d = 0; d < HD; ++d) {
                    g->work_att_out[h * HD + d] += w * hv[d];
                }
            }
            free(scores);
        }
    }

    /* Output projection */
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", layer);
    if (acquire_weight(g, name, &l) == VG_OK) {
        float *proj_out = (float *)malloc(D * sizeof(float));
        if (proj_out) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, l.tensor->ggml_type, l.data, l.size, g->work_att_out, proj_out, D, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(l.tensor->ggml_type, l.data, g->work_att_out, proj_out, D, D);
            }
            for (uint32_t i = 0; i < D; ++i) hidden[i] = g->work_res[i] + proj_out[i];
            free(proj_out);
        }
        release_weight(&l);
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

    snprintf(name, sizeof(name), "output.weight");
    st = acquire_weight(g, name, &wl);
    if (st == VG_OK) {
#ifdef VG_HAS_VULKAN
        if (g->vk && vk_matmul(g, wl.tensor->ggml_type, wl.data, wl.size, g->work_embd, g->logits, g->cfg.n_vocab, D) == VG_OK) {
            /* Vulkan dispatch succeeded */
        } else
#endif
        {
            vg_cpu_matmul(wl.tensor->ggml_type, wl.data, g->work_embd, g->logits, g->cfg.n_vocab, D);
        }
        release_weight(&wl);
    } else {
        snprintf(name, sizeof(name), "token_embd.weight");
        st = acquire_weight(g, name, &wl);
        if (st == VG_OK) {
#ifdef VG_HAS_VULKAN
            if (g->vk && vk_matmul(g, wl.tensor->ggml_type, wl.data, wl.size, g->work_embd, g->logits, g->cfg.n_vocab, D) == VG_OK) {
                /* Vulkan dispatch succeeded */
            } else
#endif
            {
                vg_cpu_matmul(wl.tensor->ggml_type, wl.data, g->work_embd, g->logits, g->cfg.n_vocab, D);
            }
            release_weight(&wl);
        } else {
            return VG_E_INVALID;
        }
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

    /* Dispatch */
    if (ggml_type == 0) {
        st = vg_vk_matvec_f32(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
    } else if (ggml_type == 1) {
        st = vg_vk_matvec_f16(g->vk, wb, sb, x, y, 1, in_dim, out_dim);
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
