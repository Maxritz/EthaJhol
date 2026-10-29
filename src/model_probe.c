#include "vg/model_probe.h"
#include "vg/quant.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Architectures whose tensor naming and graph shape match the llama-style evaluator. */
static const char *const k_archs[] = {
    "llama", "mistral", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
    "gemma", "gemma2", "gemma3", "smollm", "tinyllama",
    /* 1.58-bit ternary family: same block layout plus attn/ffn sub-norms */
    "bitnet",
    /* MoE families with the same block layout */
    "deepseek_moe", "mixtral"
};

static int arch_supported(const char *arch) {
    if (!arch) return 0;
    for (size_t i = 0; i < sizeof(k_archs) / sizeof(k_archs[0]); ++i)
        if (strcmp(arch, k_archs[i]) == 0) return 1;
    return 0;
}

static int meta_int(VG_GGUF *file, const char *key, int32_t *out) {
    const char *v = vg_gguf_meta(file, key);
    if (!v) return 0;
    const char *p = v;
    while (*p == '[' || *p == ' ' || *p == ',') ++p;
    char *end = NULL;
    long x = strtol(p, &end, 10);
    if (end == p) return 0;
    *out = (int32_t)x;
    return 1;
}

static void add_quant(VG_ModelProbe *p, uint32_t type) {
    const VG_QuantInfo *qi = vg_quant_info(type);
    const char *name = qi ? qi->name : "UNKNOWN";
    for (uint32_t i = 0; i < p->n_quant_kinds; ++i) if (p->quant_names[i] == name) return;
    if (p->n_quant_kinds < 12) p->quant_names[p->n_quant_kinds++] = name;
}

static void fail(VG_ModelProbe *p, VG_Status st, const char *tensor, const char *fmt, ...) {
    if (p->status != VG_OK) return; /* keep the first (root) cause */
    p->ok = 0;
    p->status = st;
    if (tensor) { snprintf(p->detail_tensor, sizeof(p->detail_tensor), "%s", tensor); }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->detail, sizeof(p->detail), fmt, ap);
    va_end(ap);
}

/* Validate one 1-D tensor (norm weights, biases): presence, shape and quant type. */
static int require_vector(VG_ModelProbe *p, VG_GGUF *file, const char *name, uint32_t dim) {
    const VG_GGUF_Tensor *t = vg_gguf_find_tensor(file, name);
    if (!t) {
        fail(p, VG_E_UNSUPPORTED, name, "required tensor '%s' is missing", name);
        return 0;
    }
    const VG_QuantInfo *qi = vg_quant_info(t->ggml_type);
    if (!qi || !qi->supported_scalar) {
        fail(p, VG_E_UNSUPPORTED, name, "tensor '%s' uses %s, which has no CPU dequantizer",
             name, qi ? qi->name : "an unknown type");
        return 0;
    }
    if (t->n_dims != 1 || (dim && t->dims[0] != dim)) {
        fail(p, VG_E_FORMAT, name, "tensor '%s' has shape %llu, expected %u",
             name, (unsigned long long)(t->n_dims ? t->dims[0] : 0), dim);
        return 0;
    }
    return 1;
}

/* Validate one weight tensor: presence, supported quant type and shape contract. */
static int require_tensor(VG_ModelProbe *p, VG_GGUF *file, const char *name,
                          uint32_t dim0, uint32_t dim1, int check_dims) {
    const VG_GGUF_Tensor *t = vg_gguf_find_tensor(file, name);
    if (!t) {
        fail(p, VG_E_UNSUPPORTED, name, "required tensor '%s' is missing", name);
        return 0;
    }
    const VG_QuantInfo *qi = vg_quant_info(t->ggml_type);
    if (!qi) {
        fail(p, VG_E_UNSUPPORTED, name, "tensor '%s' uses unknown ggml type %u", name, t->ggml_type);
        return 0;
    }
    add_quant(p, t->ggml_type);
    if (!qi->supported_scalar) {
        fail(p, VG_E_UNSUPPORTED, name, "tensor '%s' uses %s, which has no CPU dequantizer", name, qi->name);
        return 0;
    }
    if (check_dims) {
        if (t->n_dims < 2) {
            fail(p, VG_E_FORMAT, name, "tensor '%s' is not a matrix (%u dims)", name, t->n_dims);
            return 0;
        }
        if (dim0 && t->dims[0] != dim0) {
            fail(p, VG_E_FORMAT, name, "tensor '%s' has %llu input rows, expected %u",
                 name, (unsigned long long)t->dims[0], dim0);
            return 0;
        }
        if (dim1 && t->dims[1] != dim1) {
            fail(p, VG_E_FORMAT, name, "tensor '%s' has %llu outputs, expected %u",
                 name, (unsigned long long)t->dims[1], dim1);
            return 0;
        }
        if (t->dims[0] % qi->block_size) {
            fail(p, VG_E_FORMAT, name, "tensor '%s' row length %llu is not a multiple of %u (%s block)",
                 name, (unsigned long long)t->dims[0], qi->block_size, qi->name);
            return 0;
        }
    }
    return 1;
}

VG_Status vg_model_probe(VG_GGUF *file, VG_ModelProbe *out) {
    if (!file || !out) return VG_E_INVALID;
    memset(out, 0, sizeof(*out));
    out->status = VG_OK;
    out->ok = 1;

    const char *arch = vg_gguf_meta(file, "general.architecture");
    snprintf(out->arch, sizeof(out->arch), "%s", arch ? arch : "(none)");
    if (!arch) {
        fail(out, VG_E_FORMAT, NULL, "missing general.architecture metadata");
        return VG_OK;
    }

    char key[160];
    int32_t v = 0;
    if (meta_int(file, "llama.context_length", &v) || meta_int(file, "qwen2.context_length", &v)) out->n_ctx = (uint32_t)v;
    if (!out->n_ctx) {
        snprintf(key, sizeof(key), "%s.context_length", arch);
        if (meta_int(file, key, &v) && v > 0) out->n_ctx = (uint32_t)v;
    }
    if (!out->n_ctx) out->n_ctx = 2048;

    v = 0; snprintf(key, sizeof(key), "%s.block_count", arch); meta_int(file, key, &v); out->n_layer = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.embedding_length", arch); meta_int(file, key, &v); out->n_embd = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.attention.head_count", arch); meta_int(file, key, &v); out->n_head = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch); meta_int(file, key, &v); out->n_head_kv = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.feed_forward_length", arch); meta_int(file, key, &v); out->n_ff = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.rope.dimension_count", arch); meta_int(file, key, &v); out->n_rot = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.expert_count", arch); meta_int(file, key, &v); out->n_expert = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.expert_used_count", arch); meta_int(file, key, &v); out->n_expert_used = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.vocab_size", arch); meta_int(file, key, &v); out->n_vocab = (uint32_t)(v > 0 ? v : 0);
    v = 0; snprintf(key, sizeof(key), "%s.attention.key_length", arch); meta_int(file, key, &v); out->head_dim = (uint32_t)(v > 0 ? v : 0);

    /* Derive anything the metadata omitted from the embedding table. */
    const VG_GGUF_Tensor *te = vg_gguf_find_tensor(file, "token_embd.weight");
    if (!te) {
        fail(out, VG_E_UNSUPPORTED, "token_embd.weight", "required tensor 'token_embd.weight' is missing");
        return VG_OK;
    }
    if (!out->n_vocab && te->n_dims >= 2) out->n_vocab = (uint32_t)te->dims[1];
    if (!out->n_embd && te->n_dims >= 1) out->n_embd = (uint32_t)te->dims[0];
    if (!out->n_head) {
        fail(out, VG_E_FORMAT, NULL, "metadata '%s.attention.head_count' is missing", arch);
        return VG_OK;
    }
    if (!out->n_head_kv) out->n_head_kv = out->n_head;
    if (!out->head_dim && out->n_head) out->head_dim = out->n_embd / out->n_head;
    if (!out->n_rot) out->n_rot = out->head_dim;
    if (out->n_rot > out->head_dim) out->n_rot = out->head_dim;

    add_quant(out, te->ggml_type);
    if (!vg_quant_info(te->ggml_type) || !vg_quant_info(te->ggml_type)->supported_scalar) {
        const VG_QuantInfo *qi = vg_quant_info(te->ggml_type);
        fail(out, VG_E_UNSUPPORTED, "token_embd.weight", "token_embd.weight uses %s, which has no CPU dequantizer",
             qi ? qi->name : "an unknown type");
        return VG_OK;
    }

    const char *tok = vg_gguf_meta(file, "tokenizer.ggml.model");
    out->has_tokenizer = tok != NULL;
    out->has_bias = vg_gguf_find_tensor(file, "blk.0.attn_q.bias") != NULL;
    out->has_qk_norm = vg_gguf_find_tensor(file, "blk.0.attn_q_norm.weight") != NULL;
    out->has_sub_norm = vg_gguf_find_tensor(file, "blk.0.attn_sub_norm.weight") != NULL;
    out->is_moe = out->n_expert > 1 && vg_gguf_find_tensor(file, "blk.0.ffn_gate_inp.weight") != NULL;
    if (out->is_moe && !out->n_expert_used) out->n_expert_used = 1;

    if (!arch_supported(arch)) {
        fail(out, VG_E_UNSUPPORTED, NULL,
             "architecture '%s' is not implemented (supported: llama, mistral, qwen2/3, gemma, mixtral)", arch);
        return VG_OK;
    }
    if (!out->n_layer || !out->n_embd || !out->head_dim) {
        fail(out, VG_E_FORMAT, NULL, "model geometry is incomplete (n_layer=%u n_embd=%u head_dim=%u)",
             out->n_layer, out->n_embd, out->head_dim);
        return VG_OK;
    }

    /* Validate the output head and final norm. */
    char name[160];
    if (!require_tensor(out, file, "output_norm.weight", out->n_embd, 0, 0)) return VG_OK;
    {
        const VG_GGUF_Tensor *lm = vg_gguf_find_tensor(file, "output.weight");
        if (!lm) lm = te;
        const VG_QuantInfo *qi = vg_quant_info(lm->ggml_type);
        add_quant(out, lm->ggml_type);
        if (!qi || !qi->supported_scalar) {
            fail(out, VG_E_UNSUPPORTED, lm->name, "output head '%s' uses %s, which has no CPU dequantizer",
                 lm->name, qi ? qi->name : "an unknown type");
            return VG_OK;
        }
        if (lm->n_dims < 2 || lm->dims[0] != out->n_embd) {
            fail(out, VG_E_FORMAT, lm->name, "output head '%s' expects %u input rows", lm->name, out->n_embd);
            return VG_OK;
        }
    }

    /* Per-layer contract. The evaluator needs these exact tensors. */
    const uint32_t D = out->n_embd;
    const uint32_t qdim = out->n_head * out->head_dim;
    const uint32_t kdim = out->n_head_kv * out->head_dim;
    for (uint32_t l = 0; l < out->n_layer; ++l) {
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        if (!require_tensor(out, file, name, D, 0, 0)) return VG_OK;
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
        if (!require_tensor(out, file, name, D, qdim, 1)) return VG_OK;
        snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
        if (!require_tensor(out, file, name, D, kdim, 1)) return VG_OK;
        snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
        if (!require_tensor(out, file, name, D, kdim, 1)) return VG_OK;
        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
        if (!require_tensor(out, file, name, qdim, D, 1)) return VG_OK;
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        if (!require_tensor(out, file, name, D, 0, 0)) return VG_OK;

        /* BitNet b1.58 sub-norms: the attention context is normalised over n_embd and the
         * SwiGLU output over n_ff. Both are mandatory for that architecture. */
        if (out->has_sub_norm) {
            snprintf(name, sizeof(name), "blk.%u.attn_sub_norm.weight", l);
            if (!require_vector(out, file, name, qdim)) return VG_OK;
            snprintf(name, sizeof(name), "blk.%u.ffn_sub_norm.weight", l);
            if (!require_vector(out, file, name, out->n_ff)) return VG_OK;
        }

        if (out->is_moe) {
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", l);
            if (!require_tensor(out, file, name, D, out->n_expert, 1)) return VG_OK;
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", l);
            if (!require_tensor(out, file, name, D, 0, 1)) return VG_OK;
            snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", l);
            if (!require_tensor(out, file, name, D, 0, 1)) return VG_OK;
            snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", l);
            if (!require_tensor(out, file, name, out->n_ff, D, 1)) return VG_OK;
            const VG_GGUF_Tensor *ge = vg_gguf_find_tensor(file, "blk.0.ffn_gate_exps.weight");
            if (ge && ge->n_dims >= 3 && ge->dims[2] != out->n_expert) {
                fail(out, VG_E_FORMAT, ge->name,
                     "tensor '%s' stacks %llu experts but metadata declares %u",
                     ge->name, (unsigned long long)ge->dims[2], out->n_expert);
                return VG_OK;
            }
        } else {
            const int dense_gate = vg_gguf_find_tensor(file, "blk.0.ffn_gate.weight") != NULL;
            if (dense_gate) {
                snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
                if (!require_tensor(out, file, name, D, out->n_ff, 1)) return VG_OK;
            }
            snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
            if (!require_tensor(out, file, name, D, out->n_ff, 1)) return VG_OK;
            snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
            if (!require_tensor(out, file, name, out->n_ff, D, 1)) return VG_OK;
        }
    }

    /* Everything the evaluator will dispatch. */
    snprintf(out->required_ops, sizeof(out->required_ops),
             "matmul[%u types], rmsnorm, rope(n_rot=%u), softmax%s%s%s%s",
             out->n_quant_kinds, out->n_rot,
             out->has_qk_norm ? ", qk_norm" : "",
             out->has_bias ? ", bias_add" : "",
             out->has_sub_norm ? ", sub_norm" : "",
             out->is_moe ? ", moe_route, expert_matmul, expert_cache" : ", swiglu");
    return VG_OK;
}

void vg_model_probe_report(const VG_ModelProbe *p, char *buf, size_t cap) {
    if (!p || !buf || !cap) return;
    char quants[160] = {0};
    for (uint32_t i = 0; i < p->n_quant_kinds; ++i) {
        if (i) strncat(quants, ",", sizeof(quants) - strlen(quants) - 1);
        strncat(quants, p->quant_names[i], sizeof(quants) - strlen(quants) - 1);
    }
    snprintf(buf, cap,
             "arch: %s%s\n"
             "geometry: layers=%u embd=%u heads=%u kv_heads=%u head_dim=%u ff=%u rot=%u ctx=%u vocab=%u\n"
             "experts: total=%u used=%u%s\n"
             "quant types: %s\n"
             "features: tokenizer=%s bias=%s qk_norm=%s sub_norm=%s\n"
             "ops: %s\n"
             "result: %s%s%s",
             p->arch, p->ok ? "" : "  [UNSUPPORTED]",
             p->n_layer, p->n_embd, p->n_head, p->n_head_kv, p->head_dim, p->n_ff, p->n_rot, p->n_ctx, p->n_vocab,
             p->n_expert, p->n_expert_used, p->is_moe ? " (MoE)" : "",
             quants[0] ? quants : "(none)",
             p->has_tokenizer ? "yes" : "no", p->has_bias ? "yes" : "no", p->has_qk_norm ? "yes" : "no",
             p->has_sub_norm ? "yes" : "no",
             /* Rejected models return before op resolution, so say so instead of printing nothing. */
             p->required_ops[0] ? p->required_ops : "(not analysed: rejected above)",
             p->ok ? "SUPPORTED" : "NOT SUPPORTED",
             p->ok ? "" : ": ",
             p->ok ? "" : p->detail);
}
