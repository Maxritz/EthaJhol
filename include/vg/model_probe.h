#ifndef VG_MODEL_PROBE_H
#define VG_MODEL_PROBE_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Result of validating a GGUF file against what this engine can actually execute.
 * `ok` is 1 only when every required operator and tensor layout is implemented, in which
 * case `status` is VG_OK. Otherwise `status` is the precise failure and `detail` names the
 * exact missing/unsupported piece (tensor name, quant type or operator). */
typedef struct VG_ModelProbe {
    int ok;
    VG_Status status;
    char arch[48];
    char detail[256];
    char detail_tensor[96];

    uint32_t n_vocab;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t n_rot;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ctx;
    int has_tokenizer;
    int has_bias;
    int has_qk_norm;
    int has_sub_norm;   /* BitNet b1.58 attn/ffn sub-norm weights present */
    int has_post_norm;  /* gemma2/3 sandwich norms (attn_post_norm, ffn_post_norm) */
    int has_shared_expert; /* MoE with an always-on shared expert */
    float logit_softcap;   /* non-zero when the model caps final logits */
    int is_moe;

    /* Distinct quant types covering the required weight tensors. */
    uint32_t n_quant_kinds;
    const char *quant_names[12];
    /* Operator set the graph will execute for this model. */
    char required_ops[320];
} VG_ModelProbe;

/* Inspect an opened GGUF file. Never fails for a readable file: problems are reported in
 * the struct so callers can print a full diagnosis. */
VG_Status vg_model_probe(VG_GGUF *file, VG_ModelProbe *out);

/* Human-readable multi-line report (writes at most `cap` bytes including the terminator). */
void vg_model_probe_report(const VG_ModelProbe *probe, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
