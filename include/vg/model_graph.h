#ifndef VG_MODEL_GRAPH_H
#define VG_MODEL_GRAPH_H

#include "gguf.h"
#include "tensor_source.h"
#include "cpu_ops.h"
#include "lora.h"

#ifdef VG_HAS_VULKAN
#include "vulkan_backend.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_ModelGraph VG_ModelGraph;

typedef struct VG_ModelConfig {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_ff;
    uint32_t head_dim;
    uint32_t n_rot;
    float rms_eps;
    float freq_base;
    int32_t bos_id;
    int32_t eos_id;
    int has_biases;
    int has_gate_up;
    int has_sub_norm;   /* BitNet b1.58 attn/ffn sub-norm RMSNorm weights */
    int is_swa;
    uint32_t n_ctx;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t sliding_window;
} VG_ModelConfig;

VG_Status vg_model_graph_load(VG_GGUF *file, VG_TensorSource *source, VG_ModelGraph **out);
void vg_model_graph_free(VG_ModelGraph *graph);
const VG_ModelConfig *vg_model_graph_config(const VG_ModelGraph *graph);

VG_Status vg_model_graph_decode(VG_ModelGraph *graph, int32_t input_token,
                                int32_t kv_pos, float *logits_out);

VG_Status vg_model_graph_embed(VG_ModelGraph *graph, int32_t token, float *embd_out);
const float *vg_model_graph_logits(const VG_ModelGraph *graph);
void vg_model_graph_reset(VG_ModelGraph *graph);

typedef struct VG_MoEState {
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t *expert_ids;
} VG_MoEState;

/* Top-k expert routing. Pass gate_logits when the caller already computed the router
 * output; pass hidden (and NULL logits) to let the graph evaluate the router projection.
 * Both NULL selects the first n_expert_used experts (used by capability checks). */
VG_Status vg_model_graph_moe_route(VG_ModelGraph *graph, uint32_t layer,
                                   const float *hidden, float *gate_logits,
                                   VG_MoEState *state);
void vg_moe_state_free(VG_MoEState *state);

/* Attach a LoRA adapter. The delta y += scale * B * (A * x) is applied inside every
 * weight matmul the adapter targets, so quantized weights never need requantizing.
 * Pass NULL to detach. `scale` applies on top of each target's own `.scaling` value. */
VG_Status vg_model_graph_set_lora(VG_ModelGraph *graph, const VG_LoraAdapter *adapter, float scale);

/* Expert weight cache statistics (hits avoid re-resolving a stacked expert slice). */
void vg_model_graph_expert_stats(VG_ModelGraph *graph, uint64_t *hits, uint64_t *misses,
                                 uint64_t *evictions);
VG_Status vg_model_graph_set_expert_cache(VG_ModelGraph *graph, uint32_t capacity);

#ifdef VG_HAS_VULKAN
void vg_model_graph_set_vulkan(VG_ModelGraph *graph, VG_VK *vk);
#endif

#ifdef __cplusplus
}
#endif
#endif
