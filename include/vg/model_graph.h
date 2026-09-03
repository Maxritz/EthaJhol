#ifndef VG_MODEL_GRAPH_H
#define VG_MODEL_GRAPH_H

#include "gguf.h"
#include "tensor_source.h"
#include "cpu_ops.h"

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
    int is_swa;
    uint32_t n_ctx;
} VG_ModelConfig;

VG_Status vg_model_graph_load(VG_GGUF *file, VG_TensorSource *source, VG_ModelGraph **out);
void vg_model_graph_free(VG_ModelGraph *graph);
const VG_ModelConfig *vg_model_graph_config(const VG_ModelGraph *graph);

VG_Status vg_model_graph_decode(VG_ModelGraph *graph, int32_t input_token,
                                int32_t kv_pos, float *logits_out);

VG_Status vg_model_graph_embed(VG_ModelGraph *graph, int32_t token, float *embd_out);
const float *vg_model_graph_logits(const VG_ModelGraph *graph);
void vg_model_graph_reset(VG_ModelGraph *graph);

#ifdef VG_HAS_VULKAN
void vg_model_graph_set_vulkan(VG_ModelGraph *graph, VG_VK *vk);
#endif

#ifdef __cplusplus
}
#endif
#endif
