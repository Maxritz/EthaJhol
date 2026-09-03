#ifndef VG_LORA_H
#define VG_LORA_H

#include "gguf.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_LoraAdapter VG_LoraAdapter;

/* Load a LoRA adapter from a GGUF file. Returns the rank and target layer info. */
VG_Status vg_lora_load(VG_GGUF *file, VG_LoraAdapter **out);
void vg_lora_free(VG_LoraAdapter *adapter);

/* Get the LoRA rank. */
uint32_t vg_lora_rank(const VG_LoraAdapter *adapter);

/* Apply LoRA to a weight in-place: weight += scale * (A @ B).
 * weight: [out_dim, in_dim], A: [rank, in_dim], B: [out_dim, rank].
 * ggml_type: quantization type of the original weight. */
VG_Status vg_lora_apply(const VG_LoraAdapter *adapter, const char *tensor_name,
                        float *weight, uint32_t out_dim, uint32_t in_dim, float scale);

/* Find a LoRA tensor by name (e.g. "blk.0.attn_q.loraA"). Returns data pointer and size, or NULL. */
const void *vg_lora_tensor(const VG_LoraAdapter *adapter, const char *name, size_t *out_size);

#ifdef __cplusplus
}
#endif
#endif
