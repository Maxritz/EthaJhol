#ifndef VG_LORA_H
#define VG_LORA_H

#include "gguf.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_LoraAdapter VG_LoraAdapter;

/* Load a LoRA adapter from a GGUF file.

 * Both tensor naming conventions published in the wild are accepted:
 *   "blk.0.attn_q.weight.lora_a" / ".lora_b"   (current converter output)
 *   "blk.0.attn_q.loraA"           / ".loraB"  (older adapters)
 * A per-target ".scaling" scalar tensor is honoured when present.

 * A is stored in ggml order as {in_dim, rank} and B as {rank, out_dim}; both are
 * converted to float32 once at load time so the decode path never re-dequantizes. */
VG_Status vg_lora_load(VG_GGUF *file, VG_LoraAdapter **out);
void vg_lora_free(VG_LoraAdapter *adapter);

/* Get the LoRA rank. */
uint32_t vg_lora_rank(const VG_LoraAdapter *adapter);

/* Number of adapter targets and the weight name of target i (e.g. "blk.0.attn_q.weight"). */
size_t vg_lora_target_count(const VG_LoraAdapter *adapter);
const char *vg_lora_target_name(const VG_LoraAdapter *adapter, size_t index);

/* Apply the adapter to a matmul result: y += scale * scaling * B * (A * x).

 * This is the exact LoRA forward pass, so the quantized weight stays untouched and no
 * requantization error is introduced. Returns 1 when a delta was applied, 0 when the
 * adapter has no target for `weight_name`, and -1 on invalid arguments. */
int vg_lora_apply_delta(const VG_LoraAdapter *adapter, const char *weight_name,
                        const float *x, float *y, uint32_t out_dim, uint32_t in_dim,
                        float scale);

/* Materialize the delta into a dense F32 weight: weight[i,j] += scale * scaling * (B @ A)[i,j].
 * Only for callers that keep float weights; the engine uses vg_lora_apply_delta instead. */
VG_Status vg_lora_apply(const VG_LoraAdapter *adapter, const char *tensor_name,
                        float *weight, uint32_t out_dim, uint32_t in_dim, float scale);

/* Rank of the adapter target matching a weight tensor name (0 if absent). */
uint32_t vg_lora_target_rank(const VG_LoraAdapter *adapter, const char *weight_name);

#ifdef __cplusplus
}
#endif
#endif
