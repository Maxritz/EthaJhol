#ifndef VG_VULKAN_BACKEND_H
#define VG_VULKAN_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_VK VG_VK;
typedef struct VG_VKBuffer VG_VKBuffer;
typedef struct VG_VKInfo {
    char name[256];
    uint32_t api_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint64_t device_local_bytes;
    uint64_t device_local_budget;
    int discrete;
} VG_VKInfo;

typedef struct VG_VKConfig {
    uint32_t device_index;
    const char *shader_dir;
    uint64_t memory_budget_bytes;
    int prefer_discrete;
} VG_VKConfig;

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out);
void vg_vk_close(VG_VK *vk);
VG_Status vg_vk_info(const VG_VK *vk, VG_VKInfo *out);
uint64_t vg_vk_memory_used(const VG_VK *vk);
VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out);
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes);
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer);

/* Persistent device-local cache (COMPASS method 2.1/2.4): uploads once keyed by
 * (key, tag), lives in VRAM for the process lifetime, never re-uploaded. */
VG_Status vg_vk_cache_acquire(VG_VK *vk, const void *key, uint32_t tag,
                              const void *data, size_t bytes, VG_VKBuffer **out);
VG_Status vg_vk_cache_get(VG_VK *vk, const void *key, uint32_t tag, VG_VKBuffer **out);
/* Host-visible scratch buffer reused across calls (grow-only, keyed by slot index). */
VG_Status vg_vk_scratch_acquire(VG_VK *vk, uint32_t slot, size_t bytes, VG_VKBuffer **out);
/* Persistent DEVICE-LOCAL scratch (grow-only, keyed by slot index). Used for
 * device-resident state the GPU reads/writes directly (e.g. per-layer KV cache). */
VG_Status vg_vk_dev_scratch_acquire(VG_VK *vk, uint32_t slot, size_t bytes, VG_VKBuffer **out);

/* Batched matvecs: n independent ops recorded into ONE command buffer and
 * submitted once (COMPASS method 3.1) with per-dispatch barriers (method 4.3). */
typedef struct VG_VKMatvecReq {
    const VG_VKBuffer *weights; /* repacked, aligned */
    const VG_VKBuffer *scales;
    const float *x;
    float *y;
    uint32_t in_dim;
    uint32_t out_dim;
} VG_VKMatvecReq;
VG_Status vg_vk_matvec_q8_0_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n);
VG_Status vg_vk_matvec_f32_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n);
VG_Status vg_vk_matvec_f16_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n);

/* Matrix-vector products with different quantization types.
 * weights/scales are device buffers; x is host input; y is host output. */
VG_Status vg_vk_matvec_i8(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales,
                          const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);
VG_Status vg_vk_matvec_f16(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales,
                          const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);
VG_Status vg_vk_matvec_f32(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales,
                          const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);
VG_Status vg_vk_matvec_q4_0(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales,
                           const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);
VG_Status vg_vk_matvec_q8_0(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales,
                           const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);
/* Q4_K matvec reading the raw on-disk GGML blocks directly (no repack).
 * weights is a device buffer holding the tensor's native 144-byte blocks. */
VG_Status vg_vk_matvec_q4_k(VG_VK *vk, const VG_VKBuffer *weights,
                           const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output);

/* RMSNorm: out[i] = input[i] / sqrt(mean(input^2) + eps) * weight[i].
 * input and weight are device buffers; out is host output (dim floats). */
VG_Status vg_vk_rmsnorm(VG_VK *vk, const VG_VKBuffer *input, const VG_VKBuffer *weight,
                        float *out, uint32_t dim, float eps);

/* Softmax in-place on device buffer of dim floats. */
VG_Status vg_vk_softmax(VG_VK *vk, VG_VKBuffer *data, uint32_t dim);

/* RoPE in-place on device buffer. data contains q and k interleaved per head.
 * head_dim = dimension per head, n_rot = rotation dimension, pos = sequence position. */
VG_Status vg_vk_rope(VG_VK *vk, VG_VKBuffer *data, uint32_t head_dim, uint32_t n_rot,
                     float freq_scale, float theta_base, uint32_t pos);

/* SwiGLU: out[i] = swish(gate[i]) * up[i]. All device buffers. */
VG_Status vg_vk_swiglu(VG_VK *vk, const VG_VKBuffer *gate, const VG_VKBuffer *up,
                       VG_VKBuffer *out, uint32_t dim);


VG_Status vg_vk_ffn_fused(VG_VK *vk,
                          const VG_VKBuffer *gate_w, const VG_VKBuffer *gate_s,
                          const VG_VKBuffer *up_w,   const VG_VKBuffer *up_s,
                          const VG_VKBuffer *down_w, const VG_VKBuffer *down_s,
                          const float *x, float *y, uint32_t D, uint32_t FF);

/* Fused layer tail: o_proj(attn) + resid -> rmsnorm(norm_w) -> gate/up -> swiglu
 * -> on-device quant -> down -> resid add, in ONE command buffer. Writes the
 * layer output (post-FFN residual) to y. attn/resid are host vectors (D). */
VG_Status vg_vk_layer_tail(VG_VK *vk,
                           const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
                           const float *attn, const float *resid,
                           const VG_VKBuffer *norm_w,
                           const VG_VKBuffer *gate_w, const VG_VKBuffer *gate_s,
                           const VG_VKBuffer *up_w,   const VG_VKBuffer *up_s,
                           const VG_VKBuffer *down_w, const VG_VKBuffer *down_s,
                           float *y, uint32_t D, uint32_t FF, float eps);

/* Fused FFN for TWO MoE experts in ONE command buffer: x staged/quantized
 * once, then gate/up/swiglu/quant/down per expert. y0/y1 are host D-vectors. */
VG_Status vg_vk_ffn_fused_x2(VG_VK *vk,
    const VG_VKBuffer *gw0, const VG_VKBuffer *gs0, const VG_VKBuffer *uw0, const VG_VKBuffer *us0, const VG_VKBuffer *dw0, const VG_VKBuffer *ds0,
    const VG_VKBuffer *gw1, const VG_VKBuffer *gs1, const VG_VKBuffer *uw1, const VG_VKBuffer *us1, const VG_VKBuffer *dw1, const VG_VKBuffer *ds1,
    const float *x, float *y0, float *y1, uint32_t D, uint32_t FF);

/* Fused MoE head: o_proj(attn) + resid -> rmsnorm(ffn_norm) -> on-device quant
 * -> router, in ONE command buffer. Writes the post-attention hidden
 * (h1 = resid + o_proj*attn, D floats) to hidden_out and the router logits
 * (NE floats) to logits_out. This collapses the separate o_proj submit and the
 * router submit into one, so the caller only needs one more submit for experts. */
VG_Status vg_vk_moe_head(VG_VK *vk,
    const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
    const float *attn, const float *resid,
    const VG_VKBuffer *norm_w,
    const VG_VKBuffer *route_w, const VG_VKBuffer *route_s,
    float *hidden_out, float *logits_out, uint32_t D, uint32_t NE, float eps);

/* One dense transformer layer in a SINGLE command buffer, fully device-resident:
 *   rmsnorm(attn_norm) -> QKV -> rope(q,k) -> append K/V to device caches ->
 *   attn_scores -> o_proj -> +resid -> rmsnorm(ffn_norm) -> gate/up -> swiglu ->
 *   quant -> down -> +resid
 * `hidden_dev` is persistent device state carried across layers; when first!=0
 * host `hidden_in` is staged into it, when last!=0 it is copied back to
 * `hidden_out`. KV caches are position-major [n_ctx][KV_dim] device buffers.
 * Requires Q8_0 weights, H*HD==D, no q/k norm or biases (caller verifies). */
/* Optional device-resident finalisation for the dense token path.
 * When supplied, output RMSNorm and the Q8 lm_head are recorded into the
 * same command buffer as the transformer layers. Only the final logits are
 * copied back to host memory. */
typedef struct VG_VKFinalHead {
    const VG_VKBuffer *norm_w;
    const VG_VKBuffer *lm_w, *lm_s;
    uint32_t vocab, D;
    float eps;
    float *logits_out;
} VG_VKFinalHead;

typedef struct VG_VKDenseLayer {
    const VG_VKBuffer *attn_norm_w;
    const VG_VKBuffer *q_w, *q_s, *k_w, *k_s, *v_w, *v_s;
    const VG_VKBuffer *rope_freqs;      /* may be NULL */
    VG_VKBuffer *k_cache, *v_cache;
    const VG_VKBuffer *o_w, *o_s;
    const VG_VKBuffer *ffn_norm_w;
    const VG_VKBuffer *gate_w, *gate_s, *up_w, *up_s, *down_w, *down_s;
    VG_VKBuffer *hidden_dev;
    const float *hidden_in;
    float *hidden_out;
    int first, last;
    uint32_t D, H, HKV, HD, KV_dim, n_ctx, n_rot, FF, Q_dim, pos;
    int has_freqs;
    float freq_base, eps;
} VG_VKDenseLayer;

VG_Status vg_vk_dense_layer(VG_VK *vk, const VG_VKDenseLayer *d);

/* Record ALL n layers into a SINGLE command buffer and submit once (one fence
 * wait per token instead of one per layer). L[0].first stages hidden_in and
 * L[n-1].last reads hidden_out back; intermediate layers keep hidden on device. */
VG_Status vg_vk_dense_forward(VG_VK *vk, const VG_VKDenseLayer *layers, uint32_t n);
VG_Status vg_vk_dense_forward_final(VG_VK *vk, const VG_VKDenseLayer *layers, uint32_t n,
                                    const VG_VKFinalHead *head);
#ifdef __cplusplus
}
#endif
#endif
