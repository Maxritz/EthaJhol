#include "vg/vulkan_backend.h"

#include <stdlib.h>
#include <string.h>

/* CPU-only fallback used when no Vulkan SDK is available. Every entry point reports
 * VG_E_UNSUPPORTED so callers fall back to the CPU kernels instead of silently doing
 * nothing. It must cover the complete header: vg_core's model_graph references the whole
 * dispatch surface, so a partial stub fails at link time. */
struct VG_VK { int unavailable; };
struct VG_VKBuffer { size_t bytes; };

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out) { (void)cfg; if (out) *out = NULL; return VG_E_UNSUPPORTED; }
void vg_vk_close(VG_VK *vk) { free(vk); }
VG_Status vg_vk_info(const VG_VK *vk, VG_VKInfo *out) { (void)vk; (void)out; return VG_E_UNSUPPORTED; }
uint64_t vg_vk_memory_used(const VG_VK *vk) { (void)vk; return 0; }

VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out) {
    (void)vk; (void)data; (void)bytes;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes) { (void)vk; (void)buffer; (void)data; (void)bytes; return VG_E_UNSUPPORTED; }
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer) { (void)vk; free(buffer); }

VG_Status vg_vk_cache_acquire(VG_VK *vk, const void *key, uint32_t tag,
                              const void *data, size_t bytes, VG_VKBuffer **out) {
    (void)vk; (void)key; (void)tag; (void)data; (void)bytes;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_cache_get(VG_VK *vk, const void *key, uint32_t tag, VG_VKBuffer **out) {
    (void)vk; (void)key; (void)tag;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_scratch_acquire(VG_VK *vk, uint32_t slot, size_t bytes, VG_VKBuffer **out) {
    (void)vk; (void)slot; (void)bytes;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_dev_scratch_acquire(VG_VK *vk, uint32_t slot, size_t bytes, VG_VKBuffer **out) {
    (void)vk; (void)slot; (void)bytes;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_matvec_q8_0_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n) { (void)vk; (void)reqs; (void)n; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_matvec_f32_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n) { (void)vk; (void)reqs; (void)n; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_matvec_f16_batch(VG_VK *vk, const VG_VKMatvecReq *reqs, uint32_t n) { (void)vk; (void)reqs; (void)n; return VG_E_UNSUPPORTED; }

#define VG_STUB_MATVEC(name) \
    VG_Status name(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales, \
                   const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) { \
        (void)vk; (void)weights; (void)scales; (void)x; (void)y; (void)rows; (void)input; (void)output; \
        return VG_E_UNSUPPORTED; \
    }
VG_STUB_MATVEC(vg_vk_matvec_i8)
VG_STUB_MATVEC(vg_vk_matvec_f16)
VG_STUB_MATVEC(vg_vk_matvec_f32)
VG_STUB_MATVEC(vg_vk_matvec_q4_0)
VG_STUB_MATVEC(vg_vk_matvec_q8_0)
#undef VG_STUB_MATVEC

#define VG_STUB_MATVEC_K(name) \
    VG_Status name(VG_VK *vk, const VG_VKBuffer *w, const float *x, float *y, \
                   uint32_t rows, uint32_t input, uint32_t output) { \
        (void)vk; (void)w; (void)x; (void)y; (void)rows; (void)input; (void)output; \
        return VG_E_UNSUPPORTED; \
    }
VG_STUB_MATVEC_K(vg_vk_matvec_q4_k)
VG_STUB_MATVEC_K(vg_vk_matvec_q6_k)
#undef VG_STUB_MATVEC_K

VG_Status vg_vk_rmsnorm(VG_VK *vk, const VG_VKBuffer *input, const VG_VKBuffer *weight,
                        float *out, uint32_t dim, float eps) {
    (void)vk; (void)input; (void)weight; (void)out; (void)dim; (void)eps;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_softmax(VG_VK *vk, VG_VKBuffer *data, uint32_t dim) { (void)vk; (void)data; (void)dim; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_rope(VG_VK *vk, VG_VKBuffer *data, uint32_t head_dim, uint32_t n_rot,
                     float freq_scale, float theta_base, uint32_t pos) {
    (void)vk; (void)data; (void)head_dim; (void)n_rot; (void)freq_scale; (void)theta_base; (void)pos;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_swiglu(VG_VK *vk, const VG_VKBuffer *gate, const VG_VKBuffer *up,
                       VG_VKBuffer *out, uint32_t dim) {
    (void)vk; (void)gate; (void)up; (void)out; (void)dim;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_ffn_fused(VG_VK *vk,
                          const VG_VKBuffer *gate_w, const VG_VKBuffer *gate_s,
                          const VG_VKBuffer *up_w,   const VG_VKBuffer *up_s,
                          const VG_VKBuffer *down_w, const VG_VKBuffer *down_s,
                          const float *x, float *y, uint32_t D, uint32_t FF) {
    (void)vk; (void)gate_w; (void)gate_s; (void)up_w; (void)up_s; (void)down_w; (void)down_s;
    (void)x; (void)y; (void)D; (void)FF;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_layer_tail(VG_VK *vk,
                           const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
                           const float *attn, const float *resid,
                           const VG_VKBuffer *norm_w,
                           const VG_VKBuffer *gate_w, const VG_VKBuffer *gate_s,
                           const VG_VKBuffer *up_w,   const VG_VKBuffer *up_s,
                           const VG_VKBuffer *down_w, const VG_VKBuffer *down_s,
                           float *y, uint32_t D, uint32_t FF, float eps) {
    (void)vk; (void)o_w; (void)o_s; (void)attn; (void)resid; (void)norm_w;
    (void)gate_w; (void)gate_s; (void)up_w; (void)up_s; (void)down_w; (void)down_s;
    (void)y; (void)D; (void)FF; (void)eps;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_ffn_fused_x2(VG_VK *vk,
    const VG_VKBuffer *gw0, const VG_VKBuffer *gs0, const VG_VKBuffer *uw0, const VG_VKBuffer *us0, const VG_VKBuffer *dw0, const VG_VKBuffer *ds0,
    const VG_VKBuffer *gw1, const VG_VKBuffer *gs1, const VG_VKBuffer *uw1, const VG_VKBuffer *us1, const VG_VKBuffer *dw1, const VG_VKBuffer *ds1,
    const float *x, float *y0, float *y1, uint32_t D, uint32_t FF) {
    (void)vk; (void)gw0; (void)gs0; (void)uw0; (void)us0; (void)dw0; (void)ds0;
    (void)gw1; (void)gs1; (void)uw1; (void)us1; (void)dw1; (void)ds1;
    (void)x; (void)y0; (void)y1; (void)D; (void)FF;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_moe_head(VG_VK *vk,
    const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
    const float *attn, const float *resid,
    const VG_VKBuffer *norm_w,
    const VG_VKBuffer *route_w, const VG_VKBuffer *route_s,
    float *hidden_out, float *logits_out, uint32_t D, uint32_t NE, float eps) {
    (void)vk; (void)o_w; (void)o_s; (void)attn; (void)resid; (void)norm_w;
    (void)route_w; (void)route_s; (void)hidden_out; (void)logits_out; (void)D; (void)NE; (void)eps;
    return VG_E_UNSUPPORTED;
}

VG_Status vg_vk_dense_layer(VG_VK *vk, const VG_VKDenseLayer *d) { (void)vk; (void)d; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_dense_forward(VG_VK *vk, const VG_VKDenseLayer *layers, uint32_t n) { (void)vk; (void)layers; (void)n; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_dense_forward_final(VG_VK *vk, const VG_VKDenseLayer *layers, uint32_t n,
                                    const VG_VKFinalHead *head) {
    (void)vk; (void)layers; (void)n; (void)head;
    return VG_E_UNSUPPORTED;
}
