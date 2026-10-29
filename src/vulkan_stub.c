#include "vg/vulkan_backend.h"

#include <stdlib.h>

/* CPU-only fallback used when no Vulkan SDK is available. Every entry point reports
 * VG_E_UNSUPPORTED so callers fall back to the CPU kernels instead of silently doing nothing. */
struct VG_VK { int unavailable; };
struct VG_VKBuffer { size_t bytes; };
struct VG_VKWeight { uint32_t ggml_type; };

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out) { (void)cfg; if (out) *out = NULL; return VG_E_UNSUPPORTED; }
void vg_vk_close(VG_VK *vk) { free(vk); }
VG_Status vg_vk_info(const VG_VK *vk, VG_VKInfo *out) { (void)vk; (void)out; return VG_E_UNSUPPORTED; }
void vg_vk_stats(const VG_VK *vk, uint64_t *uploads, uint64_t *dispatches, uint64_t *bytes_resident) {
    (void)vk;
    if (uploads) *uploads = 0;
    if (dispatches) *dispatches = 0;
    if (bytes_resident) *bytes_resident = 0;
}
int vg_vk_supports_type(uint32_t ggml_type) { (void)ggml_type; return 0; }

VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out) {
    (void)vk; (void)data; (void)bytes;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes) { (void)vk; (void)buffer; (void)data; (void)bytes; return VG_E_UNSUPPORTED; }
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer) { (void)vk; free(buffer); }

VG_Status vg_vk_weight_upload(VG_VK *vk, const void *packed, size_t packed_bytes,
                              uint32_t ggml_type, uint32_t out_dim, uint32_t in_dim,
                              VG_VKWeight **out) {
    (void)vk; (void)packed; (void)packed_bytes; (void)ggml_type; (void)out_dim; (void)in_dim;
    if (out) *out = NULL;
    return VG_E_UNSUPPORTED;
}
void vg_vk_weight_release(VG_VK *vk, VG_VKWeight *weight) { (void)vk; free(weight); }
VG_Status vg_vk_matvec(VG_VK *vk, const VG_VKWeight *weight, const float *x, float *y,
                       uint32_t out_dim, uint32_t in_dim) {
    (void)vk; (void)weight; (void)x; (void)y; (void)out_dim; (void)in_dim;
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_matvec_q4_0(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vk; (void)weights; (void)x; (void)y; (void)out_dim; (void)in_dim; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_matvec_q8_0(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vk; (void)weights; (void)x; (void)y; (void)out_dim; (void)in_dim; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_matvec_f32(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vk; (void)weights; (void)x; (void)y; (void)out_dim; (void)in_dim; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_matvec_f16(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    (void)vk; (void)weights; (void)x; (void)y; (void)out_dim; (void)in_dim; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_rmsnorm(VG_VK *vk, const float *x, const float *weight, float *out, uint32_t dim, float eps) {
    (void)vk; (void)x; (void)weight; (void)out; (void)dim; (void)eps; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_softmax(VG_VK *vk, float *data, uint32_t dim) { (void)vk; (void)data; (void)dim; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_swiglu(VG_VK *vk, const float *gate, const float *up, float *out, uint32_t dim) {
    (void)vk; (void)gate; (void)up; (void)out; (void)dim; return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_rope(VG_VK *vk, float *data, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                     float theta_base, uint32_t pos) {
    (void)vk; (void)data; (void)n_head; (void)head_dim; (void)n_rot; (void)theta_base; (void)pos;
    return VG_E_UNSUPPORTED;
}
