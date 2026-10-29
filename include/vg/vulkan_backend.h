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
typedef struct VG_VKWeight VG_VKWeight;

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
    uint32_t device_index;      /* UINT32_MAX = pick automatically */
    const char *shader_dir;
    uint64_t memory_budget_bytes;
    int prefer_discrete;
} VG_VKConfig;

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out);
void vg_vk_close(VG_VK *vk);
VG_Status vg_vk_info(const VG_VK *vk, VG_VKInfo *out);
/* Lifetime counters for the residency cache, useful for diagnostics. */
void vg_vk_stats(const VG_VK *vk, uint64_t *uploads, uint64_t *dispatches, uint64_t *bytes_resident);

/* True when a dedicated compute shader exists for this ggml type. */
int vg_vk_supports_type(uint32_t ggml_type);

/* Resident weight: the packed GGML tensor lives in device memory and is reused by every
 * matvec that references it. Upload once per weight, not once per token. */
VG_Status vg_vk_weight_upload(VG_VK *vk, const void *packed, size_t packed_bytes,
                              uint32_t ggml_type, uint32_t out_dim, uint32_t in_dim,
                              VG_VKWeight **out);
void vg_vk_weight_release(VG_VK *vk, VG_VKWeight *weight);
VG_Status vg_vk_matvec(VG_VK *vk, const VG_VKWeight *weight, const float *x, float *y,
                       uint32_t out_dim, uint32_t in_dim);

/* Raw device buffers (host visible) for callers that manage their own residency. */
VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out);
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes);
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer);

/* One-shot matvec against a packed host tensor (self-test / fallback path). */
VG_Status vg_vk_matvec_q4_0(VG_VK *vk, const void *weights, const float *x, float *y,
                            uint32_t out_dim, uint32_t in_dim);
VG_Status vg_vk_matvec_q8_0(VG_VK *vk, const void *weights, const float *x, float *y,
                            uint32_t out_dim, uint32_t in_dim);
VG_Status vg_vk_matvec_f32(VG_VK *vk, const void *weights, const float *x, float *y,
                           uint32_t out_dim, uint32_t in_dim);
VG_Status vg_vk_matvec_f16(VG_VK *vk, const void *weights, const float *x, float *y,
                           uint32_t out_dim, uint32_t in_dim);

/* Vector operations. Each stages host data, dispatches once and reads the result back. */
VG_Status vg_vk_rmsnorm(VG_VK *vk, const float *x, const float *weight, float *out,
                        uint32_t dim, float eps);
VG_Status vg_vk_softmax(VG_VK *vk, float *data, uint32_t dim);
VG_Status vg_vk_swiglu(VG_VK *vk, const float *gate, const float *up, float *out, uint32_t dim);
VG_Status vg_vk_rope(VG_VK *vk, float *data, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                     float theta_base, uint32_t pos);

#ifdef __cplusplus
}
#endif
#endif
