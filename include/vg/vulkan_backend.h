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
VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out);
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes);
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer);

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

#ifdef __cplusplus
}
#endif
#endif
