#include "vg/vulkan_backend.h"
#include <stdlib.h>
#include <string.h>
struct VG_VK { int unavailable; };
struct VG_VKBuffer { size_t bytes; };
VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out) { (void)cfg; if (out) *out = NULL; return VG_E_UNSUPPORTED; }
void vg_vk_close(VG_VK *vk) { free(vk); }
VG_Status vg_vk_info(const VG_VK *vk, VG_VKInfo *out) { (void)vk; (void)out; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_buffer_upload(VG_VK *vk, const void *data, size_t bytes, VG_VKBuffer **out) { (void)vk; (void)data; (void)bytes; if (out) *out = NULL; return VG_E_UNSUPPORTED; }
VG_Status vg_vk_buffer_read(VG_VK *vk, const VG_VKBuffer *buffer, void *data, size_t bytes) { (void)vk; (void)buffer; (void)data; (void)bytes; return VG_E_UNSUPPORTED; }
void vg_vk_buffer_release(VG_VK *vk, VG_VKBuffer *buffer) { (void)vk; free(buffer); }
VG_Status vg_vk_matvec_i8(VG_VK *vk, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) { (void)vk; (void)weights; (void)scales; (void)x; (void)y; (void)rows; (void)input; (void)output; return VG_E_UNSUPPORTED; }
