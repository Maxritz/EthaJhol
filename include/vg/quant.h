#ifndef VG_QUANT_H
#define VG_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VG_QuantKind {
    VG_QUANT_F32    = 0,
    VG_QUANT_F16    = 1,
    VG_QUANT_Q4_0   = 2,
    VG_QUANT_Q4_1   = 3,
    VG_QUANT_F64    = 4,
    VG_QUANT_Q5_0   = 6,
    VG_QUANT_Q5_1   = 7,
    VG_QUANT_Q8_0   = 8,
    VG_QUANT_Q8_1   = 9,
    VG_QUANT_Q2_K   = 10,
    VG_QUANT_Q3_K   = 11,
    VG_QUANT_Q4_K   = 12,
    VG_QUANT_Q5_K   = 13,
    VG_QUANT_Q6_K   = 14,
    VG_QUANT_Q8_K   = 15,
    VG_QUANT_IQ2_XXS = 16,
    VG_QUANT_IQ2_XS  = 17,
    VG_QUANT_IQ3_XXS = 18,
    VG_QUANT_IQ1_S   = 19,
    VG_QUANT_IQ4_NL  = 20,
    VG_QUANT_IQ3_S   = 21,
    VG_QUANT_IQ2_S   = 22,
    VG_QUANT_IQ4_XS  = 23,
    VG_QUANT_I8      = 24,
    VG_QUANT_I16     = 25,
    VG_QUANT_I32     = 26,
    VG_QUANT_I64     = 27,
    VG_QUANT_IQ1_M   = 29,
    VG_QUANT_BF16    = 30,
    VG_QUANT_MXFP4   = 39,
    VG_QUANT_NVFP4   = 40,
    VG_QUANT_Q1_0    = 41,
    VG_QUANT_Q2_0    = 42
} VG_QuantKind;

typedef struct VG_QuantInfo {
    VG_QuantKind kind;
    const char *name;
    uint32_t block_size;
    uint32_t bytes_per_block;
    uint32_t alignment;
    int supported_scalar;
    int supported_vulkan;
} VG_QuantInfo;

const VG_QuantInfo *vg_quant_info(uint32_t ggml_type);
VG_Status vg_quant_validate(uint32_t ggml_type, uint64_t elements, uint64_t bytes);
VG_Status vg_dequantize_row(uint32_t ggml_type, const void *packed, size_t packed_bytes,
                            float *out, size_t elements);
VG_Status vg_quantized_dot(uint32_t ggml_type, const void *packed, size_t packed_bytes,
                           const float *x, size_t elements, float *out);

#ifdef __cplusplus
}
#endif
#endif
