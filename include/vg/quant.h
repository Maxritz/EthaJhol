#ifndef VG_QUANT_H
#define VG_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
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
    /* Ternary / 1-bit types (BitNet b1.58 and the Ternary-Bonsai family). Layouts are
     * byte-identical to ggml: TQ1_0 = 54 B/256 w (1.6875 bpw, radix-3 packing),
     * TQ2_0 = 66 B/256 w (2.0625 bpw, 2-bit planes), Q1_0 = 18 B/128 w, Q2_0 = 18 B/64 w. */
    VG_QUANT_TQ1_0   = 34,
    VG_QUANT_TQ2_0   = 35,
    VG_QUANT_MXFP4   = 39,
    VG_QUANT_NVFP4   = 40,
    VG_QUANT_Q1_0    = 41,   /* 128 weights per 18-byte block: 1 sign bit per weight     */
    VG_QUANT_Q2_0    = 42    /*  64 weights per 18-byte block: 2 bits, codes -1/0/+1/+2  */
} VG_QuantKind;

#define VG_QUANT_BYTES_PER_BLOCK_F32     4
#define VG_QUANT_BYTES_PER_BLOCK_F16     2
#define VG_QUANT_BYTES_PER_BLOCK_Q4_0    18
#define VG_QUANT_BYTES_PER_BLOCK_Q4_1    20
#define VG_QUANT_BYTES_PER_BLOCK_F64     8
#define VG_QUANT_BYTES_PER_BLOCK_Q5_0    22
#define VG_QUANT_BYTES_PER_BLOCK_Q5_1    24
#define VG_QUANT_BYTES_PER_BLOCK_Q8_0    34
#define VG_QUANT_BYTES_PER_BLOCK_Q8_1    36
#define VG_QUANT_BYTES_PER_BLOCK_Q2_K    84
#define VG_QUANT_BYTES_PER_BLOCK_Q3_K    110
#define VG_QUANT_BYTES_PER_BLOCK_Q4_K    144
#define VG_QUANT_BYTES_PER_BLOCK_Q5_K    176
#define VG_QUANT_BYTES_PER_BLOCK_Q6_K    210
#define VG_QUANT_BYTES_PER_BLOCK_Q8_K    292
#define VG_QUANT_BYTES_PER_BLOCK_IQ2_XXS 66
#define VG_QUANT_BYTES_PER_BLOCK_IQ2_XS  74
#define VG_QUANT_BYTES_PER_BLOCK_IQ3_XXS 98
#define VG_QUANT_BYTES_PER_BLOCK_IQ1_S   50
#define VG_QUANT_BYTES_PER_BLOCK_IQ4_NL  18
#define VG_QUANT_BYTES_PER_BLOCK_IQ3_S   110
#define VG_QUANT_BYTES_PER_BLOCK_IQ2_S   82
#define VG_QUANT_BYTES_PER_BLOCK_IQ4_XS  136
#define VG_QUANT_BYTES_PER_BLOCK_I8      1
#define VG_QUANT_BYTES_PER_BLOCK_I16     2
#define VG_QUANT_BYTES_PER_BLOCK_I32     4
#define VG_QUANT_BYTES_PER_BLOCK_I64     8
#define VG_QUANT_BYTES_PER_BLOCK_IQ1_M   56
#define VG_QUANT_BYTES_PER_BLOCK_BF16    2
#define VG_QUANT_BYTES_PER_BLOCK_MXFP4   17
#define VG_QUANT_BYTES_PER_BLOCK_NVFP4   36
#define VG_QUANT_BYTES_PER_BLOCK_Q1_0    18
#define VG_QUANT_BYTES_PER_BLOCK_Q2_0    18

typedef struct VG_QuantInfo {
    VG_QuantKind kind;
    const char *name;
    uint32_t block_size;
    uint32_t bytes_per_block;
    uint32_t alignment;
    int supported_scalar;   /* dequantize + fused matmul available on the CPU path */
    int supported_vulkan;   /* dedicated compute shader exists in shaders/ */
} VG_QuantInfo;

/* IEEE-754 binary16 -> binary32, exactly rounded (including subnormals and Inf/NaN).
 * Shared by every CPU kernel so dequantization never disagrees between paths. */
static inline float vg_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            /* Subnormal half: normalize the mantissa into a binary32 normal. */
            uint32_t e = 127u - 15u + 1u;
            while (!(mant & 0x400u)) { mant <<= 1u; --e; }
            mant &= 0x3FFu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

const VG_QuantInfo *vg_quant_info(uint32_t ggml_type);
VG_Status vg_quant_validate(uint32_t ggml_type, uint64_t elements, uint64_t bytes);
VG_Status vg_dequantize_row(uint32_t ggml_type, const void *packed, size_t packed_bytes,
                            float *out, size_t elements);
VG_Status vg_quantized_dot(uint32_t ggml_type, const void *packed, size_t packed_bytes,
                           const float *x, size_t elements, float *out);

/* Dequantize exactly one block (block_size elements) into out. Used by blocked kernels
 * that never materialize a full dequantized matrix. */
VG_Status vg_dequantize_block(uint32_t ggml_type, const void *block, float *out);

#ifdef __cplusplus
}
#endif
#endif
