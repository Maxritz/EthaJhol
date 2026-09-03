#include "vg/quant.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static void test_legacy(void) {
    unsigned char q4[18] = {0x00, 0x3c}; memset(q4 + 2, 0x88, 16); float out[32]; assert(vg_dequantize_row(VG_QUANT_Q4_0, q4, sizeof(q4), out, 32) == VG_OK); for (size_t i = 0; i < 32; ++i) assert(fabsf(out[i]) < 1e-6f);
    unsigned char q8[34] = {0x00, 0x3c}; for (size_t i = 0; i < 32; ++i) q8[2 + i] = (unsigned char)(i + 1); float x[32]; for (size_t i = 0; i < 32; ++i) x[i] = 1.0f; float dot = 0; assert(vg_quantized_dot(VG_QUANT_Q8_0, q8, sizeof(q8), x, 32, &dot) == VG_OK); assert(fabsf(dot - 528.0f) < 1e-3f);
}
static void test_scalar(void) {
    uint16_t f16[2] = {0x3c00, 0xc000}; float out[2]; assert(vg_dequantize_row(VG_QUANT_F16, f16, sizeof(f16), out, 2) == VG_OK); assert(fabsf(out[0] - 1.0f) < 1e-6f && fabsf(out[1] + 2.0f) < 1e-6f);
    assert(vg_quant_validate(VG_QUANT_Q2_0, 64, 18) == VG_OK); assert(vg_quant_validate(VG_QUANT_Q2_0, 32, 18) != VG_OK); assert(vg_quant_info(VG_QUANT_Q6_K)->bytes_per_block == 210);
}
int main(void) { test_legacy(); test_scalar(); return 0; }
