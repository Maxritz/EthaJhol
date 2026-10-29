#ifndef VG_CPU_OPS_H
#define VG_CPU_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fused quantized matrix-vector product: y[out_dim] = W[out_dim, in_dim] * x[in_dim].
 * W stays packed in GGUF quantized format; x and y are float32. */
void vg_cpu_matmul_q4_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
void vg_cpu_matmul_q4_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
void vg_cpu_matmul_q6_k(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
void vg_cpu_matmul_q8_0(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
void vg_cpu_matmul_f32(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);
void vg_cpu_matmul_f16(const void *W, const float *x, float *y, uint32_t out_dim, uint32_t in_dim);

/* Generic quantized matmul dispatching on ggml_type. Returns VG_E_UNSUPPORTED for types
 * without a scalar dequantizer. Row-parallel when more than one CPU thread is enabled. */
VG_Status vg_cpu_matmul(uint32_t ggml_type, const void *W, const float *x, float *y,
                        uint32_t out_dim, uint32_t in_dim);

/* Single-threaded, allocation-free reference path with identical math. Used by tests to
 * verify the threaded/fused kernels never drift. */
VG_Status vg_cpu_matmul_ref(uint32_t ggml_type, const void *W, const float *x, float *y,
                            uint32_t out_dim, uint32_t in_dim);

/* Thread count for the CPU kernels. 0 = auto (hardware concurrency). Returns the count
 * that will actually be used. */
int vg_cpu_set_threads(int n_threads);
int vg_cpu_get_threads(void);

/* RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i] */
void vg_cpu_rmsnorm(float *out, const float *x, const float *weight, uint32_t dim, float eps);

/* RoPE (Rotary Position Embedding) in-place on q and k. */
void vg_cpu_rope(float *q, float *k, uint32_t head_dim, uint32_t n_rot,
                 float freq_scale, float theta_base, int32_t pos);

/* Precomputed cos/sin table for one position, shared by all heads (n_rot/2 entries). */
void vg_cpu_rope_table(float *cos_out, float *sin_out, uint32_t n_rot, uint32_t head_dim,
                       float freq_scale, float theta_base, int32_t pos);
/* Apply a precomputed table to a single head in place. */
void vg_cpu_rope_apply(float *v, uint32_t head_dim, uint32_t n_rot,
                       const float *cos_tab, const float *sin_tab);
/* Apply a precomputed table to all heads of a [n_head, head_dim] tensor in place. */
void vg_cpu_rope_apply_all(float *v, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                           const float *cos_tab, const float *sin_tab);

/* Softmax in-place on x[0..dim-1]. */
void vg_cpu_softmax(float *x, uint32_t dim);

/* SwiGLU: out = swish(gate) * up, where gate/up are [dim] and out is [dim]. */
void vg_cpu_swiglu(float *out, const float *gate, const float *up, uint32_t dim);

/* GELU approximation. */
void vg_cpu_gelu(float *x, uint32_t dim);

/* Add two vectors. */
void vg_cpu_add(float *out, const float *a, const float *b, uint32_t dim);

/* Scale a vector in-place. */
void vg_cpu_scale(float *x, uint32_t dim, float scale);

/* Dot product with a bias-free scale, used by the graph evaluator. */
float vg_cpu_dot(const float *a, const float *b, uint32_t dim);

#ifdef __cplusplus
}
#endif
#endif
