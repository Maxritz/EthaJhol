#ifndef VG_FULL_ENGINE_H
#define VG_FULL_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"
#include "tensor_source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_GenerateConfig {
    int32_t n_predict;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_gpu_layers;
    uint32_t seed;
    int32_t top_k;
    float top_p;
    float temperature;
    float repeat_penalty;
    int32_t repeat_last_n;
    int32_t lazy_mode;       /* 0 off, 1 auto, 2 on */
    int32_t load_mode;       /* llama_load_mode numeric value; -1 auto */
    int32_t use_mmap;        /* compatibility alias: 0 disables, nonzero enables */
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t n_seq_max;
    const char *lora_path;
    float lora_scale;
    VG_TensorSource *prefetch_source; /* optional: native tensor source for readahead */
} VG_GenerateConfig;

typedef int (*VG_TokenCallback)(const char *piece, size_t bytes, int32_t token_id, void *user);

/* Full graph backend entry point. The llama.cpp bridge is optional at build time; the C ABI
 * remains usable by another model plugin with the same streaming contract. */
VG_Status vg_generate(const char *model_path, const char *prompt,
                      const VG_GenerateConfig *config,
                      VG_TokenCallback callback, void *user);

/* Native CPU inference engine (no llama.cpp dependency). Uses model_graph + tokenizer. */
VG_Status vg_generate_native(const char *model_path, const char *prompt,
                              const VG_GenerateConfig *config,
                              VG_TokenCallback callback, void *user);

#ifdef __cplusplus
}
#endif
#endif
