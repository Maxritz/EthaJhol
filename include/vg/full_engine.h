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
    uint32_t seed;
    int32_t top_k;
    float top_p;
    float temperature;
    float repeat_penalty;
    int32_t repeat_last_n;
    int32_t use_mmap;
    int32_t n_threads;
    int32_t use_gpu;
    int32_t ngl_layers;
} VG_GenerateConfig;

typedef int (*VG_TokenCallback)(const char *piece, size_t bytes, int32_t token_id, void *user);

/* Native CPU inference engine. Loads a GGUF model, evaluates the transformer
 * graph, samples the next token, and streams pieces back through the callback. */
VG_Status vg_generate(const char *model_path, const char *prompt,
                      const VG_GenerateConfig *config,
                      VG_TokenCallback callback, void *user);

#ifdef __cplusplus
}
#endif
#endif
