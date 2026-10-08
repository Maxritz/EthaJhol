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
} VG_GenerateConfig;

typedef int (*VG_TokenCallback)(const char *piece, size_t bytes, int32_t token_id, void *user);

/* Subtract `penalty` from the logit of every token that appears in `history`,
 * to discourage repetition. Uses the additive form (logit -= penalty per
 * occurrence); tokens absent from history are untouched. This is the llama.cpp
 * "additive" repeat-penalty variant and is numerically safe because it only
 * shifts already-computed logits, so softmax ordering is preserved up to the
 * penalized mass. Idempotent w.r.t. the same history (applying it twice on an
 * unchanged history is NOT intended). Pure function: safe to unit-test. */
void vg_apply_repetition_penalty(float *logits, uint32_t n_vocab,
                                 const int32_t *history, size_t hist_len,
                                 float penalty);

/* Native CPU inference engine. Loads a GGUF model, evaluates the transformer
 * graph, samples the next token, and streams pieces back through the callback. */
VG_Status vg_generate(const char *model_path, const char *prompt,
                      const VG_GenerateConfig *config,
                      VG_TokenCallback callback, void *user);

#ifdef __cplusplus
}
#endif
#endif
