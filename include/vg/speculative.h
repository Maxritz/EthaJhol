#ifndef VG_SPECULATIVE_H
#define VG_SPECULATIVE_H

#include "model_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_SpeculativeConfig {
    VG_ModelGraph *draft_graph;    /* smaller draft model */
    VG_ModelGraph *target_graph;   /* larger target model */
    int32_t n_draft;               /* number of draft tokens to speculatively generate */
    float temperature;
    int32_t top_k;
    float top_p;
    uint32_t seed;
} VG_SpeculativeConfig;

/* Run speculative decoding for one step. Returns the number of accepted tokens (>= 1).
 * accepted_tokens: output array of accepted token IDs (up to n_draft + 1).
 * The first token is always the target model's greedy choice; subsequent are accepted drafts. */
int32_t vg_speculative_step(const VG_SpeculativeConfig *cfg, int32_t input_token,
                            int32_t kv_pos, int32_t *accepted_tokens);

#ifdef __cplusplus
}
#endif
#endif
