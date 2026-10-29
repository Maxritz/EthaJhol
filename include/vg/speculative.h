#ifndef VG_SPECULATIVE_H
#define VG_SPECULATIVE_H

#include "model_graph.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Draft/verify speculative decoding.

 * A small draft model proposes N tokens; the target model verifies them in ONE batched
 * position sweep. A proposal is accepted with probability min(1, p_target / p_draft), and
 * on rejection the token is drawn from the normalised residual max(0, p_target - p_draft).
 * That is the standard rule (Leviathan et al., Chen et al.) and it is what keeps the output
 * distribution identical to the target model. Comparing argmax tokens instead — which this
 * module used to do — silently changes the distribution whenever sampling is enabled.

 * Both graphs must share a vocabulary, and positions are addressed directly, so a rejected
 * proposal is discarded by simply overwriting the KV slot at that position. */
typedef struct VG_SpeculativeConfig {
    VG_ModelGraph *draft_graph;
    VG_ModelGraph *target_graph;
    int32_t n_draft;        /* proposals per step, clamped to [1, 64] */
    float temperature;      /* 0 => greedy acceptance (one-hot distributions) */
    uint32_t *rng;          /* optional PRNG state; a fixed default is used when NULL */
} VG_SpeculativeConfig;

/* Verify one draft window starting at `input_token` / `kv_pos`.

 * Writes the accepted token ids to `accepted` (which must hold max_accepted entries) and
 * returns how many were written, or 0 on error/invalid arguments. The first written token
 * is always the target model's own first token, matching plain decoding. */
int32_t vg_speculative_step(const VG_SpeculativeConfig *cfg, int32_t input_token,
                            int32_t kv_pos, int32_t *accepted, int32_t max_accepted);

#ifdef __cplusplus
}
#endif
#endif
