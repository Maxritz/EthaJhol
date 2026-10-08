#include "vg/speculative.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int32_t greedy_sample(const float *logits, uint32_t n_vocab) {
    int32_t best = 0; float best_val = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > best_val) { best_val = logits[i]; best = (int32_t)i; }
    return best;
}

static float token_logprob(VG_ModelGraph *g, int32_t token, int32_t kv_pos) {
    float *logits = (float *)malloc(vg_model_graph_config(g)->n_vocab * sizeof(float));
    if (!logits) return -100.0f;
    VG_Status st = vg_model_graph_decode(g, token, kv_pos, logits);
    if (st != VG_OK) { free(logits); return -100.0f; }
    /* Softmax */
    uint32_t nv = vg_model_graph_config(g)->n_vocab;
    float mx = logits[0]; for (uint32_t i = 1; i < nv; ++i) if (logits[i] > mx) mx = logits[i];
    float s = 0.0f; for (uint32_t i = 0; i < nv; ++i) { logits[i] = expf(logits[i] - mx); s += logits[i]; }
    float prob = logits[token] / s;
    free(logits);
    return prob > 0.0f ? logf(prob) : -100.0f;
}

int32_t vg_speculative_step(const VG_SpeculativeConfig *cfg, int32_t input_token,
                            int32_t kv_pos, int32_t *accepted_tokens) {
    if (!cfg || !cfg->draft_graph || !cfg->target_graph || !accepted_tokens) return 0;
    const VG_ModelConfig *dcfg = vg_model_graph_config(cfg->draft_graph);
    const VG_ModelConfig *tcfg = vg_model_graph_config(cfg->target_graph);
    int32_t n_draft = cfg->n_draft;
    if (n_draft <= 0) n_draft = 4;

    float *draft_logits = (float *)malloc(dcfg->n_vocab * sizeof(float));
    float *target_logits = (float *)malloc(tcfg->n_vocab * sizeof(float));
    if (!draft_logits || !target_logits) { free(draft_logits); free(target_logits); return 0; }

    int32_t draft_tokens[256];
    int32_t pos = kv_pos;

    /* Generate draft tokens */
    VG_Status st = vg_model_graph_decode(cfg->draft_graph, input_token, pos, draft_logits);
    if (st != VG_OK) { free(draft_logits); free(target_logits); return 0; }
    draft_tokens[0] = greedy_sample(draft_logits, dcfg->n_vocab);

    for (int32_t i = 1; i < n_draft; ++i) {
        st = vg_model_graph_decode(cfg->draft_graph, draft_tokens[i - 1], pos + i, draft_logits);
        if (st != VG_OK) break;
        draft_tokens[i] = greedy_sample(draft_logits, dcfg->n_vocab);
    }

    /* Verify with target model */
    st = vg_model_graph_decode(cfg->target_graph, input_token, pos, target_logits);
    if (st != VG_OK) { free(draft_logits); free(target_logits); return 0; }
    int32_t target_first = greedy_sample(target_logits, tcfg->n_vocab);

    /* If draft[0] != target[0], reject all drafts */
    if (draft_tokens[0] != target_first) {
        accepted_tokens[0] = target_first;
        free(draft_logits); free(target_logits);
        return 1;
    }

    accepted_tokens[0] = target_first;
    int32_t n_accepted = 1;

    /* Verify remaining draft tokens */
    for (int32_t i = 1; i < n_draft; ++i) {
        st = vg_model_graph_decode(cfg->target_graph, draft_tokens[i - 1], pos + i, target_logits);
        if (st != VG_OK) break;
        int32_t target_tok = greedy_sample(target_logits, tcfg->n_vocab);
        if (draft_tokens[i] == target_tok) {
            accepted_tokens[n_accepted++] = target_tok;
        } else {
            /* Mismatch: accept the target's token and stop */
            accepted_tokens[n_accepted++] = target_tok;
            break;
        }
    }

    free(draft_logits); free(target_logits);
    return n_accepted;
}
