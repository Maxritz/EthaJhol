#include "vg/speculative.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VG_SPEC_MAX_DRAFT 64

/* Softmax into `p` (already allocated, n_vocab entries). Temperature 0 produces a one-hot
 * distribution on the argmax, which is exactly what greedy decoding means and lets the same
 * acceptance rule cover both modes. */
static void dist_from_logits(const float *logits, uint32_t n_vocab, float temperature, float *p) {
    if (temperature <= 0.0f) {
        memset(p, 0, (size_t)n_vocab * sizeof(float));
        int32_t best = 0;
        float bv = logits[0];
        for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
        p[best] = 1.0f;
        return;
    }
    float mx = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > mx) mx = logits[i];
    double total = 0.0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        p[i] = expf((logits[i] - mx) / temperature);
        if (!isfinite(p[i]) || p[i] < 0.0f) p[i] = 0.0f;
        total += (double)p[i];
    }
    if (total > 0.0) {
        const float inv = (float)(1.0 / total);
        for (uint32_t i = 0; i < n_vocab; ++i) p[i] *= inv;
    }
}

static uint32_t next_rand(uint32_t *state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

static float rand_unit(uint32_t *state) {
    return (float)(next_rand(state) >> 8) / 16777216.0f;
}

/* Draw from the normalised residual max(0, target - draft). If the residual is empty (the
 * distributions agreed everywhere) fall back to the target distribution. */
static int32_t sample_residual(const float *draft, const float *target, uint32_t n_vocab, uint32_t *rng) {
    double total = 0.0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        const float r = target[i] - draft[i];
        if (r > 0.0f) total += (double)r;
    }
    if (!(total > 0.0)) {
        total = 0.0;
        for (uint32_t i = 0; i < n_vocab; ++i) total += (double)target[i];
        if (!(total > 0.0)) return 0;
        const double r = (double)rand_unit(rng) * total;
        double acc = 0.0;
        for (uint32_t i = 0; i < n_vocab; ++i) {
            acc += (double)target[i];
            if (acc >= r) return (int32_t)i;
        }
        return (int32_t)(n_vocab - 1);
    }
    const double r = (double)rand_unit(rng) * total;
    double acc = 0.0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        const float d = target[i] - draft[i];
        if (d > 0.0f) {
            acc += (double)d;
            if (acc >= r) return (int32_t)i;
        }
    }
    return 0;
}

static int32_t sample_from(const float *p, uint32_t n_vocab, uint32_t *rng) {
    const float r = rand_unit(rng);
    float acc = 0.0f;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        acc += p[i];
        if (acc >= r) return (int32_t)i;
    }
    return (int32_t)(n_vocab - 1);
}

int32_t vg_speculative_step(const VG_SpeculativeConfig *cfg, int32_t input_token,
                            int32_t kv_pos, int32_t *accepted, int32_t max_accepted) {
    if (!cfg || !cfg->draft_graph || !cfg->target_graph || !accepted || max_accepted < 1) return 0;
    const VG_ModelConfig *dcfg = vg_model_graph_config(cfg->draft_graph);
    const VG_ModelConfig *tcfg = vg_model_graph_config(cfg->target_graph);
    if (!dcfg || !tcfg || dcfg->n_vocab != tcfg->n_vocab) return 0;
    const uint32_t V = tcfg->n_vocab;

    int32_t n_draft = cfg->n_draft;
    if (n_draft <= 0) n_draft = 4;
    if (n_draft > VG_SPEC_MAX_DRAFT) n_draft = VG_SPEC_MAX_DRAFT;
    if (n_draft > max_accepted) n_draft = max_accepted;

    float *d_logits = (float *)malloc((size_t)V * sizeof(float));
    float *t_logits = (float *)malloc((size_t)V * sizeof(float));
    float *p_draft = (float *)malloc((size_t)V * sizeof(float));
    float *p_target = (float *)malloc((size_t)V * sizeof(float));
    int32_t *draft = (int32_t *)malloc((size_t)n_draft * sizeof(int32_t));
    /* Per-step probability the draft gave its own proposal. Acceptance compares that
     * against the target's probability for the same token, so it has to be kept per step:
     * the distribution buffer is reused for every draft decode. */
    float *q_prop = (float *)malloc((size_t)n_draft * sizeof(float));
    if (!d_logits || !t_logits || !p_draft || !p_target || !draft || !q_prop) {
        free(d_logits); free(t_logits); free(p_draft); free(p_target); free(draft); free(q_prop);
        return 0;
    }

    uint32_t local_rng = 0x9E3779B9u;
    uint32_t *rng = cfg->rng ? cfg->rng : &local_rng;
    const float temp = cfg->temperature > 0.0f ? cfg->temperature : 0.0f;

    /* 1. Draft n_draft tokens. */
    int32_t produced = 0;
    for (int32_t i = 0; i < n_draft; ++i) {
        const int32_t in = (i == 0) ? input_token : draft[i - 1];
        if (vg_model_graph_decode(cfg->draft_graph, in, kv_pos + i, d_logits) != VG_OK) break;
        dist_from_logits(d_logits, V, temp, p_draft);
        if (temp > 0.0f) {
            draft[i] = sample_from(p_draft, V, rng);
        } else {
            int32_t best = 0;
            for (uint32_t v = 1; v < V; ++v) if (d_logits[v] > d_logits[best]) best = (int32_t)v;
            draft[i] = best;
        }
        q_prop[i] = p_draft[draft[i]];
        ++produced;
    }
    if (produced == 0) {
        free(d_logits); free(t_logits); free(p_draft); free(p_target); free(draft); free(q_prop);
        return 0;
    }

    /* 2. Verify with the target model at the same positions. */
    int32_t written = 0;
    int32_t pos = kv_pos;
    int32_t feed = input_token;
    for (int32_t i = 0; i < produced && written < max_accepted; ++i) {
        if (vg_model_graph_decode(cfg->target_graph, feed, pos, t_logits) != VG_OK) break;
        dist_from_logits(t_logits, V, temp, p_target);

        /* Target distribution at this position is the ground truth for the proposal. */
        if (temp <= 0.0f) {
            int32_t best = 0;
            for (uint32_t v = 1; v < V; ++v) if (p_target[v] > p_target[best]) best = (int32_t)v;
            if (draft[i] == best) {
                accepted[written++] = best;
                feed = best;
                ++pos;
                continue;
            }
            accepted[written++] = best;
            break;   /* rejected: the target token replaces the proposal */
        }

        const float q = q_prop[i];           /* draft probability of the proposal */
        const float p = p_target[draft[i]];  /* target probability of the proposal */
        const float accept_p = q > 0.0f ? fminf(1.0f, p / q) : 1.0f;
        if (rand_unit(rng) <= accept_p) {
            accepted[written++] = draft[i];
            feed = draft[i];
            ++pos;
            continue;
        }
        /* Rejection: re-derive this step's draft distribution (the buffer was reused by the
         * later proposals) and draw from the residual max(0, p_target - p_draft). */
        if (vg_model_graph_decode(cfg->draft_graph, feed, pos, d_logits) == VG_OK) {
            dist_from_logits(d_logits, V, temp, p_draft);
        } else {
            memset(p_draft, 0, (size_t)V * sizeof(float));
        }
        accepted[written++] = sample_residual(p_draft, p_target, V, rng);
        break;
    }

    free(d_logits); free(t_logits); free(p_draft); free(p_target); free(draft); free(q_prop);
    return written;
}
