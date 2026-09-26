#include "vg/full_engine.h"
#include "vg/model_graph.h"
#include "vg/tensor_source.h"
#include "vg/tokenizer.h"
#ifdef VG_HAS_VULKAN
#include "vg/vulkan_backend.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static int32_t sample_next_token(const float *logits, uint32_t n_vocab,
                                  float temperature, int32_t top_k, float top_p,
                                  uint32_t *rng) {
    *rng = *rng * 1103515245u + 12345u;

    if (temperature <= 0.0f || top_k <= 1) {
        int32_t best = 0;
        for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = (int32_t)i;
        return best;
    }

    float *probs = (float *)malloc(n_vocab * sizeof(float));
    if (!probs) {
        int32_t best = 0;
        for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = (int32_t)i;
        return best;
    }

    float mx = logits[0] / temperature;
    for (uint32_t i = 1; i < n_vocab; ++i) {
        float s = logits[i] / temperature;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        probs[i] = expf(logits[i] / temperature - mx);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < n_vocab; ++i) probs[i] /= sum;

    /* Top-k */
    if (top_k > 0 && (uint32_t)top_k < n_vocab) {
        float *tmp = (float *)malloc(n_vocab * sizeof(float));
        if (tmp) {
            memcpy(tmp, probs, n_vocab * sizeof(float));
            for (int32_t i = 0; i < top_k; ++i) {
                uint32_t mx_idx = (uint32_t)i;
                for (uint32_t j = (uint32_t)i + 1; j < n_vocab; ++j)
                    if (tmp[j] > tmp[mx_idx]) mx_idx = j;
                float t = tmp[i]; tmp[i] = tmp[mx_idx]; tmp[mx_idx] = t;
            }
            float threshold = tmp[top_k - 1];
            free(tmp);
            for (uint32_t i = 0; i < n_vocab; ++i)
                if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    /* Top-p */
    if (top_p > 0.0f && top_p < 1.0f) {
        int *idx = (int *)malloc(n_vocab * sizeof(int));
        if (idx) {
            for (uint32_t i = 0; i < n_vocab; ++i) idx[i] = (int)i;
            float cumsum = 0.0f;
            uint32_t cutoff = n_vocab;
            for (uint32_t i = 0; i < n_vocab; ++i) {
                uint32_t mx_idx = i;
                for (uint32_t j = i + 1; j < n_vocab; ++j)
                    if (probs[idx[j]] > probs[idx[mx_idx]]) mx_idx = j;
                int t = idx[i]; idx[i] = idx[mx_idx]; idx[mx_idx] = t;
                cumsum += probs[idx[i]];
                if (cumsum >= top_p) { cutoff = i + 1; break; }
            }
            for (uint32_t i = cutoff; i < n_vocab; ++i) probs[idx[i]] = 0.0f;
            free(idx);
        }
        sum = 0.0f;
        for (uint32_t i = 0; i < n_vocab; ++i) sum += probs[i];
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < n_vocab; ++i) probs[i] /= sum;
        } else {
            free(probs);
            int32_t best = 0;
            for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = (int32_t)i;
            return best;
        }
    }

    /* Sample */
    float r = (float)(*rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float cumsum = 0.0f;
    int32_t result = 0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) { result = (int32_t)i; break; }
    }
    free(probs);
    return result;
}

#ifdef VG_HAS_VULKAN
static const char *shader_dir(void) {
    FILE *f = fopen("shaders/matvec_i8.comp.spv", "rb"); if (f) { fclose(f); return "shaders"; }
    f = fopen("build/shaders/matvec_i8.comp.spv", "rb"); if (f) { fclose(f); return "build/shaders"; }
    return "shaders";
}
#endif

VG_Status vg_generate(const char *model_path, const char *prompt,
                              const VG_GenerateConfig *config,
                              VG_TokenCallback callback, void *user) {
    VG_GGUF *file = NULL;
    VG_Status st = vg_gguf_open(model_path, &file);
    if (st != VG_OK) return st;

    VG_TensorSource *source = NULL;
    VG_TensorSourceConfig scfg; memset(&scfg, 0, sizeof(scfg));
    scfg.host_budget_bytes = 256u * 1024u * 1024u;
    scfg.io_chunk_bytes = 4u * 1024u * 1024u;
    scfg.use_mmap = 1;
    st = vg_tensor_source_open(file, &scfg, &source);
    fprintf(stderr, "[dbg] tensor_source_open: st=%d source=%p\n", (int)st, (void*)source);

    VG_ModelGraph *graph = NULL;
    st = vg_model_graph_load(file, source, &graph);
    fprintf(stderr, "[dbg] model_graph_load: st=%d graph=%p\n", (int)st, (void*)graph);
    if (st != VG_OK) { fprintf(stderr, "[vg-trace] FAIL: vg_model_graph_load = %d\n", (int)st); if (source) vg_tensor_source_close(source); vg_gguf_close(file); return st; }

#ifdef VG_HAS_VULKAN
    if (config && config->use_gpu) {
        VG_VKConfig vkcfg; memset(&vkcfg, 0, sizeof(vkcfg));
        vkcfg.shader_dir = shader_dir();
        VG_VK *vk = NULL;
        if (vg_vk_open(&vkcfg, &vk) == VG_OK && vk) {
            vg_model_graph_set_vulkan(graph, vk);
            fprintf(stderr, "[vg-trace] Vulkan enabled for inference\n");
        } else {
            fprintf(stderr, "[vg-trace] WARN: Vulkan init failed, falling back to CPU\n");
        }
    }
#endif

    vg_model_graph_reset(graph);

    const VG_ModelConfig *mcfg = vg_model_graph_config(graph);
    int32_t n_predict = config ? config->n_predict : 128;
    int32_t n_ctx = config ? config->n_ctx : (int32_t)mcfg->n_ctx;

    VG_Tokenizer *tok = NULL;
    if (vg_tokenizer_load(file, &tok) != VG_OK) { vg_model_graph_free(graph); if (source) vg_tensor_source_close(source); vg_gguf_close(file); return VG_E_NOMEM; }

    size_t n_prompt_tokens = 0;
    int32_t *prompt_tokens = NULL;
    if (prompt && prompt[0]) {
        n_prompt_tokens = vg_tokenizer_encode(tok, prompt, NULL, 0);
        if (n_prompt_tokens > 0) {
            prompt_tokens = (int32_t *)malloc(n_prompt_tokens * sizeof(int32_t));
            if (prompt_tokens) vg_tokenizer_encode(tok, prompt, prompt_tokens, n_prompt_tokens);
        }
    }

    int32_t bos = mcfg->bos_id;
    int32_t eos = mcfg->eos_id;

    float *logits = (float *)malloc(mcfg->n_vocab * sizeof(float));
    if (!logits) { free(prompt_tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); if (source) vg_tensor_source_close(source); vg_gguf_close(file); return VG_E_NOMEM; }

    /* Prefill */
    int32_t kv_pos = 0;
    st = vg_model_graph_decode(graph, bos, kv_pos++, logits);
    if (st != VG_OK) goto done;
    for (size_t i = 0; i < n_prompt_tokens && kv_pos < n_ctx; ++i) {
        st = vg_model_graph_decode(graph, prompt_tokens[i], kv_pos++, logits);
        if (st != VG_OK) { fprintf(stderr, "[dbg] decode prompt token %zu failed st=%d\n", i, (int)st); break; }
    }
    free(prompt_tokens); prompt_tokens = NULL;

    /* Sampling config */
    float temp = config ? config->temperature : 0.0f;
    int32_t top_k = config ? config->top_k : 1;
    float top_p_val = config ? config->top_p : 1.0f;
    uint32_t rng_state = config ? config->seed : 42u;
    if (rng_state == 0) rng_state = 42u;

    /* Decode */
    for (int32_t gen = 0; gen < n_predict && kv_pos < n_ctx; ++gen) {
        int32_t next = sample_next_token(logits, mcfg->n_vocab, temp, top_k, top_p_val, &rng_state);
        fprintf(stderr, "[dbg] sampled token=%d logits[0]=%.4f logits[%d]=%.4f\n", next, logits[0], next, logits[next]);
        if (next == eos) break;

        char buf[256]; buf[0] = 0;
        size_t slen = vg_tokenizer_decode(tok, next, buf, sizeof(buf));
        if (slen > 0 && callback) {
            if (!callback(buf, slen, next, user)) break;
        }

        st = vg_model_graph_decode(graph, next, kv_pos++, logits);
        if (st != VG_OK) break;
    }

done:
    fprintf(stderr, "[dbg] cleanup: logits\n"); free(logits);
    if (prompt_tokens) free(prompt_tokens);
    fprintf(stderr, "[dbg] cleanup: tokenizer\n"); vg_tokenizer_free(tok);
    fprintf(stderr, "[dbg] cleanup: graph\n"); vg_model_graph_free(graph);
    fprintf(stderr, "[dbg] cleanup: source\n"); if (source) vg_tensor_source_close(source);
    fprintf(stderr, "[dbg] cleanup: gguf\n"); vg_gguf_close(file);
    fprintf(stderr, "[dbg] cleanup: done\n");
    return st;
}
