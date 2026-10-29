#include "vg/full_engine.h"
#include "vg/model_graph.h"
#include "vg/model_probe.h"
#include "vg/lora.h"
#include "vg/tensor_source.h"
#include "vg/tokenizer.h"
#include "vg/cpu_ops.h"
#ifdef VG_HAS_VULKAN
#include "vg/vulkan_backend.h"
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static const char *st_name(VG_Status s) {
    switch (s) {
        case VG_OK: return "ok";
        case VG_E_INVALID: return "invalid argument";
        case VG_E_IO: return "io error";
        case VG_E_FORMAT: return "malformed file";
        case VG_E_RANGE: return "out of range";
        case VG_E_NOMEM: return "out of memory";
        case VG_E_UNSUPPORTED: return "unsupported";
        case VG_E_BUSY: return "busy";
        default: return "error";
    }
}

static int verbose(void) {
    static int cached = -1;
    if (cached < 0) { const char *e = getenv("VG_DEBUG"); cached = (e && *e && strcmp(e, "0") != 0) ? 1 : 0; }
    return cached;
}

/* Repeat penalty (Keskar et al. style, as used by llama.cpp): scale already-seen logits
 * before sampling so the model does not loop on the same token. */
static void apply_repeat_penalty(float *logits, uint32_t n_vocab, const int32_t *recent,
                                 int32_t n_recent, float penalty) {
    if (!logits || !recent || n_recent <= 0 || penalty <= 1.0f) return;
    for (int32_t i = 0; i < n_recent; ++i) {
        const int32_t t = recent[i];
        if (t < 0 || (uint32_t)t >= n_vocab) continue;
        if (logits[t] > 0.0f) logits[t] /= penalty;
        else logits[t] *= penalty;
    }
}

static int32_t argmax_logits(const float *logits, uint32_t n_vocab) {
    int32_t best = 0;
    float bv = logits[0];
    for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
    return best;
}

typedef struct VG_Cand { float p; uint32_t id; } VG_Cand;

/* Descending by probability, ascending by id so ties are deterministic. */
static int cand_desc(const void *a, const void *b) {
    const VG_Cand *x = (const VG_Cand *)a, *y = (const VG_Cand *)b;
    if (x->p < y->p) return 1;
    if (x->p > y->p) return -1;
    return x->id < y->id ? -1 : (x->id > y->id ? 1 : 0);
}

/* Temperature / top-k / top-p sampling in O(V log V): one sort, then one pass each for the
 * nucleus and the draw. The previous top-p loop rescanned the whole vocabulary per selected
 * token, i.e. O(V^2) per generated token (about 8e9 comparisons for a 128k vocabulary). */
static int32_t sample_next_token(const float *logits_in, uint32_t n_vocab,
                                 float temperature, int32_t top_k, float top_p,
                                 uint32_t *rng, VG_Cand *cand) {
    *rng = *rng * 1103515245u + 12345u;
    if (temperature <= 0.0f || top_k == 1 || !cand) return argmax_logits(logits_in, n_vocab);

    float mx = logits_in[0];
    for (uint32_t i = 1; i < n_vocab; ++i) if (logits_in[i] > mx) mx = logits_in[i];
    double total = 0.0;
    for (uint32_t i = 0; i < n_vocab; ++i) {
        cand[i].p = expf((logits_in[i] - mx) / temperature);
        cand[i].id = i;
        total += (double)cand[i].p;
    }
    if (!(total > 0.0) || !isfinite(total)) return argmax_logits(logits_in, n_vocab);

    qsort(cand, n_vocab, sizeof(*cand), cand_desc);

    uint32_t keep = (top_k > 0 && (uint32_t)top_k < n_vocab) ? (uint32_t)top_k : n_vocab;
    if (top_p > 0.0f && top_p < 1.0f) {
        const double target = (double)top_p * total;
        double cum = 0.0;
        uint32_t nucleus = 0;
        for (uint32_t i = 0; i < keep; ++i) {
            cum += (double)cand[i].p;
            ++nucleus;
            if (cum >= target) break;
        }
        if (nucleus) keep = nucleus;
    }

    double norm = 0.0;
    for (uint32_t i = 0; i < keep; ++i) norm += (double)cand[i].p;
    if (!(norm > 0.0)) return argmax_logits(logits_in, n_vocab);

    const double r = (double)(*rng >> 8) / 16777216.0;
    double acc = 0.0;
    for (uint32_t i = 0; i < keep; ++i) {
        acc += (double)cand[i].p / norm;
        if (acc >= r) return (int32_t)cand[i].id;
    }
    return (int32_t)cand[0].id;
}

#ifdef VG_HAS_VULKAN
static const char *shader_dir(void) {
    FILE *f = fopen("shaders/matvec_q4_0.comp.spv", "rb");
    if (f) { fclose(f); return "shaders"; }
    f = fopen("build/shaders/matvec_q4_0.comp.spv", "rb");
    if (f) { fclose(f); return "build/shaders"; }
    return "shaders";
}
#endif

VG_Status vg_generate(const char *model_path, const char *prompt,
                      const VG_GenerateConfig *config,
                      VG_TokenCallback callback, void *user) {
    if (!model_path) return VG_E_INVALID;
    VG_GGUF *file = NULL;
    VG_Status st = vg_gguf_open(model_path, &file);
    if (st != VG_OK) {
        /* A silent exit 2 on a typo'd path looked like a crash; always name the cause. */
        fprintf(stderr, "cannot open model '%s': %s\n", model_path, st_name(st));
        return st;
    }

    /* Refuse models the evaluator cannot execute, naming the exact unsupported piece. */
    VG_ModelProbe probe;
    vg_model_probe(file, &probe);
    if (!probe.ok) {
        /* probe.detail already names the offending tensor, so it is the whole message. */
        fprintf(stderr, "model not supported: %s\n", probe.detail);
        vg_gguf_close(file);
        return probe.status == VG_OK ? VG_E_UNSUPPORTED : probe.status;
    }

    const int threads = vg_cpu_set_threads(config ? config->n_threads : 0);
    if (verbose()) fprintf(stderr, "[vg] cpu threads=%d\n", threads);

    VG_TensorSource *source = NULL;
    VG_TensorSourceConfig scfg;
    memset(&scfg, 0, sizeof(scfg));
    scfg.host_budget_bytes = 256u * 1024u * 1024u;
    scfg.io_chunk_bytes = 4u * 1024u * 1024u;
    scfg.use_mmap = config ? (config->use_mmap != 0) : 1;
    st = vg_tensor_source_open(file, &scfg, &source);
    if (st != VG_OK) { vg_gguf_close(file); return st; }

    VG_ModelGraph *graph = NULL;
    st = vg_model_graph_load(file, source, &graph);
    if (st != VG_OK) { vg_tensor_source_close(source); vg_gguf_close(file); return st; }

    /* LoRA is loaded before the first decode so the delta is present for every token. */
    VG_GGUF *lora_file = NULL;
    VG_LoraAdapter *lora = NULL;
    if (config && config->lora_path && config->lora_path[0]) {
        VG_Status lst = vg_gguf_open(config->lora_path, &lora_file);
        if (lst != VG_OK) {
            fprintf(stderr, "cannot open LoRA adapter '%s': %s\n", config->lora_path, st_name(lst));
            vg_model_graph_free(graph); vg_tensor_source_close(source); vg_gguf_close(file);
            return lst;
        }
        lst = vg_lora_load(lora_file, &lora);
        if (lst != VG_OK || !lora || vg_lora_target_count(lora) == 0) {
            fprintf(stderr, "LoRA adapter '%s' has no usable lora_a/lora_b pairs\n", config->lora_path);
            if (lora) vg_lora_free(lora);
            vg_gguf_close(lora_file);
            vg_model_graph_free(graph); vg_tensor_source_close(source); vg_gguf_close(file);
            return VG_E_FORMAT;
        }
        vg_model_graph_set_lora(graph, lora, config->lora_scale != 0.0f ? config->lora_scale : 1.0f);
        if (verbose() || 1) {
            fprintf(stderr, "[vg] lora: %zu target(s), rank %u, scale %.3f\n",
                    vg_lora_target_count(lora), vg_lora_rank(lora),
                    (double)(config->lora_scale != 0.0f ? config->lora_scale : 1.0f));
        }
    }

    /* An offload request that cannot be honoured must say so: silently running on the CPU
     * looks identical to a GPU run and hides a misconfigured deployment. */
    if (config && config->use_gpu) {
#ifndef VG_HAS_VULKAN
        fprintf(stderr, "[vg] %d layer(s) requested on GPU, but this build has no Vulkan backend; "
                        "running on CPU (0 offloaded)\n", config->ngl_layers);
#endif
    }
#ifdef VG_HAS_VULKAN
    VG_VK *vk = NULL;
    if (config && config->use_gpu) {
        VG_VKConfig vkcfg;
        memset(&vkcfg, 0, sizeof(vkcfg));
        vkcfg.device_index = UINT32_MAX;
        vkcfg.shader_dir = shader_dir();
        vkcfg.prefer_discrete = 1;
        if (vg_vk_open(&vkcfg, &vk) == VG_OK && vk) {
            vg_model_graph_set_vulkan(graph, vk);
            if (verbose()) {
                VG_VKInfo vi;
                vg_vk_info(vk, &vi);
                fprintf(stderr, "[vg] vulkan device=%s (%s)\n", vi.name, vi.discrete ? "discrete" : "integrated");
            }
        } else {
            fprintf(stderr, "[vg] %d layer(s) requested on GPU, but no usable Vulkan device was opened; "
                            "running on CPU (0 offloaded)\n", config->ngl_layers);
            vk = NULL;
        }
    }
#endif

    vg_model_graph_reset(graph);

    const VG_ModelConfig *mcfg = vg_model_graph_config(graph);
    int32_t n_predict = config ? config->n_predict : 128;
    int32_t n_ctx = config ? config->n_ctx : (int32_t)mcfg->n_ctx;
    if (n_ctx <= 0) n_ctx = (int32_t)mcfg->n_ctx;
    /* The KV cache is sized from the model metadata; never step past it. */
    if ((uint32_t)n_ctx > mcfg->n_ctx) n_ctx = (int32_t)mcfg->n_ctx;

    VG_Tokenizer *tok = NULL;
    if (!probe.has_tokenizer || vg_tokenizer_load(file, &tok) != VG_OK) {
        fprintf(stderr, "model has no usable tokenizer (tokenizer.ggml.model missing)\n");
        if (lora) vg_lora_free(lora);
        if (lora_file) vg_gguf_close(lora_file);
        vg_model_graph_free(graph);
        vg_tensor_source_close(source);
        vg_gguf_close(file);
#ifdef VG_HAS_VULKAN
        if (vk) vg_vk_close(vk);
#endif
        return VG_E_UNSUPPORTED;
    }

    size_t n_prompt_tokens = 0;
    int32_t *prompt_tokens = NULL;
    if (prompt && prompt[0]) {
        n_prompt_tokens = vg_tokenizer_encode(tok, prompt, NULL, 0);
        if (n_prompt_tokens > 0) {
            prompt_tokens = (int32_t *)malloc(n_prompt_tokens * sizeof(int32_t));
            if (prompt_tokens) vg_tokenizer_encode(tok, prompt, prompt_tokens, n_prompt_tokens);
            else n_prompt_tokens = 0;
        }
    }
    if (n_prompt_tokens + 2 > (size_t)n_ctx) {
        n_prompt_tokens = (size_t)n_ctx > 2 ? (size_t)n_ctx - 2 : 0;
        if (verbose()) fprintf(stderr, "[vg] prompt truncated to %zu tokens (ctx %d)\n", n_prompt_tokens, n_ctx);
    }

    const int32_t bos = mcfg->bos_id;
    const int32_t eos = mcfg->eos_id;
    float *logits = (float *)malloc((size_t)mcfg->n_vocab * sizeof(float));
    if (!logits) {
        free(prompt_tokens); vg_tokenizer_free(tok);
        if (lora) vg_lora_free(lora);
        if (lora_file) vg_gguf_close(lora_file);
        vg_model_graph_free(graph);
        vg_tensor_source_close(source); vg_gguf_close(file);
#ifdef VG_HAS_VULKAN
        if (vk) vg_vk_close(vk);
#endif
        return VG_E_NOMEM;
    }

    int32_t kv_pos = 0;
    /* One sampling scratch buffer for the whole run: the sort workspace is reused per token. */
    VG_Cand *cand = (VG_Cand *)malloc((size_t)mcfg->n_vocab * sizeof(VG_Cand));
    st = vg_model_graph_decode(graph, bos, kv_pos++, logits);
    for (size_t i = 0; st == VG_OK && i < n_prompt_tokens && kv_pos < n_ctx; ++i)
        st = vg_model_graph_decode(graph, prompt_tokens[i], kv_pos++, logits);

    const float temp = config ? config->temperature : 0.0f;
    const int32_t top_k = config ? config->top_k : 1;
    const float top_p_val = config ? config->top_p : 1.0f;
    const float repeat_penalty = config ? config->repeat_penalty : 1.0f;
    int32_t repeat_last_n = config ? config->repeat_last_n : 0;
    if (repeat_last_n < 0) repeat_last_n = 0;
    uint32_t rng_state = config ? config->seed : 42u;
    if (rng_state == 0u || rng_state == 0xffffffffu) rng_state = 42u;

    int32_t *recent = NULL;
    if (repeat_last_n > 0 && repeat_penalty > 1.0f) {
        recent = (int32_t *)calloc((size_t)repeat_last_n, sizeof(int32_t));
        repeat_last_n = recent ? repeat_last_n : 0;
    }
    /* Seed the penalty window from the prompt tail: without this the model only ever penalises
     * its own output and happily repeats phrases that are already in the prompt. */
    int32_t n_recent = 0;
    if (recent && prompt_tokens) {
        const size_t keep_from = n_prompt_tokens > (size_t)repeat_last_n
                                     ? n_prompt_tokens - (size_t)repeat_last_n : 0;
        for (size_t i = keep_from; i < n_prompt_tokens && n_recent < repeat_last_n; ++i)
            recent[n_recent++] = prompt_tokens[i];
    }
    free(prompt_tokens);
    prompt_tokens = NULL;

    for (int32_t gen = 0; st == VG_OK && gen < n_predict && kv_pos < n_ctx; ++gen) {
        apply_repeat_penalty(logits, mcfg->n_vocab, recent, n_recent, repeat_penalty);
        const int32_t next = sample_next_token(logits, mcfg->n_vocab, temp, top_k, top_p_val, &rng_state, cand);
        if (next == eos) break;

        char buf[256];
        buf[0] = 0;
        const size_t slen = vg_tokenizer_decode(tok, next, buf, sizeof(buf));
        if (slen > 0 && callback && !callback(buf, slen, next, user)) break;

        if (recent) {
            if (n_recent < repeat_last_n) recent[n_recent++] = next;
            else {
                memmove(recent, recent + 1, (size_t)(repeat_last_n - 1) * sizeof(int32_t));
                recent[repeat_last_n - 1] = next;
            }
        }
        st = vg_model_graph_decode(graph, next, kv_pos++, logits);
    }

    free(recent);
    free(cand);
    free(logits);
    vg_tokenizer_free(tok);
    vg_model_graph_free(graph);
    if (lora) vg_lora_free(lora);
    if (lora_file) vg_gguf_close(lora_file);
    vg_tensor_source_close(source);
    vg_gguf_close(file);
#ifdef VG_HAS_VULKAN
    if (vk) vg_vk_close(vk);
#endif
    return st;
}
