#include "vg/full_engine.h"
#include "vg/model_graph.h"
#include "vg/tensor_source.h"
#include "vg/tokenizer.h"
#ifdef VG_HAS_VULKAN
#include "vg/vulkan_backend.h"
#include "vg/trace.h"
#ifdef _WIN32
#include <windows.h>
#endif
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

void vg_apply_repetition_penalty(float *logits, uint32_t n_vocab,
                                 const int32_t *history, size_t hist_len,
                                 float penalty) {
    if (!logits || !history || hist_len == 0 || penalty <= 1.0f) return;
    for (size_t i = 0; i < hist_len; ++i) {
        int32_t t = history[i];
        if (t >= 0 && (uint32_t)t < n_vocab) logits[t] -= penalty;
    }
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
    fprintf(stderr, "[vg] DBG vg_generate_entered\n");
    VG_Status st = vg_gguf_open(model_path, &file);
    fprintf(stderr, "[vg] DBG after_gguf_open st=%d\n", (int)st);
    if (st) return st;
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);

    VG_TensorSource *source = NULL;
    VG_TensorSourceConfig scfg; memset(&scfg, 0, sizeof(scfg));
    scfg.host_budget_bytes = 256u * 1024u * 1024u;
    scfg.io_chunk_bytes = 4u * 1024u * 1024u;
    scfg.use_mmap = 1;
    st = vg_tensor_source_open(file, &scfg, &source);

    VG_ModelGraph *graph = NULL;
    st = vg_model_graph_load(file, source, &graph);
    if (st != VG_OK) { if (source) vg_tensor_source_close(source); vg_gguf_close(file); return st; }
    fprintf(stderr, "[vg] DBG graph_loaded\n");

#ifdef VG_HAS_VULKAN
    if (config && config->use_gpu) {
        VG_VKConfig vkcfg; memset(&vkcfg, 0, sizeof(vkcfg));
        vkcfg.shader_dir = shader_dir();
        vkcfg.prefer_discrete = 1; /* never bind a software ICD (SwiftShader) when a dGPU exists */
        VG_VK *vk = NULL;
        if (vg_vk_open(&vkcfg, &vk) == VG_OK && vk) {
            vg_model_graph_set_vulkan(graph, vk);
            VG_VKInfo vi;
            if (vg_vk_info(vk, &vi) == VG_OK) {
                fprintf(stderr, "[vg] Vulkan inference on %s (%.2f GiB VRAM, %s)\n",
                        vi.name, (double)vi.device_local_bytes / (1024.0 * 1024.0 * 1024.0),
                        vi.discrete ? "discrete" : "integrated");
            } else {
                fprintf(stderr, "[vg] Vulkan enabled for inference\n");
            }
        } else {
            fprintf(stderr, "[vg] warning: Vulkan init failed, falling back to CPU\n");
        }
    }
#endif

    fprintf(stderr, "[vg] DBG before_reset\n");
    vg_model_graph_reset(graph);
    fprintf(stderr, "[vg] DBG after_reset\n");

    const VG_ModelConfig *mcfg = vg_model_graph_config(graph);
    int32_t n_predict = config ? config->n_predict : 128;
    int32_t n_ctx = config ? config->n_ctx : (int32_t)mcfg->n_ctx;

    VG_Tokenizer *tok = NULL;
    fprintf(stderr, "[vg] DBG before_tokenizer_load\n");
    if (vg_tokenizer_load(file, &tok) != VG_OK) { vg_model_graph_free(graph); if (source) vg_tensor_source_close(source); vg_gguf_close(file); return VG_E_NOMEM; }
    fprintf(stderr, "[vg] DBG tok loaded bos=%d eos=%d byte=%d\n", vg_tokenizer_bos_id(tok), vg_tokenizer_eos_id(tok), vg_tokenizer_is_byte_bpe(tok));

    size_t n_prompt_tokens = 0;
    int32_t *prompt_tokens = NULL;
    if (prompt && prompt[0]) {
        n_prompt_tokens = vg_tokenizer_encode(tok, prompt, NULL, 0);
        fprintf(stderr, "[vg] DBG encode_count=%zu prompt_len=%zu\n", n_prompt_tokens, strlen(prompt));
        if (n_prompt_tokens > 0) {
            prompt_tokens = (int32_t *)malloc(n_prompt_tokens * sizeof(int32_t));
            if (prompt_tokens) vg_tokenizer_encode(tok, prompt, prompt_tokens, n_prompt_tokens);
            fprintf(stderr, "[vg] DBG prompt n=%zu:", n_prompt_tokens);
            for (size_t i = 0; i < n_prompt_tokens && i < 32; ++i) fprintf(stderr, " %d", prompt_tokens ? prompt_tokens[i] : -1);
            fprintf(stderr, "\n");
        }
    }

     /* The tokenizer owns BOS policy: Qwen3 sets add_bos_token=0 and has no bos id,
      * so prepending it would feed token -1 into the graph. Llama-3 templates begin
      * with <|begin_of_text|>, so avoid doubling the BOS (a second BOS at position 0
      * shifts every key/value by one and corrupts attention on the context). */
      int32_t bos = vg_tokenizer_add_bos(tok) ? vg_tokenizer_bos_id(tok) : -1;
      int32_t eos = vg_tokenizer_eos_id(tok);
      fprintf(stderr, "[vg] DBG bos=%d eos=%d add_bos=%d prompt0=%d\n", bos, eos, vg_tokenizer_add_bos(tok), n_prompt_tokens>0?(prompt_tokens?prompt_tokens[0]:-1):-1);
      if (bos >= 0 && n_prompt_tokens > 0 && prompt_tokens[0] == bos) bos = -1;





    float *logits = (float *)malloc(mcfg->n_vocab * sizeof(float));
    if (!logits) { free(prompt_tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); if (source) vg_tensor_source_close(source); vg_gguf_close(file); return VG_E_NOMEM; }

    /* Sampling config */
    float temp = config ? config->temperature : 0.0f;
    int32_t top_k = config ? config->top_k : 1;
    float top_p_val = config ? config->top_p : 1.0f;
    float repeat_penalty = config ? config->repeat_penalty : 1.0f;
    int32_t repeat_last_n = config ? config->repeat_last_n : 0;
    uint32_t rng_state = config ? config->seed : 42u;
    if (rng_state == 0) rng_state = 42u;

    /* History of recently used tokens (generated + tail of prompt) for the
     * repetition penalty, so the penalty is active from the first output token. */
    int32_t *recent = NULL;
    size_t recent_cap = (repeat_last_n > 0) ? (size_t)repeat_last_n : 0;
    size_t recent_len = 0;
    if (recent_cap) {
        recent = (int32_t *)calloc(recent_cap, sizeof(int32_t));
        if (!recent) recent_cap = 0;
    }

    /* Prefill */
    int32_t kv_pos = 0;
    fprintf(stderr, "[vg] prompt n=%zu:", n_prompt_tokens);
    for (size_t i = 0; i < n_prompt_tokens; ++i) fprintf(stderr, " %d", prompt_tokens ? prompt_tokens[i] : -1);
    fprintf(stderr, "\n");
    if (bos >= 0) {
        st = vg_model_graph_decode(graph, bos, kv_pos++, logits);
        if (st != VG_OK) goto done;
    }
    for (size_t i = 0; i < n_prompt_tokens && kv_pos < n_ctx; ++i) {
        st = vg_model_graph_decode(graph, prompt_tokens[i], kv_pos++, logits);
        if (st != VG_OK) goto done;
    }

    {
        int32_t b[8];
        for (int k = 0; k < 8; ++k) b[k] = -1;
        for (uint32_t i = 0; i < mcfg->n_vocab; ++i) {
            for (int k = 0; k < 8; ++k) {
                if (b[k] < 0 || logits[i] > logits[b[k]]) {
                    for (int m = 7; m > k; --m) b[m] = b[m - 1];
                    b[k] = (int32_t)i;
                    break;
                }
            }
        }
        fprintf(stderr, "[vg] top8@last:");
        for (int k = 0; k < 8; ++k) fprintf(stderr, " %d=%.4f", b[k], logits[b[k]]);
        fprintf(stderr, "\n");
    }

    /* Seed the repetition history with the tail of the prompt so the penalty
     * reaches back into the user's text from the first generated token. */
    if (recent_cap && n_prompt_tokens) {
        size_t start = n_prompt_tokens > recent_cap ? n_prompt_tokens - recent_cap : 0;
        for (size_t i = start; i < n_prompt_tokens; ++i) recent[recent_len++] = prompt_tokens[i];
    }
    free(prompt_tokens); prompt_tokens = NULL;

    /* Decode */
    int first_piece = 1;
    for (int32_t gen = 0; gen < n_predict && kv_pos < n_ctx; ++gen) {
        if (recent_cap && recent_len > 0)
            vg_apply_repetition_penalty(logits, mcfg->n_vocab, recent, recent_len, repeat_penalty);
        int32_t next = sample_next_token(logits, mcfg->n_vocab, temp, top_k, top_p_val, &rng_state);
        fprintf(stderr, "[vg] gen %d id=%d\n", gen, next);
        if (next == eos) break;

        char buf[256]; buf[0] = 0;
        size_t slen = vg_tokenizer_decode(tok, next, buf, sizeof(buf));
        if (slen > 0 && callback) {
            /* The first generated piece carries the SentencePiece dummy prefix; drop
             * it so streamed text does not start with a stray space. */
            if (first_piece && !vg_tokenizer_is_byte_bpe(tok) &&
                (unsigned char)vg_tokenizer_token_text(tok, next)[0] == 0xE2 &&
                (unsigned char)vg_tokenizer_token_text(tok, next)[1] == 0x96 &&
                (unsigned char)vg_tokenizer_token_text(tok, next)[2] == 0x81 &&
                buf[0] == ' ') {
                memmove(buf, buf + 1, slen - 1);
                --slen;
            }
            first_piece = 0;
            if (slen > 0 && !callback(buf, slen, next, user)) break;
        }

        if (recent_cap) {
            if (recent_len < recent_cap) { recent[recent_len++] = next; }
            else { memmove(recent, recent + 1, (recent_cap - 1) * sizeof(int32_t)); recent[recent_cap - 1] = next; }
        }

        uint64_t t_dec0 = vg_trace_now_ns();
        st = vg_model_graph_decode(graph, next, kv_pos++, logits);
        {
            static uint64_t dbg_sum = 0; static int dbg_n = 0;
            uint64_t dt = vg_trace_now_ns() - t_dec0;
            if (gen > 0) { dbg_sum += dt; ++dbg_n; if ((dbg_n % 8) == 0) fprintf(stderr, "[vg] decode avg=%.1f ms\n", (double)dbg_sum / 1e6 / (double)dbg_n); }
        }
        if (st != VG_OK) break;
    }

done:
    if (source) {
        VG_TensorSourceStats ts;
        vg_tensor_source_stats(source, &ts);
        fprintf(stderr, "[vg] stats: disk/copy read=%.1f MiB  host-resident=%.1f MiB (budget %.0f MiB)\n",
                (double)ts.bytes_read / 1048576.0, (double)ts.bytes_resident / 1048576.0, (double)ts.bytes_budget / 1048576.0);
    }
#ifdef VG_HAS_VULKAN
#endif
    free(recent);
    free(logits);
    if (prompt_tokens) free(prompt_tokens);
    vg_tokenizer_free(tok);
    vg_model_graph_free(graph);
    if (source) vg_tensor_source_close(source);
    vg_gguf_close(file);
  #ifdef _WIN32
    { IO_COUNTERS io; if (GetProcessIoCounters(GetCurrentProcess(), &io)) fprintf(stderr, "[vg] stats: disk read=%.1f MiB\n", (double)io.ReadTransferCount / 1048576.0); }
  #endif
    return st;
}
