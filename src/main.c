#include "vg/gguf.h"
#include "vg/tensor_source.h"
#include "vg/tokenizer.h"
#include "vg/model_graph.h"
#ifdef VG_BUILD_FULL_ENGINE
#include "vg/full_engine.h"
#endif
#ifdef VG_HAS_VULKAN
#include "vg/vulkan_backend.h"
#include <vulkan/vulkan.h>
#endif
#include "vg/plugin_loader.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

static const char *status_name(VG_Status s) { switch (s) { case VG_OK: return "ok"; case VG_E_INVALID: return "invalid"; case VG_E_IO: return "io"; case VG_E_FORMAT: return "format"; case VG_E_RANGE: return "range"; case VG_E_NOMEM: return "nomem"; case VG_E_UNSUPPORTED: return "unsupported"; default: return "error"; } }

static int32_t sample_token(float *logits, uint32_t n_vocab, float temperature, int32_t top_k,
                            float top_p, float repeat_penalty, const int32_t *hist, int32_t hist_len,
                            uint32_t *rng) {
    if (temperature <= 0.0f) {
        int32_t best = 0; for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = (int32_t)i;
        return best;
    }
    if (repeat_penalty != 1.0f && hist) {
        for (int32_t i = 0; i < hist_len; ++i) {
            int32_t t = hist[i];
            if (t >= 0 && (uint32_t)t < n_vocab && logits[t] > 0.0f) logits[t] /= repeat_penalty;
            else if (t >= 0 && (uint32_t)t < n_vocab && logits[t] < 0.0f) logits[t] *= repeat_penalty;
        }
    }
    for (uint32_t i = 0; i < n_vocab; ++i) logits[i] /= temperature;
    /* Softmax */
    float mx = logits[0]; for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float *probs = (float *)malloc(n_vocab * sizeof(float));
    if (!probs) { int32_t best = 0; for (uint32_t i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = (int32_t)i; return best; }
    float s = 0.0f;
    for (uint32_t i = 0; i < n_vocab; ++i) { probs[i] = expf(logits[i] - mx); s += probs[i]; }
    for (uint32_t i = 0; i < n_vocab; ++i) probs[i] /= s;
    /* Top-k */
    if (top_k > 0 && (uint32_t)top_k < n_vocab) {
        float *tmp = (float *)malloc(n_vocab * sizeof(float));
        if (tmp) {
            memcpy(tmp, probs, n_vocab * sizeof(float));
            for (int32_t i = 0; i < top_k; ++i) {
                int32_t mi = i; for (uint32_t j = i + 1; j < n_vocab; ++j) if (tmp[j] > tmp[mi]) mi = (int32_t)j;
                float t = tmp[i]; tmp[i] = tmp[mi]; tmp[mi] = t;
            }
            float thresh = tmp[top_k - 1]; free(tmp);
            for (uint32_t i = 0; i < n_vocab; ++i) if (probs[i] < thresh) probs[i] = 0.0f;
        }
    }
    /* Top-p */
    if (top_p > 0.0f && top_p < 1.0f) {
        int32_t *idx = (int32_t *)malloc(n_vocab * sizeof(int32_t));
        if (idx) {
            for (uint32_t i = 0; i < n_vocab; ++i) idx[i] = (int32_t)i;
            float cum = 0.0f; uint32_t cutoff = n_vocab;
            for (uint32_t i = 0; i < n_vocab; ++i) {
                int32_t mi = (int32_t)i; for (uint32_t j = i + 1; j < n_vocab; ++j) if (probs[idx[j]] > probs[idx[mi]]) mi = (int32_t)j;
                int32_t t = idx[i]; idx[i] = idx[mi]; idx[mi] = t;
                cum += probs[idx[i]];
                if (cum >= top_p) { cutoff = i + 1; break; }
            }
            for (uint32_t i = cutoff; i < n_vocab; ++i) probs[idx[i]] = 0.0f; free(idx);
            s = 0.0f; for (uint32_t i = 0; i < n_vocab; ++i) s += probs[i];
            if (s > 0.0f) for (uint32_t i = 0; i < n_vocab; ++i) probs[i] /= s;
        }
    }
    /* Sample from distribution */
    *rng = *rng * 1103515245u + 12345u;
    float r = (float)(*rng >> 16) / 65536.0f;
    int32_t best = 0; float cum = probs[0];
    for (uint32_t i = 1; i < n_vocab; ++i) { cum += probs[i]; if (cum >= r) { best = (int32_t)i; break; } }
    free(probs);
    return best;
}
static VG_Status count_chunk(const void *data, size_t bytes, uint64_t off, void *user) { (void)data; (void)off; *(uint64_t *)user += bytes; return VG_OK; }
static int inspect(const char *path) {
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g); if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    printf("GGUF: %s\nsize: %" PRIu64 " bytes\ndata_base: %" PRIu64 "\ntensors: %" PRIu64 "\n", path, vg_gguf_file_size(g), vg_gguf_data_base(g), vg_gguf_tensor_count(g));
    const char *arch = vg_gguf_meta(g, "general.architecture"); const char *name = vg_gguf_meta(g, "general.name"); if (arch) printf("architecture: %s\n", arch); if (name) printf("name: %s\n", name);
    for (uint64_t i = 0, shown = 0; i < vg_gguf_tensor_count(g) && shown < 12; ++i, ++shown) { const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i); printf("  %-48s type=%u dims=", t->name, t->ggml_type); for (uint32_t d = 0; d < t->n_dims; ++d) printf("%s%" PRIu64, d ? "x" : "", t->dims[d]); printf(" bytes=%" PRIu64 "\n", t->nbytes); }
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg)); cfg.host_budget_bytes = 256u * 1024u * 1024u; cfg.io_chunk_bytes = 4u * 1024u * 1024u; cfg.use_mmap = 1; VG_TensorSource *src = NULL; st = vg_tensor_source_open(g, &cfg, &src); if (st == VG_OK) { VG_TensorSourceStats ss; vg_tensor_source_stats(src, &ss); printf("lazy source: host_budget=%" PRIu64 " MiB\n", (uint64_t)(ss.bytes_budget / (1024u * 1024u))); vg_tensor_source_close(src); }
    vg_gguf_close(g); return 0;
}
static void plugin_log(int level, const char *message, void *user) { (void)user; fprintf(stderr, "[plugin:%d] %s\n", level, message); }
static int plugin_info(const char *path) {
    VG_PluginHost host = { VG_PLUGIN_ABI_VERSION, sizeof(VG_PluginHost), plugin_log, NULL }; VG_PluginHandle *h = NULL; VG_Status st = vg_plugin_load(path, &host, &h); if (st != VG_OK) { fprintf(stderr, "plugin load failed: %s\n", status_name(st)); return 2; }
    const VG_Plugin *p = vg_plugin(h); printf("plugin: %s %s\nkind: %u\ndescription: %s\n", p->info.name, p->info.version, (unsigned)p->info.kind, p->info.description); vg_plugin_unload(h); return 0;
}
#ifdef VG_HAS_VULKAN
static const char *shader_dir(void) { FILE *f = fopen("shaders/matvec_i8.comp.spv", "rb"); if (f) { fclose(f); return "shaders"; } f = fopen("build/shaders/matvec_i8.comp.spv", "rb"); if (f) { fclose(f); return "build/shaders"; } return "shaders"; }
static int vk_self_test(void) {
    VG_VK *v = NULL; VG_VKConfig cfg; memset(&cfg, 0, sizeof(cfg)); cfg.device_index = UINT32_MAX; cfg.shader_dir = shader_dir();
    VG_Status st = vg_vk_open(&cfg, &v); if (st != VG_OK) { fprintf(stderr, "Vulkan unavailable: %s\n", status_name(st)); return 2; }
    const int8_t w[8] = {1, 2, 3, 4, -1, -2, -3, -4}; const float scales[2] = {1.0f, 0.5f}; const float x[4] = {1, 1, 1, 1}; float y[2] = {0, 0}; VG_VKBuffer *wb = NULL, *sb = NULL;
    st = vg_vk_buffer_upload(v, w, sizeof(w), &wb); if (st == VG_OK) st = vg_vk_buffer_upload(v, scales, sizeof(scales), &sb); if (st == VG_OK) st = vg_vk_matvec_i8(v, wb, sb, x, y, 1, 4, 2);
    printf("vulkan self-test: status=%s y=[%.3f, %.3f]\n", status_name(st), y[0], y[1]); int ok = st == VG_OK && y[0] > 9.99f && y[0] < 10.01f && y[1] > -5.01f && y[1] < -4.99f;
    vg_vk_buffer_release(v, wb); vg_vk_buffer_release(v, sb); vg_vk_close(v); return ok ? 0 : 2;
}
static int vk_info(void) {
    VG_VK *v = NULL; VG_VKConfig cfg; memset(&cfg, 0, sizeof(cfg)); cfg.device_index = UINT32_MAX; cfg.shader_dir = shader_dir(); cfg.prefer_discrete = 1; VG_Status st = vg_vk_open(&cfg, &v); if (st != VG_OK) { fprintf(stderr, "Vulkan unavailable: %s\n", status_name(st)); return 2; }
    VG_VKInfo i; vg_vk_info(v, &i); printf("Vulkan: %s\nAPI: %u.%u.%u\nlocal heap: %.2f GiB\n", i.name, VK_VERSION_MAJOR(i.api_version), VK_VERSION_MINOR(i.api_version), VK_VERSION_PATCH(i.api_version), (double)i.device_local_bytes / (1024.0 * 1024.0 * 1024.0)); vg_vk_close(v); return 0;
}
static int cpu_infer(const char *path, const char *prompt) {
    fprintf(stderr, "CPU native inference: %s\n", path);
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g);
    if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    VG_TensorSourceConfig sc; memset(&sc, 0, sizeof(sc)); sc.host_budget_bytes = 256u * 1024u * 1024u; sc.io_chunk_bytes = 4u * 1024u * 1024u; sc.use_mmap = 1;
    VG_TensorSource *src = NULL; st = vg_tensor_source_open(g, &sc, &src);
    if (st != VG_OK) { fprintf(stderr, "tensor source failed: %s\n", status_name(st)); vg_gguf_close(g); return 2; }
    VG_ModelGraph *graph = NULL; st = vg_model_graph_load(g, src, &graph);
    if (st != VG_OK) { fprintf(stderr, "model graph load failed: %s\n", status_name(st)); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    const VG_ModelConfig *mc = vg_model_graph_config(graph);
    fprintf(stderr, "arch: %u layers, %u embd, %u heads, head_dim=%u, n_head_kv=%u, n_ff=%u, n_rot=%u, eps=%.2e, freq_base=%.1f (CPU only)\n", mc->n_layer, mc->n_embd, mc->n_head, mc->head_dim, mc->n_head_kv, mc->n_ff, mc->n_rot, mc->rms_eps, mc->freq_base);

    VG_Tokenizer *tok = NULL;
    st = vg_tokenizer_load(g, &tok);
    if (st != VG_OK) { fprintf(stderr, "tokenizer load failed: %s\n", status_name(st)); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }

    const char *text = prompt ? prompt : "The capital of France is";
    size_t n_tokens = vg_tokenizer_encode(tok, text, NULL, 0);
    int32_t *tokens = (int32_t *)malloc(n_tokens * sizeof(int32_t));
    if (!tokens) { vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    vg_tokenizer_encode(tok, text, tokens, n_tokens);
    fprintf(stderr, "prompt tokens: %zu\n", n_tokens);

    float *logits = (float *)malloc(mc->n_vocab * sizeof(float));
    if (!logits) { free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }

    /* Prefill: BOS token first, then prompt tokens */
    int32_t kv_pos = 0;
    st = vg_model_graph_decode(graph, mc->bos_id, kv_pos++, logits);
    if (st != VG_OK) { fprintf(stderr, "prefill decode failed at BOS: %s\n", status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    for (size_t i = 0; i < n_tokens; ++i) {
        st = vg_model_graph_decode(graph, tokens[i], kv_pos++, logits);
        if (st != VG_OK) { fprintf(stderr, "prefill decode failed at token %zu: %s\n", i, status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    }

    { int32_t best = 0; float bv = logits[0];
      for (uint32_t i = 1; i < mc->n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
      fprintf(stderr, "post-prefill best=%d val=%.3f\n", best, bv); }

    /* Generate up to 128 tokens */
    uint32_t rng_state = 42u;
    int32_t repeat_last_n = 64;
    int32_t *repeat_buf = (int32_t *)calloc(repeat_last_n, sizeof(int32_t));
    int32_t repeat_count = 0;
    for (int gi = 0; gi < 128; ++gi) {
        int32_t best = sample_token(logits, mc->n_vocab, 0.7f, 40, 0.95f, 1.1f,
                                    repeat_buf, repeat_count, &rng_state);
        if (best == (int32_t)mc->eos_id) break;
        char buf[256]; vg_tokenizer_decode(tok, best, buf, sizeof(buf));
        printf("%s", buf); fflush(stdout);
        if (repeat_count < repeat_last_n) { repeat_buf[repeat_count++] = best; }
        else { memmove(repeat_buf, repeat_buf + 1, (repeat_last_n - 1) * sizeof(int32_t)); repeat_buf[repeat_last_n - 1] = best; }
        st = vg_model_graph_decode(graph, best, kv_pos, logits);
        if (st != VG_OK) { fprintf(stderr, "\ngenerate decode failed at pos %d: %s\n", kv_pos, status_name(st)); break; }
        ++kv_pos;
    }
    printf("\n");
    free(repeat_buf);
    free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g);
    return 0;
}

static int vk_infer(const char *path, const char *prompt) {
    fprintf(stderr, "Vulkan native inference: %s\n", path);
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g);
    if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    VG_TensorSourceConfig sc; memset(&sc, 0, sizeof(sc)); sc.host_budget_bytes = 256u * 1024u * 1024u; sc.io_chunk_bytes = 4u * 1024u * 1024u; sc.use_mmap = 1;
    VG_TensorSource *src = NULL; st = vg_tensor_source_open(g, &sc, &src);
    if (st != VG_OK) { fprintf(stderr, "tensor source failed: %s\n", status_name(st)); vg_gguf_close(g); return 2; }
    VG_ModelGraph *graph = NULL; st = vg_model_graph_load(g, src, &graph);
    if (st != VG_OK) { fprintf(stderr, "model graph load failed: %s\n", status_name(st)); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    VG_VK *v = NULL; VG_VKConfig cfg; memset(&cfg, 0, sizeof(cfg)); cfg.device_index = UINT32_MAX; cfg.shader_dir = shader_dir(); cfg.prefer_discrete = 1;
    st = vg_vk_open(&cfg, &v);
    if (st != VG_OK) { fprintf(stderr, "Vulkan unavailable: %s\n", status_name(st)); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    vg_model_graph_set_vulkan(graph, v);
    VG_VKInfo vi; vg_vk_info(v, &vi);
    const VG_ModelConfig *mc = vg_model_graph_config(graph);
    fprintf(stderr, "GPU: %s\narch: %u layers, %u embd, %u heads, head_dim=%u, n_head_kv=%u, n_ff=%u, n_rot=%u, eps=%.2e, freq_base=%.1f\n", vi.name, mc->n_layer, mc->n_embd, mc->n_head, mc->head_dim, mc->n_head_kv, mc->n_ff, mc->n_rot, mc->rms_eps, mc->freq_base);

    VG_Tokenizer *tok = NULL;
    st = vg_tokenizer_load(g, &tok);
    if (st != VG_OK) { fprintf(stderr, "tokenizer load failed: %s\n", status_name(st)); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }

    const char *text = prompt ? prompt : "The capital of France is";
    size_t n_tokens = vg_tokenizer_encode(tok, text, NULL, 0);
    int32_t *tokens = (int32_t *)malloc(n_tokens * sizeof(int32_t));
    if (!tokens) { vg_tokenizer_free(tok); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    vg_tokenizer_encode(tok, text, tokens, n_tokens);
    fprintf(stderr, "prompt tokens: %zu\n", n_tokens);

    float *logits = (float *)malloc(mc->n_vocab * sizeof(float));
    if (!logits) { free(tokens); vg_tokenizer_free(tok); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }

    /* Prefill: BOS token first, then prompt tokens */
    int32_t kv_pos = 0;
    st = vg_model_graph_decode(graph, mc->bos_id, kv_pos++, logits);
    if (st != VG_OK) { fprintf(stderr, "prefill decode failed at BOS: %s\n", status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    for (size_t i = 0; i < n_tokens; ++i) {
        st = vg_model_graph_decode(graph, tokens[i], kv_pos++, logits);
        if (st != VG_OK) { fprintf(stderr, "prefill decode failed at token %zu: %s\n", i, status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    }

    { int32_t best = 0; float bv = logits[0];
      for (uint32_t i = 1; i < mc->n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
      fprintf(stderr, "post-prefill best=%d val=%.3f\n", best, bv); }

    /* Generate up to 128 tokens */
    uint32_t rng_state = 42u;
    int32_t repeat_last_n = 64;
    int32_t *repeat_buf = (int32_t *)calloc(repeat_last_n, sizeof(int32_t));
    int32_t repeat_count = 0;
    for (int gi = 0; gi < 128; ++gi) {
        int32_t best = sample_token(logits, mc->n_vocab, 0.7f, 40, 0.95f, 1.1f,
                                    repeat_buf, repeat_count, &rng_state);
        if (best == (int32_t)mc->eos_id) break;
        char buf[256]; vg_tokenizer_decode(tok, best, buf, sizeof(buf));
        printf("%s", buf); fflush(stdout);
        if (repeat_count < repeat_last_n) { repeat_buf[repeat_count++] = best; }
        else { memmove(repeat_buf, repeat_buf + 1, (repeat_last_n - 1) * sizeof(int32_t)); repeat_buf[repeat_last_n - 1] = best; }
        st = vg_model_graph_decode(graph, best, kv_pos, logits);
        if (st != VG_OK) { fprintf(stderr, "\ngenerate decode failed at pos %d: %s\n", kv_pos, status_name(st)); break; }
        ++kv_pos;
    }
    printf("\n");

    free(repeat_buf);
    free(logits); free(tokens); vg_tokenizer_free(tok); vg_vk_close(v); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g);
    return 0;
}
#endif
static int tokenize_cmd(const char *path, const char *text) {
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g); if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    VG_Tokenizer *tok = NULL; st = vg_tokenizer_load(g, &tok); if (st != VG_OK) { fprintf(stderr, "tokenizer load failed: %s\n", status_name(st)); vg_gguf_close(g); return 2; }
    size_t n = vg_tokenizer_encode(tok, text, NULL, 0);
    int32_t *ids = (int32_t *)malloc(n * sizeof(int32_t));
    if (!ids) { vg_tokenizer_free(tok); vg_gguf_close(g); return 2; }
    vg_tokenizer_encode(tok, text, ids, n);
    printf("tokens: %zu\n", n);
    for (size_t i = 0; i < n; ++i) {
        char buf[256]; vg_tokenizer_decode(tok, ids[i], buf, sizeof(buf));
        printf("  [%zu] %d %s\n", i, ids[i], buf);
    }
    free(ids); vg_tokenizer_free(tok); vg_gguf_close(g); return 0;
}
static int logits_dump(const char *path, const char *prompt) {
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g);
    if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    VG_TensorSourceConfig sc; memset(&sc, 0, sizeof(sc)); sc.host_budget_bytes = 256u * 1024u * 1024u; sc.io_chunk_bytes = 4u * 1024u * 1024u; sc.use_mmap = 1;
    VG_TensorSource *src = NULL; st = vg_tensor_source_open(g, &sc, &src);
    if (st != VG_OK) { fprintf(stderr, "tensor source failed: %s\n", status_name(st)); vg_gguf_close(g); return 2; }
    VG_ModelGraph *graph = NULL; st = vg_model_graph_load(g, src, &graph);
    if (st != VG_OK) { fprintf(stderr, "model graph load failed: %s\n", status_name(st)); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    const VG_ModelConfig *mc = vg_model_graph_config(graph);
    VG_Tokenizer *tok = NULL; st = vg_tokenizer_load(g, &tok);
    if (st != VG_OK) { fprintf(stderr, "tokenizer load failed: %s\n", status_name(st)); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    const char *text = prompt ? prompt : "The capital of France is";
    size_t n_tokens = vg_tokenizer_encode(tok, text, NULL, 0);
    int32_t *tokens = (int32_t *)malloc(n_tokens * sizeof(int32_t));
    if (!tokens) { vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    vg_tokenizer_encode(tok, text, tokens, n_tokens);
    fprintf(stderr, "prompt: \"%s\"\ntokens: %zu [", text, n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) fprintf(stderr, "%s%d", i ? ", " : "", tokens[i]);
    fprintf(stderr, "]\n");
    float *logits = (float *)malloc(mc->n_vocab * sizeof(float));
    if (!logits) { free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    /* Prefill with BOS */
    int32_t kv_pos = 0;
    st = vg_model_graph_decode(graph, mc->bos_id, kv_pos++, logits);
    if (st != VG_OK) { fprintf(stderr, "BOS decode failed: %s\n", status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    for (size_t i = 0; i < n_tokens; ++i) {
        st = vg_model_graph_decode(graph, tokens[i], kv_pos++, logits);
        if (st != VG_OK) { fprintf(stderr, "prefill failed at token %zu: %s\n", i, status_name(st)); free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g); return 2; }
    }
    /* Top-5 greedy */
    printf("=== Top-5 logits (greedy) ===\n");
    for (int rank = 0; rank < 5; ++rank) {
        int32_t best = 0; float bv = logits[0];
        for (uint32_t i = 1; i < mc->n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = (int32_t)i; }
        char buf[256]; vg_tokenizer_decode(tok, best, buf, sizeof(buf));
        printf("  #%d token=%d \"%.60s\" logit=%.4f\n", rank + 1, best, buf, bv);
        logits[best] = -1e30f;
    }
    free(logits); free(tokens); vg_tokenizer_free(tok); vg_model_graph_free(graph); vg_tensor_source_close(src); vg_gguf_close(g);
    return 0;
}
static int diagnostics(const char *path) {
    VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(path, &g); if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; }
    printf("=== Model Diagnostics: %s ===\n", path);
    printf("file_size: %" PRIu64 " bytes\n", vg_gguf_file_size(g));
    printf("data_base: %" PRIu64 "\n", vg_gguf_data_base(g));
    printf("tensor_count: %" PRIu64 "\n", vg_gguf_tensor_count(g));
    const char *arch = vg_gguf_meta(g, "general.architecture");
    const char *name = vg_gguf_meta(g, "general.name");
    if (arch) printf("architecture: %s\n", arch);
    if (name) printf("name: %s\n", name);
    /* Tensor type distribution */
    uint64_t type_counts[64] = {0}; uint64_t type_bytes[64] = {0};
    for (uint64_t i = 0; i < vg_gguf_tensor_count(g); ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i);
        if (t->ggml_type < 64) { type_counts[t->ggml_type]++; type_bytes[t->ggml_type] += t->nbytes; }
    }
    printf("\ntensor type distribution:\n");
    for (uint32_t i = 0; i < 64; ++i) if (type_counts[i]) printf("  type %u: %" PRIu64 " tensors (%" PRIu64 " bytes)\n", i, type_counts[i], type_bytes[i]);
    /* Tokenizer info */
    const char *tok_model = vg_gguf_meta(g, "tokenizer.ggml.model");
    if (tok_model) printf("\ntokenizer model: %s\n", tok_model);
    VG_Tokenizer *tok = NULL;
    if (vg_tokenizer_load(g, &tok) == VG_OK) { printf("vocab size: %zu\n", vg_tokenizer_vocab_size(tok)); vg_tokenizer_free(tok); }
    /* KV cache sizing estimate */
    uint32_t n_layers = 0, kv_heads = 0, head_dim = 0;
    const char *s;
    s = vg_gguf_meta(g, "llama.attention.layer_count"); if (s) n_layers = (uint32_t)strtol(s, NULL, 10);
    s = vg_gguf_meta(g, "llama.attention.head_count_kv"); if (s) kv_heads = (uint32_t)strtol(s, NULL, 10);
    s = vg_gguf_meta(g, "llama.attention.key_length"); if (s) head_dim = (uint32_t)strtol(s, NULL, 10);
    if (n_layers && kv_heads && head_dim) {
        printf("\nKV shape: n_layers=%u kv_heads=%u head_dim=%u\n", n_layers, kv_heads, head_dim);
        uint64_t per_token = (uint64_t)n_layers * kv_heads * head_dim * 2; /* f16 */
        printf("per-token KV: %" PRIu64 " bytes (%.1f KiB)\n", per_token, (double)per_token / 1024.0);
        printf("128K context estimate: %.1f GiB\n", (double)per_token * 131072.0 / (1024.0 * 1024.0 * 1024.0));
    }
    /* Storage source test */
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg)); cfg.host_budget_bytes = 64u * 1024u * 1024u; cfg.io_chunk_bytes = 4u * 1024u * 1024u; cfg.use_mmap = 1;
    VG_TensorSource *src = NULL; st = vg_tensor_source_open(g, &cfg, &src);
    if (st == VG_OK) {
        VG_TensorSourceStats ss; vg_tensor_source_stats(src, &ss);
        printf("\nstorage source: host_budget=%" PRIu64 " MiB\n", ss.bytes_budget / (1024u * 1024u));
        /* Acquire first tensor to test */
        const VG_GGUF_Tensor *t0 = vg_gguf_tensor_at(g, 0);
        if (t0) { VG_TensorLease l; st = vg_tensor_acquire(src, t0->name, VG_TIER_HOST, &l); if (st == VG_OK) { printf("first tensor '%s': acquired OK (%zu bytes)\n", t0->name, l.size); vg_tensor_release(&l); } else { printf("first tensor '%s': acquire failed: %s\n", t0->name, status_name(st)); } }
        vg_tensor_source_stats(src, &ss); printf("stats: lookups=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " evictions=%" PRIu64 "\n", ss.lookups, ss.hits, ss.misses, ss.evictions);
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g); return 0;
}
static int emit_stdout(const char *piece, size_t bytes, int32_t token, void *user) { (void)token; (void)user; fwrite(piece, 1, bytes, stdout); fflush(stdout); return 1; }
#ifdef VG_BUILD_FULL_ENGINE
static int chat_mode(const char *path, const char *system_prompt) {
    fprintf(stderr, "Chat mode (model: %s)\n", path);
    fprintf(stderr, "Type your message and press Enter. Type 'exit' to quit.\n\n");
    char line[4096];
    for (;;) {
        fprintf(stderr, "You: ");
        fflush(stderr);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        if (len == 0) continue;
        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) break;
        char prompt[8192];
        if (system_prompt && system_prompt[0]) {
            snprintf(prompt, sizeof(prompt), "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", system_prompt, line);
        } else {
            snprintf(prompt, sizeof(prompt), "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", line);
        }
        fprintf(stderr, "Assistant: ");
        VG_GenerateConfig c; memset(&c, 0, sizeof(c)); c.n_predict = 512; c.n_ctx = 4096; c.seed = 0xffffffffu; c.top_k = 40; c.top_p = 0.95f; c.temperature = 0.7f; c.repeat_penalty = 1.1f; c.repeat_last_n = 64; c.use_mmap = 1;
        VG_Status st = vg_generate(path, prompt, &c, emit_stdout, NULL);
        fprintf(stderr, "\n\n");
        if (st != VG_OK) { fprintf(stderr, "generate failed: %s\n", status_name(st)); break; }
    }
    return 0;
}
#endif
int main(int argc, char **argv) {
#ifdef VG_HAS_VULKAN
    if (argc == 2 && strcmp(argv[1], "--vulkan-info") == 0) return vk_info();
    if (argc == 2 && strcmp(argv[1], "--vulkan-self-test") == 0) return vk_self_test();
    if (argc == 3 && strcmp(argv[1], "--vulkan-infer") == 0) return vk_infer(argv[2], NULL);
    if (argc == 4 && strcmp(argv[1], "--vulkan-infer") == 0) return vk_infer(argv[2], argv[3]);
#endif
    if (argc == 3 && strcmp(argv[1], "--cpu-infer") == 0) return cpu_infer(argv[2], NULL);
    if (argc == 4 && strcmp(argv[1], "--cpu-infer") == 0) return cpu_infer(argv[2], argv[3]);
    if (argc == 3 && strcmp(argv[1], "plugin-info") == 0) return plugin_info(argv[2]);
    if (argc == 3 && strcmp(argv[1], "inspect") == 0) return inspect(argv[2]);
    if (argc == 3 && strcmp(argv[1], "diagnostics") == 0) return diagnostics(argv[2]);
    if (argc == 4 && strcmp(argv[1], "tokenize") == 0) return tokenize_cmd(argv[2], argv[3]);
    if (argc == 3 && strcmp(argv[1], "logits") == 0) return logits_dump(argv[2], NULL);
    if (argc == 4 && strcmp(argv[1], "logits") == 0) return logits_dump(argv[2], argv[3]);
#ifdef VG_BUILD_FULL_ENGINE
    if (argc >= 3 && strcmp(argv[1], "chat") == 0) { const char *sys = argc > 3 ? argv[3] : ""; return chat_mode(argv[2], sys); }
#endif
    if (argc == 4 && strcmp(argv[1], "stream") == 0) { VG_GGUF *g = NULL; VG_Status st = vg_gguf_open(argv[2], &g); if (st != VG_OK) { fprintf(stderr, "open failed: %s\n", status_name(st)); return 2; } VG_TensorSource *s = NULL; VG_TensorSourceConfig c; memset(&c, 0, sizeof(c)); c.io_chunk_bytes = 4u * 1024u * 1024u; c.use_mmap = 0; st = vg_tensor_source_open(g, &c, &s); uint64_t total = 0; if (st == VG_OK) st = vg_tensor_stream(s, argv[3], c.io_chunk_bytes, count_chunk, &total); printf("streamed=%" PRIu64 " status=%s\n", total, status_name(st)); vg_tensor_source_close(s); vg_gguf_close(g); return st == VG_OK ? 0 : 2; }
    fprintf(stderr, "usage: %s --vulkan-info | --vulkan-self-test | --vulkan-infer MODEL.gguf [PROMPT] | inspect MODEL.gguf | diagnostics MODEL.gguf | tokenize MODEL.gguf TEXT | stream MODEL.gguf TENSOR | plugin-info PLUGIN\n", argv[0]); return 1;
}
