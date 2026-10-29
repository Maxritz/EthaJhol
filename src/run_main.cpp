#include "vg/full_engine.h"
#include "vg/gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static std::chrono::steady_clock::time_point g_t_first, g_t_last;
static uint64_t g_t_n = 0;

static int emit(const char *piece, size_t bytes, int32_t token, void *user) {
    (void)token; (void)user;
    auto now = std::chrono::steady_clock::now();
    if (!g_t_n) g_t_first = now;
    g_t_last = now;
    ++g_t_n;
    std::fwrite(piece, 1, bytes, stdout);
    std::fflush(stdout);
    return 1;
}

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

/* Llama-3 chat framing. A system turn is included when system_text is non-NULL.
 * Returns a heap string (caller frees) or nullptr on OOM. */
static char *chat_template_llama3(const char *user_text, const char *system_text) {
    const char *tpl_sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
    const char *tpl_user = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
    const char *tpl = system_text ? tpl_sys : tpl_user;
    size_t need = (size_t)std::snprintf(nullptr, 0, tpl, system_text ? system_text : user_text, user_text) + 1;
    char *buf = (char *)std::malloc(need);
    if (!buf) return nullptr;
    if (system_text) std::snprintf(buf, need, tpl_sys, system_text, user_text);
    else std::snprintf(buf, need, tpl_user, user_text);
    return buf;
}

static bool is_instruct(const VG_GGUF *g) {
    const char *name = vg_gguf_meta(g, "general.name");
    if (name) {
        if (strstr(name, "Instruct") || strstr(name, "Chat") ||
            strstr(name, "chat") || strstr(name, "Qwen") ||
            strstr(name, "Gemma") || strstr(name, "gemma") ||
            strstr(name, "Phi") || strstr(name, "phi") ||
            strstr(name, "Mistral") || strstr(name, "mistral"))
            return true;
    }
    return vg_gguf_meta(g, "tokenizer.chat_template") != nullptr;
}

/* The Llama-3 chat template is only meaningful when the vocabulary actually contains its
 * control tokens; a 32k SentencePiece vocab (Mistral etc.) would see "<|start_header_id|>"
 * shredded by BPE into noise. */
static bool has_llama3_vocab(const VG_GGUF *g) {
    const char *tokens = vg_gguf_meta(g, "tokenizer.ggml.tokens");
    return tokens && strstr(tokens, "<|start_header_id|>") != nullptr;
}

static void usage(const char *p) {
    std::fprintf(stderr,
        "usage: %s -m MODEL.gguf [-p PROMPT | -f FILE] [-n TOKENS] [-c CONTEXT]\n"
        "          [--temp T] [--top-k K] [--top-p P] [--threads N] [--seed N]\n"
        "          [--no-mmap] [--ngl LAYERS] [--system S]\n"
        "          [--lora ADAPTER.gguf] [--lora-scale S]\n", p);
}

int main(int argc, char **argv) {
    const char *model = nullptr;
    const char *prompt = "Hello";
    const char *prompt_file = nullptr;
    const char *system_prompt = nullptr; /* nullptr => default text for instruct models */

    VG_GenerateConfig c{};
    c.n_predict = 128; c.n_ctx = 4096; c.seed = 0xffffffffu;
    c.top_k = 40; c.top_p = 0.95f; c.temperature = 0.8f;
    c.repeat_penalty = 1.1f; c.repeat_last_n = 64;
    c.use_mmap = 1; c.n_threads = 0;
    c.use_gpu = 0; c.ngl_layers = 0;
    c.lora_path = nullptr; c.lora_scale = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "-f") && i + 1 < argc) prompt_file = argv[++i];
        else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) c.n_predict = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-c") && i + 1 < argc) c.n_ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--temp") && i + 1 < argc) c.temperature = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--top-k") && i + 1 < argc) c.top_k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--top-p") && i + 1 < argc) c.top_p = std::strtof(argv[++i], nullptr);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) c.n_threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) c.seed = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--no-mmap")) { c.use_mmap = 0; }
        else if (!std::strcmp(argv[i], "--ngl") && i + 1 < argc) { c.ngl_layers = std::atoi(argv[++i]); c.use_gpu = (c.ngl_layers > 0) ? 1 : 0; }
        else if (!std::strcmp(argv[i], "--lora") && i + 1 < argc) { c.lora_path = argv[++i]; }
        else if (!std::strcmp(argv[i], "--lora-scale") && i + 1 < argc) { c.lora_scale = std::strtof(argv[++i], nullptr); }
        else if (!std::strcmp(argv[i], "--system") && i + 1 < argc) system_prompt = argv[++i];
        else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!model) { std::fprintf(stderr, "no model given\n"); usage(argv[0]); return 1; }

    char *prompt_buf = nullptr;
    if (prompt_file) {
        FILE *fp = std::fopen(prompt_file, "rb");
        if (!fp) { std::fprintf(stderr, "cannot open prompt file: %s\n", prompt_file); return 2; }
        /* Grow-while-reading: no ftell, so a >4 GiB offset cannot be truncated to 32 bits. */
        size_t cap = 4096, len = 0;
        char *buf = (char *)std::malloc(cap);
        if (!buf) { std::fclose(fp); return 2; }
        for (;;) {
            if (len + 4096 + 1 > cap) {
                size_t ncap = cap * 2;
                char *nb = (char *)std::realloc(buf, ncap);
                if (!nb) { std::free(buf); std::fclose(fp); return 2; }
                buf = nb; cap = ncap;
            }
            size_t got = std::fread(buf + len, 1, 4096, fp);
            len += got;
            if (got < 4096) break;
        }
        buf[len] = 0;
        std::fclose(fp);
        prompt_buf = buf;
        prompt = prompt_buf;
    }

    /* Auto-frame instruct models so an Instruct-tuned model sees real turn structure
     * instead of a raw completion prompt. A prompt that already starts with the BOS
     * token text is left alone. */
    const char *framed = prompt;
    char *framed_alloc = nullptr;
    if (prompt && prompt[0] &&
        !(prompt[0] == '<' && std::strncmp(prompt, "<|begin_of_text", 15) == 0)) {
        VG_GGUF *meta = nullptr;
        if (vg_gguf_open(model, &meta) == VG_OK && meta) {
            if (is_instruct(meta) && has_llama3_vocab(meta)) {
                const char *sys = system_prompt;
                if (!sys) sys = "You are a helpful, conversationally-fluent assistant.";
                char *w = chat_template_llama3(prompt, sys);
                if (w) { framed = w; framed_alloc = w; }
            }
            vg_gguf_close(meta);
        }
    }

    const VG_Status st = vg_generate(model, framed, &c, emit, nullptr);
    if (g_t_n > 1) {
        double secs = std::chrono::duration<double>(g_t_last - g_t_first).count();
        std::fprintf(stderr, "[vg] TG: %llu tokens in %.3fs = %.1f t/s\n",
                     (unsigned long long)g_t_n, secs,
                     (double)(g_t_n - 1) / (secs > 0 ? secs : 1e-9));
    }
    std::fputc('\n', stdout);
    std::free(prompt_buf);
    std::free(framed_alloc);
    if (st != VG_OK) {
        std::fprintf(stderr, "generation failed: %s\n", st_name(st));
        return 2;
    }
    return 0;
}
