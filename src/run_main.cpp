#include "vg/full_engine.h"
#include "vg/gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int emit(const char *piece, size_t bytes, int32_t token, void *user) {
    (void)token; (void)user;
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

static void usage(const char *p) {
    std::fprintf(stderr,
        "usage: %s -m MODEL.gguf [-p PROMPT] [-n TOKENS] [-c CONTEXT] [--temp T] [--top-k K]\n"
        "          [--top-p P] [--threads N] [--seed N] [--no-mmap] [--ngl LAYERS]\n"
        "          [--lora ADAPTER.gguf] [--lora-scale S]\n", p);
}

int main(int argc, char **argv) {
    const char *model = nullptr;
    const char *prompt = "Hello";
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
        else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!model) { std::fprintf(stderr, "no model given\n"); usage(argv[0]); return 1; }

    const VG_Status st = vg_generate(model, prompt, &c, emit, nullptr);
    std::fputc('\n', stdout);
    if (st != VG_OK) {
        std::fprintf(stderr, "generation failed: %s\n", st_name(st));
        return 2;
    }
    return 0;
}
