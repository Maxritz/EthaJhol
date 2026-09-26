#include "vg/full_engine.h"
#include "vg/gguf.h"
#include "vg/tensor_source.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int emit(const char *piece, size_t bytes, int32_t token, void *user) { (void)token; (void)user; std::fwrite(piece, 1, bytes, stdout); std::fflush(stdout); return 1; }
static void usage(const char *p) { std::fprintf(stderr, "usage: %s -m MODEL.gguf [-p PROMPT] [-n TOKENS] [-c CONTEXT] [--temp T] [--top-k K] [--top-p P] [--threads N] [--seed N] [--no-mmap] [--ngl LAYERS]\n", p); }
int main(int argc, char **argv) {
    const char *model = nullptr; const char *prompt = "Hello";
    VG_GenerateConfig c{};
    c.n_predict = 128; c.n_ctx = 4096; c.seed = 0xffffffffu;
    c.top_k = 40; c.top_p = 0.95f; c.temperature = 0.8f;
    c.repeat_penalty = 1.1f; c.repeat_last_n = 64;
    c.use_mmap = 1; c.n_threads = 0;
    c.use_gpu = 0; c.ngl_layers = 0;
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
        else { usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }

    /* Open native tensor source for readahead prefetch. */
    VG_GGUF *gguf = nullptr; VG_TensorSource *src = nullptr;
    if (vg_gguf_open(model, &gguf) == VG_OK) {
        VG_TensorSourceConfig cfg{}; cfg.host_budget_bytes = 256u * 1024u * 1024u; cfg.io_chunk_bytes = 4u * 1024u * 1024u; cfg.use_mmap = 1; cfg.prefetch_depth = 4;
        if (vg_tensor_source_open(gguf, &cfg, &src) == VG_OK) {
            c.use_mmap = 1;
        } else {
            vg_gguf_close(gguf); gguf = nullptr;
        }
    }

    /* Set up prefetch_source if we have a tensor source */
    // Note: prefetch_source field removed from config for simplicity

    VG_Status st = vg_generate(model, prompt, &c, emit, nullptr);
    std::fputc('\n', stdout);
    std::fprintf(stderr, "[dbg] run_main: fputc ok\n");
    if (src) vg_tensor_source_close(src);
    std::fprintf(stderr, "[dbg] run_main: src closed\n");
    if (gguf) vg_gguf_close(gguf);
    std::fprintf(stderr, "[dbg] run_main: gguf closed\n");
    return st == VG_OK ? 0 : 2;
}
