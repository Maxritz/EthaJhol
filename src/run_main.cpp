#include "vg/full_engine.h"
#include "vg/gguf.h"
#include "vg/tensor_source.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int emit(const char *piece, size_t bytes, int32_t token, void *user) { (void)token; (void)user; std::fwrite(piece, 1, bytes, stdout); std::fflush(stdout); return 1; }
static void usage(const char *p) { std::fprintf(stderr, "usage: %s -m MODEL.gguf [-p PROMPT] [-n TOKENS] [-c CONTEXT] [-ngl LAYERS] [--temp T] [--top-k K] [--top-p P] [--threads N] [--lora ADAPTER] [--lora-scale S] [--eager|--no-mmap|--native]\n", p); }
int main(int argc, char **argv) {
    const char *model = nullptr; const char *prompt = "Hello"; VG_GenerateConfig c{}; c.n_predict = 128; c.n_ctx = 4096; c.n_batch = 512; c.n_gpu_layers = 99; c.seed = 0xffffffffu; c.top_k = 40; c.top_p = 0.95f; c.temperature = 0.8f; c.repeat_penalty = 1.1f; c.repeat_last_n = 64; c.lazy_mode = 2; c.load_mode = 1; c.use_mmap = 1; c.n_threads = 0; c.n_threads_batch = 0; c.n_seq_max = 1; c.lora_path = nullptr; c.lora_scale = 1.0f; c.prefetch_source = nullptr;
    int use_native = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i]; else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i]; else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) c.n_predict = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "-c") && i + 1 < argc) c.n_ctx = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "-ngl") && i + 1 < argc) c.n_gpu_layers = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--temp") && i + 1 < argc) c.temperature = std::strtof(argv[++i], nullptr); else if (!std::strcmp(argv[i], "--top-k") && i + 1 < argc) c.top_k = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--top-p") && i + 1 < argc) c.top_p = std::strtof(argv[++i], nullptr); else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) c.n_threads = std::atoi(argv[++i]); else if (!std::strcmp(argv[i], "--lora") && i + 1 < argc) c.lora_path = argv[++i]; else if (!std::strcmp(argv[i], "--lora-scale") && i + 1 < argc) c.lora_scale = std::strtof(argv[++i], nullptr); else if (!std::strcmp(argv[i], "--no-mmap")) { c.use_mmap = 0; c.load_mode = 0; c.lazy_mode = 0; } else if (!std::strcmp(argv[i], "--eager")) { c.lazy_mode = 0; } else if (!std::strcmp(argv[i], "--native")) { use_native = 1; } else { usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }

    /* Open native tensor source for readahead prefetch. */
    VG_GGUF *gguf = nullptr; VG_TensorSource *src = nullptr;
    if (vg_gguf_open(model, &gguf) == VG_OK) {
        VG_TensorSourceConfig cfg{}; cfg.host_budget_bytes = 256u * 1024u * 1024u; cfg.io_chunk_bytes = 4u * 1024u * 1024u; cfg.use_mmap = 1; cfg.prefetch_depth = 4;
        if (vg_tensor_source_open(gguf, &cfg, &src) == VG_OK) c.prefetch_source = src;
    }

    VG_Status st;
    if (use_native) {
        st = vg_generate_native(model, prompt, &c, emit, nullptr);
    } else {
        st = vg_generate(model, prompt, &c, emit, nullptr);
    }
    std::fputc('\n', stdout);
    if (src) vg_tensor_source_close(src);
    if (gguf) vg_gguf_close(gguf);
    return st == VG_OK ? 0 : 2;
}
