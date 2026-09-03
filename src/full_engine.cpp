#include "vg/full_engine.h"
#include <llama.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static VG_GenerateConfig defaults() {
    VG_GenerateConfig c{}; c.n_predict = 128; c.n_ctx = 4096; c.n_batch = 512; c.n_gpu_layers = 99; c.seed = LLAMA_DEFAULT_SEED; c.top_k = 40; c.top_p = 0.95f; c.temperature = 0.8f; c.repeat_penalty = 1.1f; c.repeat_last_n = 64; c.lazy_mode = 2; c.load_mode = LLAMA_LOAD_MODE_MMAP; c.use_mmap = 1; c.n_threads = 0; c.n_threads_batch = 0; c.n_seq_max = 1; c.lora_path = nullptr; c.lora_scale = 1.0f; c.prefetch_source = nullptr; return c;
}
extern "C" VG_Status vg_generate(const char *model_path, const char *prompt, const VG_GenerateConfig *config, VG_TokenCallback callback, void *user) {
    if (!model_path || !prompt || !callback) return VG_E_INVALID;
    VG_GenerateConfig c = config ? *config : defaults(); if (c.n_predict < 1 || c.n_ctx < 1 || c.n_batch < 1) return VG_E_INVALID;
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = c.n_gpu_layers; mp.lazy_mode = static_cast<llama_lazy_mode>(c.lazy_mode < 0 ? LLAMA_LAZY_MODE_AUTO : c.lazy_mode); mp.load_mode = static_cast<llama_load_mode>(c.load_mode < 0 ? LLAMA_LOAD_MODE_AUTO : c.load_mode); if (!c.use_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model *model = llama_model_load_from_file(model_path, mp); if (!model) return VG_E_IO;

    /* Prefetch first-layer tensors through the native source if provided. */
    if (c.prefetch_source) {
        static const char *prefetch_names[] = { "token_embd.weight", "blk.0.attn.weight", "blk.0.ffn.weight", "output.weight" };
        for (size_t i = 0; i < sizeof(prefetch_names)/sizeof(prefetch_names[0]); ++i) {
            VG_TensorLease l; VG_Status ps = vg_tensor_acquire(c.prefetch_source, prefetch_names[i], VG_TIER_HOST, &l);
            if (ps == VG_OK) vg_tensor_release(&l);
        }
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, prompt, std::strlen(prompt), nullptr, 0, true, true); if (n_prompt <= 0) { llama_model_free(model); return VG_E_FORMAT; }
    std::vector<llama_token> tokens(static_cast<size_t>(n_prompt)); if (llama_tokenize(vocab, prompt, std::strlen(prompt), tokens.data(), tokens.size(), true, true) < 0) { llama_model_free(model); return VG_E_FORMAT; }
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = static_cast<uint32_t>(std::max(c.n_ctx, n_prompt + c.n_predict)); cp.n_batch = static_cast<uint32_t>(std::max(c.n_batch, n_prompt)); cp.n_threads = c.n_threads; cp.n_threads_batch = c.n_threads_batch; cp.n_seq_max = static_cast<uint32_t>(std::max(1, c.n_seq_max));
    llama_context *ctx = llama_init_from_model(model, cp); if (!ctx) { llama_model_free(model); return VG_E_NOMEM; }
    llama_adapter_lora *adapter = nullptr; llama_adapter_lora *adapters[1] = {nullptr}; float scales[1] = {c.lora_scale == 0.0f ? 1.0f : c.lora_scale};
    if (c.lora_path) { adapter = llama_adapter_lora_init(model, c.lora_path); if (!adapter) { llama_free(ctx); llama_model_free(model); return VG_E_IO; } adapters[0] = adapter; if (llama_set_adapters_lora(ctx, adapters, 1, scales) != 0) { llama_adapter_lora_free(adapter); llama_free(ctx); llama_model_free(model); return VG_E_UNSUPPORTED; } }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params(); llama_sampler *smpl = llama_sampler_chain_init(sp); if (!smpl) { llama_free(ctx); llama_model_free(model); return VG_E_NOMEM; }
    if (c.repeat_penalty != 1.0f && c.repeat_last_n > 0) llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), c.repeat_last_n, c.repeat_penalty, 0.0f, 0.0f));
    if (c.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(c.top_k));
    if (c.top_p > 0.0f && c.top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(c.top_p, 1));
    if (c.temperature > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_temp(c.temperature));
    llama_sampler_chain_add(smpl, c.seed == LLAMA_DEFAULT_SEED ? llama_sampler_init_dist(LLAMA_DEFAULT_SEED) : llama_sampler_init_dist(c.seed));
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    VG_Status status = VG_OK;
    for (int32_t generated = 0; generated < c.n_predict; ++generated) {
        if (llama_decode(ctx, batch) != 0) { status = VG_E_IO; break; }
        llama_token id = llama_sampler_sample(smpl, ctx, -1); if (llama_vocab_is_eog(vocab, id)) break;
        char buf[4096]; int32_t n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true); if (n < 0) { status = VG_E_FORMAT; break; }
        if (!callback(buf, static_cast<size_t>(n), id, user)) break;
        batch = llama_batch_get_one(&id, 1);
    }
    llama_sampler_free(smpl); if (adapter) llama_adapter_lora_free(adapter); llama_free(ctx); llama_model_free(model); return status;
}
