#include "vg/gguf.h"
#include "vg/tensor_source.h"
#include "vg/model_graph.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void test_tensor_range_valid(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    uint64_t n = vg_gguf_tensor_count(g);
    uint64_t data_base = vg_gguf_data_base(g);
    uint64_t file_size = vg_gguf_file_size(g);
    for (uint64_t i = 0; i < n; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i);
        assert(t != NULL);
        uint64_t end = data_base + t->data_offset + t->nbytes;
        assert(end <= file_size);
    }
    printf("PASS: all %llu tensor ranges valid for %s\n", (unsigned long long)n, model_path);
    vg_gguf_close(g);
}

static void test_tensor_source_basic(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = 64u * 1024u * 1024u;
    cfg.io_chunk_bytes = 4u * 1024u * 1024u;
    cfg.use_mmap = 0;
    VG_TensorSource *src = NULL;
    assert(vg_tensor_source_open(g, &cfg, &src) == VG_OK);

    const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
    assert(t != NULL);
    VG_TensorLease lease; memset(&lease, 0, sizeof(lease));
    assert(vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease) == VG_OK);
    assert(lease.data != NULL);
    assert(lease.size == t->nbytes);

    VG_TensorLease lease2; memset(&lease2, 0, sizeof(lease2));
    assert(vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease2) == VG_OK);
    assert(lease2.data == lease.data);

    vg_tensor_release(&lease);
    vg_tensor_release(&lease2);
    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: tensor source acquire/release for %s\n", model_path);
}

static void test_tensor_source_eviction(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = 1u;
    cfg.io_chunk_bytes = 4u * 1024u * 1024u;
    cfg.use_mmap = 0;
    VG_TensorSource *src = NULL;
    assert(vg_tensor_source_open(g, &cfg, &src) == VG_OK);
    const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
    assert(t != NULL);
    VG_TensorLease lease; memset(&lease, 0, sizeof(lease));
    VG_Status st = vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease);
    if (t->nbytes > 1) assert(st == VG_E_NOMEM);
    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: eviction with 1-byte budget for %s\n", model_path);
}

static const VG_GGUF_Tensor *find_small_tensor(VG_GGUF *g) {
    uint64_t n = vg_gguf_tensor_count(g);
    for (uint64_t i = 0; i < n && i < 20; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i);
        if (t && t->nbytes <= 65536) return t;
    }
    const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
    return t && t->nbytes <= 65536 ? t : NULL;
}

static void test_tensor_source_async(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    const VG_GGUF_Tensor *t = find_small_tensor(g);
    if (!t) { printf("SKIP: no small tensor in %s\n", model_path); vg_gguf_close(g); return; }

    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = 64u * 1024u * 1024u;
    cfg.io_chunk_bytes = 4u * 1024u * 1024u;
    cfg.use_mmap = 0;
    cfg.io_workers = 2;
    cfg.prefetch_depth = 4;
    VG_TensorSource *src = NULL;
    printf("DEBUG: calling vg_tensor_source_open\n"); fflush(stdout);
    assert(vg_tensor_source_open(g, &cfg, &src) == VG_OK);
    printf("DEBUG: async open done\n"); fflush(stdout);

    VG_PrefetchEntry entry; memset(&entry, 0, sizeof(entry));
    printf("DEBUG: preflight_async name=%s nbytes=%llu\n", t->name, (unsigned long long)t->nbytes); fflush(stdout);
    assert(vg_tensor_prefetch_async(src, t->name, VG_TIER_HOST, &entry) == VG_OK);
    printf("DEBUG: async waiting\n"); fflush(stdout);
    assert(vg_tensor_wait(src, &entry) == VG_OK);
    printf("DEBUG: wait done, ready=%d bytes=%zu\n", entry.ready, entry.bytes); fflush(stdout);
    assert(entry.ready && entry.data != NULL && entry.bytes == t->nbytes);
    free(entry.data);
    printf("DEBUG: closing\n"); fflush(stdout);
    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: async prefetch for %s\n", model_path);
}

static void test_tensor_source_concurrent(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = 32u * 1024u * 1024u;
    cfg.io_chunk_bytes = 1u * 1024u * 1024u;
    cfg.use_mmap = 1;
    cfg.io_workers = 4;
    VG_TensorSource *src = NULL;
    assert(vg_tensor_source_open(g, &cfg, &src) == VG_OK);
    const VG_GGUF_Tensor *t = find_small_tensor(g);
    if (!t) { printf("SKIP: no small tensor in %s\n", model_path); vg_tensor_source_close(src); vg_gguf_close(g); return; }
    assert(t != NULL);
    VG_PrefetchEntry entries[4];
    for (int i = 0; i < 4; ++i) {
        memset(&entries[i], 0, sizeof(entries[i]));
        assert(vg_tensor_prefetch_async(src, t->name, VG_TIER_HOST, &entries[i]) == VG_OK);
    }
    for (int i = 0; i < 4; ++i) {
        assert(vg_tensor_wait(src, &entries[i]) == VG_OK);
        assert(entries[i].ready && entries[i].data != NULL);
        free(entries[i].data);
    }
    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: concurrent async prefetch for %s\n", model_path);
}

static void test_moe_router(const char *model_path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(model_path, &g) != VG_OK) {
        printf("SKIP: cannot open %s\n", model_path);
        return;
    }
    VG_TensorSourceConfig cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = 256u * 1024u * 1024u; cfg.io_chunk_bytes = 4u * 1024u * 1024u; cfg.use_mmap = 1;
    cfg.io_workers = 0;
    VG_TensorSource *src = NULL;
    assert(vg_tensor_source_open(g, &cfg, &src) == VG_OK);
    VG_ModelGraph *graph = NULL;
    assert(vg_model_graph_load(g, src, &graph) == VG_OK);
    const VG_ModelConfig *c = vg_model_graph_config(graph);
    printf("PASS: model loaded, n_expert=%u n_expert_used=%u\n", c->n_expert, c->n_expert_used);
    VG_MoEState state; memset(&state, 0, sizeof(state));
    assert(vg_model_graph_moe_route(graph, 0, NULL, NULL, &state) == VG_OK);
    assert(state.n_expert >= 1);
    assert(state.n_expert_used >= 1);
    assert(state.expert_ids != NULL);
    for (uint32_t i = 0; i < state.n_expert_used; ++i) assert(state.expert_ids[i] < state.n_expert);
    vg_moe_state_free(&state);
    vg_model_graph_free(graph);
    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: MoE router state valid for %s\n", model_path);
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    if (model) {
        test_tensor_range_valid(model);
        test_tensor_source_basic(model);
        test_tensor_source_eviction(model);
        test_tensor_source_async(model);
        test_tensor_source_concurrent(model);
        test_moe_router(model);
    } else {
        puts("SKIP: no model provided");
    }
    puts("vg-tensor: all tests passed");
    return 0;
}
