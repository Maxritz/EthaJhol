/* Tensor-source / GGUF geometry tests.
 *
 * These used assert() for both checks and side effects. Release builds define NDEBUG, so every
 * assertion (and every call made inside one) vanished: the suite reported success while doing
 * nothing, and an unfinished async prefetch later wrote into a dead stack frame. Checks are now
 * explicit so they run in every configuration. */
#include "vg/gguf.h"
#include "vg/model_graph.h"
#include "vg/tensor_source.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } \
} while (0)

#define CHECK_ST(expr, want) do { \
    VG_Status _st = (expr); \
    if (_st != (want)) { \
        fprintf(stderr, "FAIL %s:%d: %s -> %d (want %d)\n", __FILE__, __LINE__, #expr, (int)_st, (int)(want)); \
        ++g_failures; \
    } \
} while (0)

static VG_TensorSource *open_source(VG_GGUF *g, size_t budget, int use_mmap, unsigned io_workers) {
    VG_TensorSourceConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.host_budget_bytes = budget;
    cfg.io_chunk_bytes = 1u * 1024u * 1024u;
    cfg.use_mmap = use_mmap;
    cfg.io_workers = io_workers;
    VG_TensorSource *src = NULL;
    if (vg_tensor_source_open(g, &cfg, &src) != VG_OK) return NULL;
    return src;
}

static void test_tensor_range_valid(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { fprintf(stderr, "FAIL: cannot open %s\n", path); ++g_failures; return; }
    const uint64_t n = vg_gguf_tensor_count(g);
    CHECK(n > 0);
    const uint64_t data_base = vg_gguf_data_base(g);
    const uint64_t file_size = vg_gguf_file_size(g);
    for (uint64_t i = 0; i < n; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i);
        CHECK(t != NULL);
        if (!t) continue;
        const uint64_t end = data_base + t->data_offset + t->nbytes;
        CHECK(end <= file_size);
        CHECK(t->nbytes > 0);
    }
    /* Every declared tensor must be findable by name (hash index + linear fallback agree). */
    for (uint64_t i = 0; i < n; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, i);
        if (!t) continue;
        CHECK(vg_gguf_find_tensor(g, t->name) == t);
    }
    vg_gguf_close(g);
    printf("PASS: %llu tensor ranges valid in %s\n", (unsigned long long)n, path);
}

static void test_tensor_source_basic(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 64u * 1024u * 1024u, 0, 0);
    CHECK(src != NULL);
    if (!src) { vg_gguf_close(g); return; }

    const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
    CHECK(t != NULL);
    if (t) {
        VG_TensorLease lease;
        memset(&lease, 0, sizeof(lease));
        CHECK_ST(vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease), VG_OK);
        CHECK(lease.data != NULL);
        CHECK(lease.size == t->nbytes);

        /* A second lease of the same tensor must hit the cache, not re-read the file. */
        VG_TensorLease lease2;
        memset(&lease2, 0, sizeof(lease2));
        CHECK_ST(vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease2), VG_OK);
        CHECK(lease2.data == lease.data);

        VG_TensorSourceStats stats;
        vg_tensor_source_stats(src, &stats);
        CHECK(stats.hits >= 1);
        CHECK(stats.bytes_resident >= t->nbytes);

        vg_tensor_release(&lease);
        vg_tensor_release(&lease2);
        CHECK(lease.data == NULL && lease2.data == NULL);
    }

    /* A missing tensor is an error, not a crash. */
    VG_TensorLease missing;
    memset(&missing, 0, sizeof(missing));
    CHECK(vg_tensor_acquire(src, "does.not.exist", VG_TIER_HOST, &missing) != VG_OK);
    CHECK(missing.data == NULL);

    vg_tensor_source_close(src);
    vg_gguf_close(g);
    printf("PASS: tensor source acquire/release/cache\n");
}

static void test_tensor_source_eviction(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 1, 0, 0);
    CHECK(src != NULL);
    if (src) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
        CHECK(t != NULL);
        if (t) {
            VG_TensorLease lease;
            memset(&lease, 0, sizeof(lease));
            VG_Status st = vg_tensor_acquire(src, t->name, VG_TIER_HOST, &lease);
            /* A tensor larger than the whole budget still loads, as a transient lease. */
            if (t->nbytes > 1) CHECK(st == VG_OK);
            if (st == VG_OK) vg_tensor_release(&lease);

            VG_TensorSourceStats stats;
            vg_tensor_source_stats(src, &stats);
            CHECK(stats.bytes_resident == 0); /* nothing could be retained */
        }
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: budget eviction\n");
}

/* mmap-backed storage lease: the pointer must land inside the mapping and stay valid across
 * releases, which is what the graph and the GPU weight cache rely on. */
static void test_tensor_source_mmap(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 1u * 1024u * 1024u, 1, 0);
    CHECK(src != NULL);
    if (src) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
        CHECK(t != NULL);
        if (t) {
            VG_TensorLease a, b;
            memset(&a, 0, sizeof(a));
            memset(&b, 0, sizeof(b));
            CHECK_ST(vg_tensor_acquire(src, t->name, VG_TIER_STORAGE, &a), VG_OK);
            CHECK(a.tier == VG_TIER_STORAGE);
            const void *first = a.data;
            vg_tensor_release(&a);
            CHECK_ST(vg_tensor_acquire(src, t->name, VG_TIER_STORAGE, &b), VG_OK);
            CHECK(b.data == first); /* stable address: no copy, no relocation */
            vg_tensor_release(&b);
        }
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: mmap storage lease\n");
}

static void test_tensor_source_async(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 64u * 1024u * 1024u, 0, 2);
    CHECK(src != NULL);
    if (src) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
        CHECK(t != NULL);
        if (t) {
            VG_PrefetchEntry entry;
            memset(&entry, 0, sizeof(entry));
            CHECK_ST(vg_tensor_prefetch_async(src, t->name, VG_TIER_HOST, &entry), VG_OK);
            CHECK_ST(vg_tensor_wait(src, &entry), VG_OK);
            CHECK(entry.ready == 1);
            CHECK(entry.data != NULL);
            CHECK(entry.bytes == t->nbytes);
            free(entry.data);
            entry.data = NULL;
        }
        /* Every request must be settled before close: outstanding ones reference caller memory. */
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: async prefetch\n");
}

static void test_tensor_source_concurrent(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 32u * 1024u * 1024u, 1, 4);
    CHECK(src != NULL);
    if (src) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
        CHECK(t != NULL);
        if (t) {
            VG_PrefetchEntry entries[4];
            int submitted = 0;
            for (int i = 0; i < 4; ++i) {
                memset(&entries[i], 0, sizeof(entries[i]));
                if (vg_tensor_prefetch_async(src, t->name, VG_TIER_HOST, &entries[i]) == VG_OK) ++submitted;
            }
            CHECK(submitted == 4);
            for (int i = 0; i < 4; ++i) {
                if (!entries[i].ready && !entries[i].name) continue;
                CHECK_ST(vg_tensor_wait(src, &entries[i]), VG_OK);
                CHECK(entries[i].data != NULL);
                free(entries[i].data);
                entries[i].data = NULL;
            }
        }
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: concurrent async prefetch\n");
}

static void test_tensor_source_bad_names(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 4u * 1024u * 1024u, 1, 1);
    CHECK(src != NULL);
    if (src) {
        VG_PrefetchEntry entry;
        memset(&entry, 0, sizeof(entry));
        CHECK(vg_tensor_prefetch_async(src, "not.a.tensor", VG_TIER_HOST, &entry) == VG_OK);
        CHECK(vg_tensor_wait(src, &entry) == VG_E_IO); /* reported, not silently ignored */
        CHECK(entry.error == 1);
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: async error reporting\n");
}

static void test_read_and_stream(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
    CHECK(t != NULL);
    if (t) {
        unsigned char *buf = (unsigned char *)malloc((size_t)t->nbytes);
        CHECK(buf != NULL);
        if (buf) {
            CHECK_ST(vg_gguf_read(g, t, 0, buf, (size_t)t->nbytes), VG_OK);
            /* Out-of-range reads are rejected instead of reading past the tensor. */
            CHECK(vg_gguf_read(g, t, 1, buf, (size_t)t->nbytes) == VG_E_RANGE);
            CHECK(vg_gguf_read(g, t, 0, buf, (size_t)t->nbytes + 1u) == VG_E_RANGE);
            free(buf);
        }
    }
    vg_gguf_close(g);
    printf("PASS: tensor read bounds\n");
}

typedef struct VG_StreamCount {
    size_t bytes;
    size_t calls;
} VG_StreamCount;

static VG_Status stream_count(const void *chunk, size_t bytes, uint64_t off, void *user) {
    (void)chunk;
    (void)off;
    VG_StreamCount *c = (VG_StreamCount *)user;
    c->bytes += bytes;
    ++c->calls;
    return VG_OK;
}

static void test_stream_chunked(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 1u * 1024u * 1024u, 1, 0);
    CHECK(src != NULL);
    if (src) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(g, 0);
        CHECK(t != NULL);
        if (t) {
            VG_StreamCount c;
            memset(&c, 0, sizeof(c));
            CHECK_ST(vg_tensor_stream(src, t->name, 512, stream_count, &c), VG_OK);
            CHECK(c.bytes == (size_t)t->nbytes);
            CHECK(c.calls == (size_t)((t->nbytes + 511u) / 512u));
        }
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: chunked streaming\n");
}

static void test_moe_router(const char *path) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open(path, &g) != VG_OK) { ++g_failures; return; }
    VG_TensorSource *src = open_source(g, 256u * 1024u * 1024u, 1, 0);
    CHECK(src != NULL);
    if (src) {
        VG_ModelGraph *graph = NULL;
        CHECK_ST(vg_model_graph_load(g, src, &graph), VG_OK);
        CHECK(graph != NULL);
        if (graph) {
            const VG_ModelConfig *c = vg_model_graph_config(graph);
            CHECK(c != NULL);
            if (c) {
                printf("model: layers=%u experts=%u/%u\n", c->n_layer, c->n_expert, c->n_expert_used);
                VG_MoEState state;
                memset(&state, 0, sizeof(state));
                CHECK_ST(vg_model_graph_moe_route(graph, 0, NULL, NULL, &state), VG_OK);
                CHECK(state.n_expert >= 1);
                CHECK(state.n_expert_used >= 1);
                CHECK(state.expert_ids != NULL);
                for (uint32_t i = 0; i < state.n_expert_used && state.expert_ids; ++i)
                    CHECK(state.expert_ids[i] < state.n_expert);
                vg_moe_state_free(&state);
                CHECK(state.expert_ids == NULL);
            }
            vg_model_graph_free(graph);
        }
        vg_tensor_source_close(src);
    }
    vg_gguf_close(g);
    printf("PASS: graph load + MoE router state\n");
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    if (!model) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 1;
    }
    test_tensor_range_valid(model);
    test_read_and_stream(model);
    test_tensor_source_basic(model);
    test_tensor_source_eviction(model);
    test_tensor_source_mmap(model);
    test_tensor_source_async(model);
    test_tensor_source_concurrent(model);
    test_tensor_source_bad_names(model);
    test_stream_chunked(model);
    test_moe_router(model);

    if (g_failures) {
        fprintf(stderr, "vg-tensor: %d failure(s)\n", g_failures);
        return 1;
    }
    puts("vg-tensor: all tests passed");
    return 0;
}
