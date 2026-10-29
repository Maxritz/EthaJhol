#include "vg/kv_cache.h"

#include <stdio.h>
#include <string.h>

/* Explicit checks, not assert(): NDEBUG removes assert() bodies, and every call in this file
 * used to live inside one, so a Release build ran an empty test that always passed. The
 * compiler said so too — "unused variable 'page'/'bytes'/'matched'/'snapshot'". */
static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
} while (0)

#define CHECK_OK(expr) do { \
    VG_Status s_ = (expr); \
    if (s_ != VG_OK) { fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, (int)s_); ++failures; } \
} while (0)

static void test_kv_and_radix(void) {
    VG_KVConfig c;
    memset(&c, 0, sizeof(c));
    c.page_tokens = 4; c.n_layers = 2; c.kv_heads = 2; c.head_dim = 8;
    c.element_bytes = 2; c.host_budget_bytes = 1u << 20; c.max_pages = 64;

    VG_KVStore *store = NULL;
    CHECK_OK(vg_kv_open(&c, &store));
    if (!store) return;

    VG_KVSequence *seq = NULL;
    CHECK_OK(vg_kv_sequence_init(store, &seq));
    if (!seq) { vg_kv_close(store); return; }

    for (uint64_t i = 0; i < 10; ++i) CHECK_OK(vg_kv_sequence_append(seq, 100 + i));
    CHECK(vg_kv_sequence_length(seq) == 10);

    /* 10 tokens with 4 per page must land in 3 pages. */
    VG_KVStats after_append;
    vg_kv_stats(store, &after_append);
    CHECK(after_append.page_allocs == 3);

    void *page = NULL;
    size_t bytes = 0;
    CHECK_OK(vg_kv_page_acquire(seq, 1, 9, VG_TIER_HOST, &page, &bytes));
    CHECK(page != NULL);
    CHECK(bytes == 2u * 8u * 2u * 4u);   /* kv_heads * head_dim * element_bytes * page_tokens */

    VG_KVSequence *fork = NULL;
    CHECK_OK(vg_kv_sequence_fork(seq, 8, &fork));
    if (fork) {
        CHECK(vg_kv_sequence_length(fork) == 8);
        CHECK(memcmp(vg_kv_sequence_tokens(seq), vg_kv_sequence_tokens(fork),
                     8 * sizeof(uint64_t)) == 0);
    }

    VG_Radix *radix = NULL;
    CHECK_OK(vg_radix_open(store, &radix));
    if (radix) {
        CHECK_OK(vg_radix_insert(radix, seq, 10, 0x1234, 0));
        size_t matched = 0;
        VG_KVSequence *snapshot = NULL;
        CHECK_OK(vg_radix_lookup(radix, vg_kv_sequence_tokens(seq), 10, &matched, &snapshot));
        CHECK(matched == 10);
        CHECK(snapshot != NULL);
        if (snapshot) CHECK(vg_kv_sequence_length(snapshot) == 10);

        /* A non-matching prefix must not claim a hit. */
        uint64_t other[3] = { 999, 1000, 1001 };
        size_t matched2 = 123;
        VG_KVSequence *snap2 = NULL;
        CHECK_OK(vg_radix_lookup(radix, other, 3, &matched2, &snap2));
        CHECK(matched2 == 0);
        CHECK(snap2 == NULL);
        vg_radix_close(radix);
    }

    VG_KVStats st;
    vg_kv_stats(store, &st);
    CHECK(st.prefix_hits == 1);
    CHECK(st.page_allocs >= 3);

    if (fork) vg_kv_sequence_close(fork);
    vg_kv_sequence_close(seq);
    vg_kv_close(store);
}

int main(void) {
    test_kv_and_radix();
    if (failures) { fprintf(stderr, "vg-core: %d failure(s)\n", failures); return 1; }
    puts("vg-core: all tests passed");
    return 0;
}
