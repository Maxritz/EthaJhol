#include "vg/kv_cache.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void test_kv_and_radix(void) {
    VG_KVConfig c; memset(&c, 0, sizeof(c)); c.page_tokens = 4; c.n_layers = 2; c.kv_heads = 2; c.head_dim = 8; c.element_bytes = 2; c.host_budget_bytes = 1u << 20; c.max_pages = 64;
    VG_KVStore *store = NULL; assert(vg_kv_open(&c, &store) == VG_OK); VG_KVSequence *seq = NULL; assert(vg_kv_sequence_init(store, &seq) == VG_OK);
    for (uint64_t i = 0; i < 10; ++i) assert(vg_kv_sequence_append(seq, 100 + i) == VG_OK); assert(vg_kv_sequence_length(seq) == 10);
    void *page = NULL; size_t bytes = 0; assert(vg_kv_page_acquire(seq, 1, 9, VG_TIER_HOST, &page, &bytes) == VG_OK); assert(page && bytes > 0);
    VG_KVSequence *fork = NULL; assert(vg_kv_sequence_fork(seq, 8, &fork) == VG_OK); assert(vg_kv_sequence_length(fork) == 8); assert(memcmp(vg_kv_sequence_tokens(seq), vg_kv_sequence_tokens(fork), 8 * sizeof(uint64_t)) == 0);
    VG_Radix *radix = NULL; assert(vg_radix_open(store, &radix) == VG_OK); assert(vg_radix_insert(radix, seq, 10, 0x1234, 0) == VG_OK);
    size_t matched = 0; VG_KVSequence *snapshot = NULL; assert(vg_radix_lookup(radix, vg_kv_sequence_tokens(seq), 10, &matched, &snapshot) == VG_OK); assert(matched == 10 && snapshot != NULL);
    VG_KVStats st; vg_kv_stats(store, &st); assert(st.prefix_hits == 1 && st.page_allocs >= 3);
    vg_kv_sequence_close(fork); vg_kv_sequence_close(seq); vg_radix_close(radix); vg_kv_close(store);
}
int main(void) { test_kv_and_radix(); puts("vg-core: all tests passed"); return 0; }
