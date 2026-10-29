#include "vg/kv_cache.h"
#include "vg/trace.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Measures paged-KV append/radix costs. The default is intentionally small: each page holds
 * n_layers * kv_heads * head_dim * page_tokens * element_bytes, so 100k tokens at 32x8x128
 * would need tens of GiB. */
int main(int argc, char **argv) {
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 4096;
    VG_KVConfig c;
    memset(&c, 0, sizeof(c));
    c.page_tokens = 128;
    c.n_layers = 32;
    c.kv_heads = 8;
    c.head_dim = 128;
    c.element_bytes = 2;
    c.max_pages = (uint32_t)((n + c.page_tokens - 1) / c.page_tokens + 4);

    VG_KVStore *store = NULL;
    VG_KVSequence *seq = NULL;
    VG_Radix *radix = NULL;
    VG_Status st = vg_kv_open(&c, &store);
    if (st != VG_OK) { fprintf(stderr, "kv_open failed: %d\n", (int)st); return 2; }
    if ((st = vg_kv_sequence_init(store, &seq)) != VG_OK) { fprintf(stderr, "sequence_init failed: %d\n", (int)st); vg_kv_close(store); return 2; }
    if ((st = vg_radix_open(store, &radix)) != VG_OK) { fprintf(stderr, "radix_open failed: %d\n", (int)st); vg_kv_sequence_close(seq); vg_kv_close(store); return 2; }

    const uint64_t t0 = vg_trace_now_ns();
    for (size_t i = 0; i < n; ++i) {
        st = vg_kv_sequence_append(seq, (uint64_t)(i & 0xffffu));
        if (st != VG_OK) { fprintf(stderr, "append failed at %zu: %d (out of memory?)\n", i, (int)st); break; }
    }
    const uint64_t t1 = vg_trace_now_ns();

    st = vg_radix_insert(radix, seq, vg_kv_sequence_length(seq), 0xfeedbeef, 0);
    size_t matched = 0;
    VG_KVSequence *snapshot = NULL;
    const uint64_t t2 = vg_trace_now_ns();
    VG_Status ls = vg_radix_lookup(radix, vg_kv_sequence_tokens(seq), vg_kv_sequence_length(seq), &matched, &snapshot);
    const uint64_t t3 = vg_trace_now_ns();

    VG_KVStats stats;
    vg_kv_stats(store, &stats);
    printf("tokens=%zu append_ms=%.3f insert=%d lookup=%d lookup_ms=%.3f matched=%zu pages=%" PRIu64 " host_gib=%.3f\n",
           vg_kv_sequence_length(seq), (double)(t1 - t0) / 1e6, (int)st, (int)ls,
           (double)(t3 - t2) / 1e6, matched, stats.page_allocs,
           (double)stats.host_bytes / (1024.0 * 1024.0 * 1024.0));

    vg_radix_close(radix);
    vg_kv_sequence_close(seq);
    vg_kv_close(store);
    return 0;
}
