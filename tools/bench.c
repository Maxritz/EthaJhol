#include "vg/kv_cache.h"
#include "vg/trace.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    size_t n = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 100000;
    VG_KVConfig c; memset(&c, 0, sizeof(c)); c.page_tokens = 128; c.n_layers = 32; c.kv_heads = 8; c.head_dim = 128; c.element_bytes = 2; c.max_pages = (uint32_t)((n + c.page_tokens - 1) / c.page_tokens + 4);
    VG_KVStore *store = NULL; VG_KVSequence *seq = NULL; VG_Radix *radix = NULL; if (vg_kv_open(&c, &store) || vg_kv_sequence_init(store, &seq) || vg_radix_open(store, &radix)) return 2;
    uint64_t t0 = vg_trace_now_ns(); for (size_t i = 0; i < n; ++i) if (vg_kv_sequence_append(seq, (uint64_t)(i & 0xffffu))) return 2; uint64_t t1 = vg_trace_now_ns();
    if (vg_radix_insert(radix, seq, n, 0xfeedbeef, 0)) return 2; size_t matched = 0; VG_KVSequence *snapshot = NULL; uint64_t t2 = vg_trace_now_ns(); vg_radix_lookup(radix, vg_kv_sequence_tokens(seq), n, &matched, &snapshot); uint64_t t3 = vg_trace_now_ns();
    VG_KVStats st; vg_kv_stats(store, &st); printf("tokens=%zu append_ms=%.3f lookup_ms=%.3f matched=%zu pages=%" PRIu64 " host_gib=%.3f\n", n, (double)(t1 - t0) / 1e6, (double)(t3 - t2) / 1e6, matched, st.page_allocs, (double)st.host_bytes / (1024.0 * 1024.0 * 1024.0));
    vg_radix_close(radix); vg_kv_sequence_close(seq); vg_kv_close(store); return 0;
}
