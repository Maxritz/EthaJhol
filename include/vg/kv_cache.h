#ifndef VG_KV_CACHE_H
#define VG_KV_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include "tensor_source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_KVStore VG_KVStore;
typedef struct VG_Radix VG_Radix;
typedef struct VG_KVSequence VG_KVSequence;

typedef struct VG_KVConfig {
    uint32_t page_tokens;
    uint32_t n_layers;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t element_bytes;
    uint64_t host_budget_bytes;
    uint64_t device_budget_bytes;
    uint32_t max_pages;
    const char *spill_dir; /* optional: directory for page spill files; NULL disables spill */
} VG_KVConfig;

typedef struct VG_KVStats {
    uint64_t page_allocs;
    uint64_t page_hits;
    uint64_t page_faults;
    uint64_t page_evictions;
    uint64_t host_bytes;
    uint64_t device_bytes;
    uint64_t prefix_lookups;
    uint64_t prefix_hits;
    uint64_t prefix_tokens_reused;
} VG_KVStats;

VG_Status vg_kv_open(const VG_KVConfig *cfg, VG_KVStore **out);
void vg_kv_close(VG_KVStore *store);
VG_Status vg_kv_sequence_init(VG_KVStore *store, VG_KVSequence **out);
void vg_kv_sequence_close(VG_KVSequence *seq);
VG_Status vg_kv_sequence_append(VG_KVSequence *seq, uint64_t token_id);
VG_Status vg_kv_sequence_fork(VG_KVSequence *src, size_t prefix_tokens,
                              VG_KVSequence **out);
size_t vg_kv_sequence_length(const VG_KVSequence *seq);
const uint64_t *vg_kv_sequence_tokens(const VG_KVSequence *seq);

/* A page is writable only while owned by one live sequence. Once inserted into the radix cache,
 * its contents become immutable and may be shared by forks. */
VG_Status vg_kv_page_acquire(VG_KVSequence *seq, uint32_t layer, size_t token,
                             VG_Tier desired, void **data, size_t *bytes);
void vg_kv_page_release(VG_KVSequence *seq, uint32_t layer, size_t token);

VG_Status vg_radix_open(VG_KVStore *store, VG_Radix **out);
void vg_radix_close(VG_Radix *radix);
VG_Status vg_radix_lookup(VG_Radix *radix, const uint64_t *tokens, size_t count,
                          size_t *matched_tokens, VG_KVSequence **snapshot);
VG_Status vg_radix_insert(VG_Radix *radix, VG_KVSequence *seq, size_t token_count,
                          uint64_t semantic_fingerprint, int tainted);
void vg_radix_invalidate(VG_Radix *radix, uint64_t model_identity,
                         uint64_t adapter_identity);
void vg_kv_stats(const VG_KVStore *store, VG_KVStats *out);

#ifdef __cplusplus
}
#endif
#endif
