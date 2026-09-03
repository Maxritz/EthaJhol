#ifndef VG_TENSOR_SOURCE_H
#define VG_TENSOR_SOURCE_H

#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VG_Tier {
    VG_TIER_STORAGE = 0,
    VG_TIER_HOST = 1,
    VG_TIER_PINNED = 2,
    VG_TIER_DEVICE = 3
} VG_Tier;

typedef struct VG_TensorSource VG_TensorSource;
typedef struct VG_TensorLease {
    const VG_GGUF_Tensor *tensor;
    const void *data;
    size_t size;
    VG_Tier tier;
    uint64_t ticket;
    VG_TensorSource *source;
} VG_TensorLease;

typedef struct VG_TensorSourceStats {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetches;
    uint64_t prefetch_hits;
    uint64_t bytes_read;
    uint64_t bytes_resident;
    uint64_t bytes_device;
    uint64_t bytes_budget;
    uint64_t device_budget;
    uint64_t evictions;
} VG_TensorSourceStats;

typedef struct VG_TensorSourceConfig {
    size_t host_budget_bytes;
    size_t pinned_budget_bytes;
    size_t device_budget_bytes;
    size_t io_chunk_bytes;
    unsigned io_workers;
    unsigned prefetch_depth;
    int use_mmap;
    int use_direct_io;
} VG_TensorSourceConfig;

VG_Status vg_tensor_source_open(VG_GGUF *file, const VG_TensorSourceConfig *cfg,
                                VG_TensorSource **out);
void vg_tensor_source_close(VG_TensorSource *source);
VG_Status vg_tensor_acquire(VG_TensorSource *source, const char *name,
                            VG_Tier desired, VG_TensorLease *out);
void vg_tensor_release(VG_TensorLease *lease);
VG_Status vg_tensor_prefetch(VG_TensorSource *source, const char *const *names,
                             size_t count, VG_Tier desired);
void vg_tensor_source_stats(const VG_TensorSource *source, VG_TensorSourceStats *out);

/* One-shot streaming read used by kernels that fuse dequantize with compute. */
VG_Status vg_tensor_stream(VG_TensorSource *source, const char *name,
                           size_t chunk_bytes,
                           VG_Status (*consume)(const void *, size_t, uint64_t, void *),
                           void *user);

/* Prefetch the next N tensors after a given tensor by index order. Non-blocking, best-effort. */
VG_Status vg_tensor_prefetch_after(VG_TensorSource *source, const char *name, unsigned depth);

#ifdef __cplusplus
}
#endif
#endif
