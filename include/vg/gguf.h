#ifndef VG_GGUF_H
#define VG_GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VG_Status {
    VG_OK = 0,
    VG_E_INVALID = -1,
    VG_E_IO = -2,
    VG_E_FORMAT = -3,
    VG_E_RANGE = -4,
    VG_E_NOMEM = -5,
    VG_E_UNSUPPORTED = -6,
    VG_E_BUSY = -7
} VG_Status;

typedef struct VG_GGUF VG_GGUF;
typedef struct VG_GGUF_Tensor {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    uint64_t data_offset;
    uint64_t nbytes;
} VG_GGUF_Tensor;

/* Opens only the GGUF header, KV metadata, and tensor index. Tensor data is lazy.
 * Not thread-safe: one VG_GGUF instance must not be accessed from multiple threads
 * concurrently. Thread-safe access requires external synchronization. */
VG_Status vg_gguf_open(const char *path, VG_GGUF **out);
void vg_gguf_close(VG_GGUF *file);
void vg_gguf_set_direct_io(VG_GGUF *file, int enable);
const char *vg_gguf_path(const VG_GGUF *file);
uint64_t vg_gguf_data_base(const VG_GGUF *file);
uint64_t vg_gguf_file_size(const VG_GGUF *file);
uint64_t vg_gguf_tensor_count(const VG_GGUF *file);
const VG_GGUF_Tensor *vg_gguf_tensor_at(const VG_GGUF *file, uint64_t index);
const VG_GGUF_Tensor *vg_gguf_find_tensor(const VG_GGUF *file, const char *name);

/* Metadata values are returned as immutable strings; numeric values are canonical text. */
const char *vg_gguf_meta(const VG_GGUF *file, const char *key);

/* Demand-read a byte range from a tensor. No tensor-sized temporary is created. */
VG_Status vg_gguf_read(const VG_GGUF *file, const VG_GGUF_Tensor *tensor,
                       uint64_t tensor_offset, void *dst, size_t nbytes);

/* Maps the whole GGUF file read-only. This reserves virtual address space only; the OS pages data on demand. */
VG_Status vg_gguf_map(const VG_GGUF *file, const void **base, uint64_t *size);
void vg_gguf_unmap(const VG_GGUF *file);

#ifdef __cplusplus
}
#endif
#endif
