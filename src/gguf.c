#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "vg/gguf.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

typedef struct VG_Meta { char *key; char *value; } VG_Meta;

typedef struct VG_HashEntry { const char *key; uint32_t index; int occupied; } VG_HashEntry;

struct VG_GGUF {
    char *path;
    FILE *index;
    uint64_t file_size;
    uint64_t data_base;
    uint64_t tensor_count;
    VG_GGUF_Tensor *tensors;
    VG_Meta *meta;
    size_t meta_count;
    VG_HashEntry *tensor_hash;
    size_t tensor_hash_cap;
    VG_HashEntry *meta_hash;
    size_t meta_hash_cap;
    int direct_io;
#ifdef _WIN32
    HANDLE map_file;
    HANDLE map_object;
#else
    int map_fd;
#endif
    void *map_base;
    uint64_t map_size;
};

static int rd(void *p, size_t n, FILE *f) { return n == fread(p, 1, n, f) ? 1 : 0; }
static int u32(FILE *f, uint32_t *v) { return rd(v, sizeof(*v), f); }
static int u64(FILE *f, uint64_t *v) { return rd(v, sizeof(*v), f); }
static int seek64(FILE *f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)off, SEEK_SET) == 0;
#endif
}
static uint64_t tell64(FILE *f) {
#ifdef _WIN32
    return (uint64_t)_ftelli64(f);
#else
    return (uint64_t)ftello(f);
#endif
}
static uint64_t align_up_u64(uint64_t x, uint64_t a) {
    return a && x <= UINT64_MAX - (a - 1u) ? (x + a - 1u) / a * a : 0;
}
#ifndef _WIN32
static char *dupstr(const char *s) { size_t n = strlen(s) + 1u; char *p = (char *)malloc(n); if (p) memcpy(p, s, n); return p; }
#define _strdup dupstr
#endif
static char *read_string(FILE *f) {
    uint64_t n;
    if (!u64(f, &n) || n > (1u << 20)) return NULL;
    char *s = (char *)malloc((size_t)n + 1u);
    if (!s || !rd(s, (size_t)n, f)) { free(s); return NULL; }
    s[n] = 0;
    return s;
}
static char *fmt_u64(uint64_t v) {
    char b[32]; (void)snprintf(b, sizeof(b), "%llu", (unsigned long long)v);
    return _strdup(b);
}
static char *fmt_i64(int64_t v) {
    char b[32]; (void)snprintf(b, sizeof(b), "%lld", (long long)v);
    return _strdup(b);
}
static char *fmt_f64(double v) {
    char b[64]; (void)snprintf(b, sizeof(b), "%.17g", v);
    return _strdup(b);
}

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) { h ^= *p; h *= 16777619u; }
    return h;
}
static VG_HashEntry *hash_find(VG_HashEntry *table, size_t cap, const char *key) {
    if (!cap) return NULL;
    uint32_t i = fnv1a(key) % cap;
    for (size_t probes = 0; probes < cap; ++probes) {
        VG_HashEntry *e = &table[i];
        if (!e->occupied) return e;
        if (strcmp(e->key, key) == 0) return e;
        i = (i + 1) % cap;
    }
    return NULL;
}
static int hash_insert(VG_HashEntry **table, size_t *cap, const char *key, uint32_t index) {
    if (*cap == 0) { *cap = 64; *table = (VG_HashEntry *)calloc(*cap, sizeof(VG_HashEntry)); if (!*table) return 0; }
    size_t threshold = *cap * 3 / 4;
    size_t count = 0; for (size_t i = 0; i < *cap; ++i) if ((*table)[i].occupied) ++count;
    if (count >= threshold) {
        size_t nc = *cap * 2; VG_HashEntry *nt = (VG_HashEntry *)calloc(nc, sizeof(VG_HashEntry)); if (!nt) return 0;
        for (size_t i = 0; i < *cap; ++i) { if ((*table)[i].occupied) { VG_HashEntry *e = hash_find(nt, nc, (*table)[i].key); e->key = (*table)[i].key; e->index = (*table)[i].index; e->occupied = 1; } }
        free(*table); *table = nt; *cap = nc;
    }
    VG_HashEntry *e = hash_find(*table, *cap, key);
    if (e->occupied) return 0;
    e->key = key; e->index = index; e->occupied = 1; return 1;
}

static int skip_value(FILE *f, uint32_t type);
static char *read_scalar_value(FILE *f, uint32_t type);
static char *read_array_text(FILE *f);
static int skip_array(FILE *f) {
    uint32_t type; uint64_t n;
    if (!u32(f, &type) || !u64(f, &n) || n > (1ull << 32)) return 0;
    for (uint64_t i = 0; i < n; ++i) if (!skip_value(f, type)) return 0;
    return 1;
}
static char *read_array_text(FILE *f) {
    uint32_t type; uint64_t n; size_t cap = 64u, len = 0;
    if (!u32(f, &type) || !u64(f, &n) || n > (1ull << 20)) return NULL;
    char *out = (char *)malloc(cap); if (!out) return NULL; out[0] = '[';
    for (uint64_t i = 0; i < n; ++i) {
        char *item = read_scalar_value(f, type); if (!item) { free(out); return NULL; }
        size_t ilen = strlen(item);
        /* For string arrays, wrap each element in quotes to match JSON array format */
        int quoted = (type == 8);
        size_t extra = ilen + (i ? 1u : 0u) + 2u + (quoted ? 2u : 0u);
        if (len + extra >= cap) { while (len + extra >= cap) { if (cap > SIZE_MAX / 2u) { free(item); free(out); return NULL; } cap *= 2u; } char *grown = (char *)realloc(out, cap); if (!grown) { free(item); free(out); return NULL; } out = grown; }
        if (i) out[++len] = ',';
        if (quoted) {
            out[++len] = '"';
            /* Escape double quotes and backslashes in string elements */
            for (size_t j = 0; j < ilen; ++j) {
                if (item[j] == '"' || item[j] == '\\') {
                    if (len + 3 >= cap) { while (len + 3 >= cap) { if (cap > SIZE_MAX / 2u) break; cap *= 2u; } char *grown = (char *)realloc(out, cap); if (!grown) { free(item); free(out); return NULL; } out = grown; }
                    out[++len] = '\\';
                }
                out[++len] = item[j];
            }
            out[++len] = '"';
        } else { memcpy(out + len + 1u, item, ilen); len += 1u + ilen; }
        free(item);
    }
    out[len + 1u] = ']'; out[len + 2u] = 0; return out;
}
static int skip_value(FILE *f, uint32_t type) {
    uint8_t b[8];
    switch (type) {
        case 0: case 1: case 7: return rd(b, 1, f);
        case 2: case 3: return rd(b, 2, f);
        case 4: case 5: case 6: return rd(b, 4, f);
        case 10: case 11: case 12: return rd(b, 8, f);
        case 8: return read_string(f) != NULL;
        case 9: return skip_array(f);
        default: return 0;
    }
}
static char *read_scalar_value(FILE *f, uint32_t type) {
    uint8_t b8; uint16_t b16; uint32_t b32; uint64_t b64; int8_t i8; int16_t i16; int32_t i32; int64_t i64; float fl; double db;
    switch (type) {
        case 0: if (!rd(&b8, 1, f)) return NULL; return fmt_u64(b8);
        case 1: if (!rd(&i8, 1, f)) return NULL; return fmt_i64(i8);
        case 2: if (!rd(&b16, 2, f)) return NULL; return fmt_u64(b16);
        case 3: if (!rd(&i16, 2, f)) return NULL; return fmt_i64(i16);
        case 4: if (!rd(&b32, 4, f)) return NULL; return fmt_u64(b32);
        case 5: if (!rd(&i32, 4, f)) return NULL; return fmt_i64(i32);
        case 6: if (!rd(&fl, 4, f)) return NULL; return fmt_f64(fl);
        case 7: if (!rd(&b8, 1, f)) return NULL; return _strdup(b8 ? "true" : "false");
        case 8: return read_string(f);
        case 10: if (!rd(&b64, 8, f)) return NULL; return fmt_u64(b64);
        case 11: if (!rd(&i64, 8, f)) return NULL; return fmt_i64(i64);
        case 12: if (!rd(&db, 8, f)) return NULL; return fmt_f64(db);
        case 9: return read_array_text(f);
        default: return NULL;
    }
}
static int quant_layout(uint32_t type, uint32_t *block, uint32_t *type_size) {
    *block = 1; *type_size = 0;
    switch (type) {
        case 0: *type_size = 4; return 1;             /* F32 */
        case 1: *type_size = 2; return 1;             /* F16 */
        case 2: *block = 32; *type_size = 18; return 1; /* Q4_0 */
        case 3: *block = 32; *type_size = 20; return 1; /* Q4_1 */
        case 4: *type_size = 8; return 1;             /* F64 */
        case 6: *block = 32; *type_size = 22; return 1; /* Q5_0 */
        case 7: *block = 32; *type_size = 24; return 1; /* Q5_1 */
        case 8: *block = 32; *type_size = 34; return 1; /* Q8_0 */
        case 9: *block = 32; *type_size = 36; return 1; /* Q8_1 */
        case 10: *block = 256; *type_size = 84; return 1; /* Q2_K */
        case 11: *block = 256; *type_size = 110; return 1; /* Q3_K */
        case 12: *block = 256; *type_size = 144; return 1; /* Q4_K */
        case 13: *block = 256; *type_size = 176; return 1; /* Q5_K */
        case 14: *block = 256; *type_size = 210; return 1; /* Q6_K */
        case 15: *block = 256; *type_size = 292; return 1; /* Q8_K */
        case 16: *block = 256; *type_size = 66; return 1;  /* IQ2_XXS */
        case 17: *block = 256; *type_size = 74; return 1;  /* IQ2_XS */
        case 18: *block = 256; *type_size = 98; return 1;  /* IQ3_XXS */
        case 19: *block = 256; *type_size = 50; return 1;  /* IQ1_S */
        case 20: *block = 32; *type_size = 18; return 1;   /* IQ4_NL */
        case 21: *block = 256; *type_size = 110; return 1; /* IQ3_S */
        case 22: *block = 256; *type_size = 82; return 1;  /* IQ2_S */
        case 23: *block = 256; *type_size = 136; return 1; /* IQ4_XS */
        case 24: *type_size = 1; return 1;             /* I8 */
        case 25: *type_size = 2; return 1;             /* I16 */
        case 26: *type_size = 4; return 1;             /* I32 */
        case 27: *type_size = 8; return 1;             /* I64 */
        case 28: *type_size = 8; return 1;             /* F64 (alt) */
        case 29: *block = 256; *type_size = 56; return 1;  /* IQ1_M */
        case 30: *type_size = 2; return 1;             /* BF16 */
        case 39: *block = 32; *type_size = 17; return 1;   /* MXFP4 */
        case 40: *block = 64; *type_size = 36; return 1;   /* NVFP4 */
        case 41: *block = 32; *type_size = 18; return 1;   /* Q1_0 */
        case 42: *block = 64; *type_size = 18; return 1;   /* Q2_0 */
        default: return 0;
    }
}
static VG_Status parse_index(VG_GGUF *g) {
    uint32_t magic, version; uint64_t kv_count;
    if (!u32(g->index, &magic) || magic != 0x46554747u || !u32(g->index, &version) || version < 2 || version > 3 ||
        !u64(g->index, &g->tensor_count) || !u64(g->index, &kv_count)) return VG_E_FORMAT;
    if (g->tensor_count > (1ull << 24) || kv_count > (1ull << 24)) return VG_E_RANGE;
    for (uint64_t i = 0; i < kv_count; ++i) {
        char *key = read_string(g->index); uint32_t type; char *value;
        if (!key || !u32(g->index, &type) || !(value = read_scalar_value(g->index, type))) { free(key); return VG_E_FORMAT; }
        VG_Meta *m = (VG_Meta *)realloc(g->meta, (g->meta_count + 1u) * sizeof(*m));
        if (!m) { free(key); free(value); return VG_E_NOMEM; }
        g->meta = m; g->meta[g->meta_count].key = key; g->meta[g->meta_count].value = value; ++g->meta_count;
    }
    uint64_t off = tell64(g->index); uint64_t table_end = off;
    g->tensors = (VG_GGUF_Tensor *)calloc((size_t)g->tensor_count, sizeof(*g->tensors));
    if (!g->tensors && g->tensor_count) return VG_E_NOMEM;
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        VG_GGUF_Tensor *t = &g->tensors[i]; char *name = read_string(g->index); uint32_t nd;
        if (!name || !u32(g->index, &nd) || nd > 4) { free(name); return VG_E_FORMAT; }
        t->name = name; t->n_dims = nd;
        for (uint32_t d = 0; d < nd; ++d) if (!u64(g->index, &t->dims[d]) || t->dims[d] == 0) return VG_E_FORMAT;
        if (!u32(g->index, &t->ggml_type) || !u64(g->index, &t->data_offset)) return VG_E_FORMAT;
        uint32_t block, ts; uint64_t rows = 1, nfast = nd ? t->dims[0] : 1;
        for (uint32_t d = 1; d < nd; ++d) { if (rows > UINT64_MAX / t->dims[d]) return VG_E_RANGE; rows *= t->dims[d]; }
        if (!quant_layout(t->ggml_type, &block, &ts) || nfast % block || rows > UINT64_MAX / (nfast / block) ||
            rows * (nfast / block) > UINT64_MAX / ts) return VG_E_UNSUPPORTED;
        t->nbytes = rows * (nfast / block) * ts; table_end = tell64(g->index);
    }
    uint64_t alignment = 32;
    const char *a = vg_gguf_meta(g, "general.alignment"); if (a) { unsigned long long x = strtoull(a, NULL, 10); if (x >= 1 && x <= 4096) alignment = (uint64_t)x; }
    g->data_base = align_up_u64(table_end, alignment); if (!g->data_base) return VG_E_RANGE;
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        VG_GGUF_Tensor *t = &g->tensors[i];
        if (t->data_offset > g->file_size - g->data_base || t->nbytes > g->file_size - g->data_base - t->data_offset) return VG_E_RANGE;
    }
    g->tensor_hash_cap = g->tensor_count > 64 ? g->tensor_count : 64;
    g->tensor_hash = (VG_HashEntry *)calloc(g->tensor_hash_cap, sizeof(VG_HashEntry));
    if (!g->tensor_hash && g->tensor_count) return VG_E_NOMEM;
    for (uint64_t i = 0; i < g->tensor_count; ++i) hash_insert(&g->tensor_hash, &g->tensor_hash_cap, g->tensors[i].name, (uint32_t)i);
    g->meta_hash_cap = g->meta_count > 64 ? g->meta_count : 64;
    g->meta_hash = (VG_HashEntry *)calloc(g->meta_hash_cap, sizeof(VG_HashEntry));
    if (!g->meta_hash && g->meta_count) return VG_E_NOMEM;
    for (size_t i = 0; i < g->meta_count; ++i) hash_insert(&g->meta_hash, &g->meta_hash_cap, g->meta[i].key, (uint32_t)i);
    (void)off; return VG_OK;
}

VG_Status vg_gguf_open(const char *path, VG_GGUF **out) {
    if (!path || !out) return VG_E_INVALID; *out = NULL;
    FILE *f = fopen(path, "rb"); if (!f) return VG_E_IO;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return VG_E_IO; }
    uint64_t size = tell64(f); if (size < 24 || !seek64(f, 0)) { fclose(f); return VG_E_FORMAT; }
    VG_GGUF *g = (VG_GGUF *)calloc(1, sizeof(*g)); if (!g) { fclose(f); return VG_E_NOMEM; }
    g->path = _strdup(path); g->index = f; g->file_size = size;
#ifndef _WIN32
    g->map_fd = -1;
#else
    g->map_file = INVALID_HANDLE_VALUE; g->map_object = NULL;
#endif
    VG_Status s = parse_index(g); if (s != VG_OK) { vg_gguf_close(g); return s; }
    *out = g; return VG_OK;
}
void vg_gguf_close(VG_GGUF *g) {
    if (!g) return; vg_gguf_unmap(g); if (g->index) fclose(g->index);
    for (uint64_t i = 0; i < g->tensor_count; ++i) free((void *)g->tensors[i].name);
    for (size_t i = 0; i < g->meta_count; ++i) { free(g->meta[i].key); free(g->meta[i].value); }
    free(g->tensors); free(g->meta); free(g->tensor_hash); free(g->meta_hash); free(g->path); free(g);
}
void vg_gguf_set_direct_io(VG_GGUF *g, int enable) { if (g) g->direct_io = enable != 0; }
const char *vg_gguf_path(const VG_GGUF *g) { return g ? g->path : NULL; }
uint64_t vg_gguf_data_base(const VG_GGUF *g) { return g ? g->data_base : 0; }
uint64_t vg_gguf_file_size(const VG_GGUF *g) { return g ? g->file_size : 0; }
uint64_t vg_gguf_tensor_count(const VG_GGUF *g) { return g ? g->tensor_count : 0; }
const VG_GGUF_Tensor *vg_gguf_tensor_at(const VG_GGUF *g, uint64_t i) { return g && i < g->tensor_count ? &g->tensors[i] : NULL; }
const VG_GGUF_Tensor *vg_gguf_find_tensor(const VG_GGUF *g, const char *name) {
    if (!g || !name) return NULL;
    if (g->tensor_hash) {
        VG_HashEntry *e = hash_find(g->tensor_hash, g->tensor_hash_cap, name);
        if (e && e->occupied) return &g->tensors[e->index];
    }
    for (uint64_t i = 0; i < g->tensor_count; ++i) if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i]; return NULL;
}
const char *vg_gguf_meta(const VG_GGUF *g, const char *key) {
    if (!g || !key) return NULL;
    if (g->meta_hash) {
        VG_HashEntry *e = hash_find(g->meta_hash, g->meta_hash_cap, key);
        if (e && e->occupied) return g->meta[e->index].value;
    }
    for (size_t i = 0; i < g->meta_count; ++i) if (strcmp(g->meta[i].key, key) == 0) return g->meta[i].value; return NULL;
}
VG_Status vg_gguf_read(const VG_GGUF *g, const VG_GGUF_Tensor *t, uint64_t off, void *dst, size_t n) {
    if (!g || !t || !dst || off > t->nbytes || (uint64_t)n > t->nbytes - off) return VG_E_RANGE;
    uint64_t abs = g->data_base + t->data_offset + off;
#ifndef _WIN32
    if (g->direct_io) {
        int fd = open(g->path, O_RDONLY | O_DIRECT);
        if (fd < 0) return VG_E_IO;
        size_t align = 512;
        size_t aligned_off = (size_t)(abs / align) * align;
        size_t skip = (size_t)abs - aligned_off;
        size_t aligned_n = ((n + skip + align - 1) / align) * align;
        void *abuf = NULL;
        if (posix_memalign(&abuf, align, aligned_n) != 0) { close(fd); return VG_E_NOMEM; }
        ssize_t rd = pread(fd, abuf, aligned_n, (off_t)aligned_off);
        close(fd);
        if (rd < 0 || (size_t)rd < skip + n) { free(abuf); return VG_E_IO; }
        memcpy(dst, (unsigned char *)abuf + skip, n);
        free(abuf);
        return VG_OK;
    }
#endif
    FILE *f = fopen(g->path, "rb"); if (!f) return VG_E_IO;
    VG_Status s = seek64(f, abs) && n == fread(dst, 1, n, f) ? VG_OK : VG_E_IO; fclose(f); return s;
}
VG_Status vg_gguf_map(const VG_GGUF *gc, const void **base, uint64_t *size) {
    if (!gc || !base || !size) return VG_E_INVALID; VG_GGUF *g = (VG_GGUF *)gc; if (g->map_base) { *base = g->map_base; *size = g->map_size; return VG_OK; }
#ifdef _WIN32
    HANDLE f = CreateFileA(g->path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL); if (f == INVALID_HANDLE_VALUE) return VG_E_IO;
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL); if (!m) { CloseHandle(f); return VG_E_IO; }
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0); if (!p) { CloseHandle(m); CloseHandle(f); return VG_E_IO; }
    g->map_file = f; g->map_object = m;
#else
    int fd = open(g->path, O_RDONLY); if (fd < 0) return VG_E_IO; void *p = mmap(NULL, (size_t)g->file_size, PROT_READ, MAP_PRIVATE, fd, 0); if (p == MAP_FAILED) { close(fd); return VG_E_IO; } g->map_fd = fd;
#endif
    g->map_base = p; g->map_size = g->file_size; *base = p; *size = g->file_size; return VG_OK;
}
void vg_gguf_unmap(const VG_GGUF *gc) {
    if (!gc) return; VG_GGUF *g = (VG_GGUF *)gc; if (!g->map_base) return;
#ifdef _WIN32
    UnmapViewOfFile(g->map_base); CloseHandle(g->map_object); CloseHandle(g->map_file); g->map_object = NULL; g->map_file = INVALID_HANDLE_VALUE;
#else
    munmap(g->map_base, (size_t)g->map_size); close(g->map_fd); g->map_fd = -1;
#endif
    g->map_base = NULL; g->map_size = 0;
}
