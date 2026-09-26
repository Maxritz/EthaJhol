#include "vg/tensor_source.h"
#include "vg/trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

typedef struct VG_CacheEntry {
    const VG_GGUF_Tensor *tensor;
    void *data;
    size_t bytes;
    size_t refs;
    uint64_t age;
    uint64_t ticket;
    VG_Tier tier;
    int owns;
} VG_CacheEntry;

typedef struct VG_IORequest {
    const char *name;
    VG_Tier tier;
    void *data;
    size_t bytes;
    int ready;
    int error;
    VG_PrefetchEntry *io_entry;
    struct VG_IORequest *next;
} VG_IORequest;

struct VG_TensorSource {
    VG_GGUF *file;
    VG_TensorSourceConfig cfg;
    VG_CacheEntry *entries;
    size_t count;
    size_t capacity;
    uint64_t clock;
    uint64_t next_ticket;
    VG_TensorSourceStats stats;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    HANDLE io_thread;
    int io_running;
#else
    pthread_mutex_t lock;
    pthread_t io_thread;
    int io_running;
#endif
    VG_IORequest *io_queue_head;
    VG_IORequest *io_queue_tail;
};

static void lock_src(VG_TensorSource *s) {
#ifdef _WIN32
    EnterCriticalSection(&s->lock);
#else
    (void)pthread_mutex_lock(&s->lock);
#endif
}
static void unlock_src(VG_TensorSource *s) {
#ifdef _WIN32
    LeaveCriticalSection(&s->lock);
#else
    (void)pthread_mutex_unlock(&s->lock);
#endif
}

#ifdef _WIN32
static DWORD WINAPI io_worker_win(LPVOID param) {
    VG_TensorSource *s = (VG_TensorSource *)param;
    while (1) {
#else
static void *io_worker(void *param) {
    VG_TensorSource *s = (VG_TensorSource *)param;
    while (1) {
#endif
        VG_IORequest *req = NULL;
        lock_src(s);
        if (s->io_queue_head) {
            req = s->io_queue_head;
            s->io_queue_head = req->next;
            if (!s->io_queue_head) s->io_queue_tail = NULL;
        }
        unlock_src(s);
        if (!req) {
            int running;
            lock_src(s); running = s->io_running; unlock_src(s);
            if (!running) {
#ifdef _WIN32
                return 0;
#else
                return NULL;
#endif
            }
            #ifdef _WIN32
            Sleep(1);
            #else
            struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL);
            #endif
            continue;
        }
        const VG_GGUF_Tensor *t = vg_gguf_find_tensor(s->file, req->name);
        if (!t) { req->error = 1; if (req->io_entry) { req->io_entry->error = 1; req->io_entry->ready = 1; } continue; }
        req->bytes = (size_t)t->nbytes;
        req->data = malloc(req->bytes ? req->bytes : 1);
        if (!req->data) { req->error = 1; if (req->io_entry) { req->io_entry->error = 1; req->io_entry->ready = 1; } continue; }
        VG_Status st = vg_gguf_read(s->file, t, 0, req->data, req->bytes);
        if (st != VG_OK) { free(req->data); req->data = NULL; req->error = 1; if (req->io_entry) { req->io_entry->error = 1; req->io_entry->ready = 1; } continue; }
        req->ready = 1;
        if (req->io_entry) {
            req->io_entry->data = req->data;
            req->io_entry->bytes = req->bytes;
            req->io_entry->ready = 1;
            req->io_entry->error = 0;
        }
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}
static size_t resident_bytes(const VG_TensorSource *s) {
    size_t total = 0; for (size_t i = 0; i < s->count; ++i) if (s->entries[i].owns) total += s->entries[i].bytes; return total;
}
static void remove_entry(VG_TensorSource *s, size_t i) {
    if (i >= s->count) return; if (s->entries[i].owns) free(s->entries[i].data);
    if (i + 1 < s->count) memmove(&s->entries[i], &s->entries[i + 1], (s->count - i - 1) * sizeof(*s->entries)); --s->count; ++s->stats.evictions;
}
static VG_CacheEntry *find_entry(VG_TensorSource *s, const VG_GGUF_Tensor *t, VG_Tier tier) {
    for (size_t i = 0; i < s->count; ++i) if (s->entries[i].tensor == t && s->entries[i].tier == tier) return &s->entries[i]; return NULL;
}
static void evict_for(VG_TensorSource *s, size_t need) {
    size_t budget = s->cfg.host_budget_bytes;
    if (budget == 0) return;
    while (resident_bytes(s) > budget || (need && resident_bytes(s) > budget - need)) {
        size_t victim = (size_t)-1; uint64_t age = UINT64_MAX;
        for (size_t i = 0; i < s->count; ++i) if (s->entries[i].owns && s->entries[i].refs == 0 && s->entries[i].age < age) { victim = i; age = s->entries[i].age; }
        if (victim == (size_t)-1) break; remove_entry(s, victim);
    }
}
VG_Status vg_tensor_source_open(VG_GGUF *file, const VG_TensorSourceConfig *cfg, VG_TensorSource **out) {
    if (!file || !out) return VG_E_INVALID; *out = NULL;
    VG_TensorSource *s = (VG_TensorSource *)calloc(1, sizeof(*s)); if (!s) return VG_E_NOMEM; s->file = file;
    s->cfg.host_budget_bytes = 512u * 1024u * 1024u; s->cfg.io_chunk_bytes = 4u * 1024u * 1024u; s->cfg.prefetch_depth = 2; s->cfg.use_mmap = 1;
    if (cfg) s->cfg = *cfg; if (s->cfg.io_chunk_bytes == 0) s->cfg.io_chunk_bytes = 4u * 1024u * 1024u; s->stats.bytes_budget = s->cfg.host_budget_bytes; s->stats.device_budget = s->cfg.device_budget_bytes;
    if (s->cfg.use_direct_io) vg_gguf_set_direct_io(file, 1);
#ifdef _WIN32
    InitializeCriticalSection(&s->lock);
    if (s->cfg.io_workers > 0) {
        s->io_running = 1;
        s->io_thread = CreateThread(NULL, 0, io_worker_win, s, 0, NULL);
        if (!s->io_thread) { DeleteCriticalSection(&s->lock); free(s); return VG_E_NOMEM; }
    }
#else
    if (pthread_mutex_init(&s->lock, NULL) != 0) { free(s); return VG_E_BUSY; }
    if (s->cfg.io_workers > 0) {
        s->io_running = 1;
        if (pthread_create(&s->io_thread, NULL, io_worker, s) != 0) { pthread_mutex_destroy(&s->lock); free(s); return VG_E_NOMEM; }
    }
#endif
    *out = s; return VG_OK;
}

void vg_tensor_source_close(VG_TensorSource *s) {
    if (!s) return;
    if (s->io_thread) {
        s->io_running = 0;
        #ifdef _WIN32
        WaitForSingleObject(s->io_thread, 5000); CloseHandle(s->io_thread);
        #else
        pthread_join(s->io_thread, NULL);
        #endif
    }
    lock_src(s); for (size_t i = 0; i < s->count; ++i) if (s->entries[i].owns) free(s->entries[i].data); free(s->entries); unlock_src(s);
    while (s->io_queue_head) { VG_IORequest *n = s->io_queue_head->next; free((void*)s->io_queue_head); s->io_queue_head = n; }
#ifdef _WIN32
    DeleteCriticalSection(&s->lock);
#else
    (void)pthread_mutex_destroy(&s->lock);
#endif
    free(s);
}
VG_Status vg_tensor_acquire(VG_TensorSource *s, const char *name, VG_Tier desired, VG_TensorLease *out) {
    if (!s || !name || !out) return VG_E_INVALID; memset(out, 0, sizeof(*out));
    uint64_t t0 = vg_trace_now_ns();
    const VG_GGUF_Tensor *t = vg_gguf_find_tensor(s->file, name); if (!t) return VG_E_INVALID;
    lock_src(s); ++s->stats.lookups; VG_CacheEntry *e = find_entry(s, t, desired);
    if (e) { ++e->refs; e->age = ++s->clock; ++s->stats.hits; out->tensor = t; out->data = e->data; out->size = e->bytes; out->tier = e->tier; out->ticket = (uint64_t)(e - s->entries + 1); out->source = s; unlock_src(s); vg_trace_emit("tensor_acquire_hit", t0, vg_trace_now_ns(), e->bytes, 0); return VG_OK; }
    ++s->stats.misses;
    /* A storage lease returns a pointer into the mapped file. Virtual mapping is lazy and does not
     * make the whole model resident. It is the preferred path for fused streaming kernels. */
    if (desired == VG_TIER_STORAGE && s->cfg.use_mmap) {
        const void *base; uint64_t size; VG_Status st = vg_gguf_map(s->file, &base, &size);
        if (st != VG_OK || t->data_offset > size - vg_gguf_data_base(s->file)) { unlock_src(s); return VG_E_IO; }
        out->tensor = t; out->data = (const unsigned char *)base + vg_gguf_data_base(s->file) + t->data_offset; out->size = (size_t)t->nbytes; out->tier = VG_TIER_STORAGE; out->source = s; out->ticket = 0; unlock_src(s); return VG_OK;
    }
    size_t n = (size_t)t->nbytes; void *data = malloc(n ? n : 1u); if (!data) { unlock_src(s); return VG_E_NOMEM; }
    VG_Status st = vg_gguf_read(s->file, t, 0, data, n); if (st != VG_OK) { free(data); unlock_src(s); return st; }
    size_t budget = desired == VG_TIER_PINNED ? s->cfg.pinned_budget_bytes : s->cfg.host_budget_bytes;
    if (desired == VG_TIER_DEVICE && s->cfg.device_budget_bytes && s->stats.bytes_device + n > s->cfg.device_budget_bytes) {
        free(data); unlock_src(s); return VG_E_NOMEM;
    }
    if (budget && n <= budget) {
        evict_for(s, n); if (s->count == s->capacity) { size_t c = s->capacity ? s->capacity * 2u : 32u; VG_CacheEntry *p = (VG_CacheEntry *)realloc(s->entries, c * sizeof(*p)); if (!p) { free(data); unlock_src(s); return VG_E_NOMEM; } s->entries = p; s->capacity = c; }
        e = &s->entries[s->count++]; memset(e, 0, sizeof(*e)); e->tensor = t; e->data = data; e->bytes = n; e->refs = 1; e->age = ++s->clock; e->ticket = ++s->next_ticket; e->tier = desired; e->owns = 1; s->stats.bytes_resident = resident_bytes(s); s->stats.bytes_budget = budget;
        if (desired == VG_TIER_DEVICE) s->stats.bytes_device += n;
        out->ticket = e->ticket;
    } else {
        /* Oversized tensors are still valid: return a transient lease instead of failing because the cache is smaller. */
        out->ticket = 0; out->source = s; out->tensor = t; out->data = data; out->size = n; out->tier = desired; s->stats.bytes_resident = resident_bytes(s); s->stats.bytes_budget = budget;
        if (desired == VG_TIER_DEVICE) s->stats.bytes_device += n;
        s->stats.bytes_read += n; unlock_src(s); vg_trace_emit("tensor_acquire_miss", t0, vg_trace_now_ns(), n, 0); return VG_OK;
    }
    out->source = s; out->tensor = t; out->data = e->data; out->size = e->bytes; out->tier = e->tier; s->stats.bytes_read += n; unlock_src(s); vg_trace_emit("tensor_acquire_cached", t0, vg_trace_now_ns(), n, 0);
    return VG_OK;
}
void vg_tensor_release(VG_TensorLease *lease) {
    if (!lease || !lease->source) return; VG_TensorSource *s = lease->source; lock_src(s);
    if (lease->ticket) { for (size_t i = 0; i < s->count; ++i) if (s->entries[i].ticket == lease->ticket && s->entries[i].tensor == lease->tensor) { if (s->entries[i].refs) --s->entries[i].refs; break; } }
    else if (lease->data && lease->tier != VG_TIER_STORAGE) free((void *)lease->data);
    unlock_src(s); memset(lease, 0, sizeof(*lease));
}
VG_Status vg_tensor_prefetch(VG_TensorSource *s, const char *const *names, size_t count, VG_Tier desired) {
    if (!s || (!names && count)) return VG_E_INVALID;
    for (size_t i = 0; i < count; ++i) { VG_TensorLease l; VG_Status st = vg_tensor_acquire(s, names[i], desired, &l); if (st != VG_OK) return st; vg_tensor_release(&l); ++s->stats.prefetches; }
    return VG_OK;
}
VG_Status vg_tensor_prefetch_async(VG_TensorSource *s, const char *name, VG_Tier desired, VG_PrefetchEntry *entry) {
    if (!s || !name || !entry) return VG_E_INVALID;
    memset(entry, 0, sizeof(*entry));
    entry->name = name;
    entry->tier = desired;
    if (!s->io_thread) return vg_tensor_prefetch(s, &name, 1, desired);
    lock_src(s);
    VG_IORequest *req = (VG_IORequest *)calloc(1, sizeof(VG_IORequest));
    if (!req) { unlock_src(s); return VG_E_NOMEM; }
    req->name = name;
    req->tier = desired;
    req->io_entry = entry;
    if (s->io_queue_tail) { s->io_queue_tail->next = req; s->io_queue_tail = req; }
    else { s->io_queue_head = s->io_queue_tail = req; }
    unlock_src(s);
    return VG_OK;
}
VG_Status vg_tensor_wait(VG_TensorSource *s, VG_PrefetchEntry *entry) {
    if (!s || !entry) return VG_E_INVALID;
    while (1) {
        lock_src(s);
        int ready = 0;
        if (entry->ready) ready = 1;
        unlock_src(s);
        if (ready) return entry->error ? VG_E_IO : VG_OK;
        #ifdef _WIN32
        Sleep(1);
        #else
        struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL);
        #endif
    }
    return VG_OK;
}
void vg_tensor_source_stats(const VG_TensorSource *sc, VG_TensorSourceStats *out) { if (!sc || !out) return; VG_TensorSource *s = (VG_TensorSource *)sc; lock_src(s); *out = s->stats; unlock_src(s); }
VG_Status vg_tensor_stream(VG_TensorSource *s, const char *name, size_t chunk, VG_Status (*consume)(const void *, size_t, uint64_t, void *), void *user) {
    if (!s || !name || !consume) return VG_E_INVALID; const VG_GGUF_Tensor *t = vg_gguf_find_tensor(s->file, name); if (!t) return VG_E_INVALID; if (!chunk) chunk = s->cfg.io_chunk_bytes;
    void *buf = malloc(chunk); if (!buf) return VG_E_NOMEM;     VG_Status st = VG_OK;
    for (uint64_t off = 0; off < t->nbytes && st == VG_OK;) { size_t n = (size_t)((t->nbytes - off) < chunk ? (t->nbytes - off) : chunk); st = vg_gguf_read(s->file, t, off, buf, n); if (st == VG_OK) st = consume(buf, n, off, user); off += n; }
    free(buf); return st;
}

VG_Status vg_tensor_prefetch_after(VG_TensorSource *s, const char *name, unsigned depth) {
    if (!s || !name || depth == 0) return VG_E_INVALID;
    const VG_GGUF_Tensor *t = vg_gguf_find_tensor(s->file, name); if (!t) return VG_E_INVALID;
    uint64_t tc = vg_gguf_tensor_count(s->file);
    for (uint64_t fi = 0; fi < tc; ++fi) {
        if (vg_gguf_tensor_at(s->file, fi) == t) {
            for (unsigned p = 1; p <= depth && fi + p < tc; ++p) {
                const VG_GGUF_Tensor *next = vg_gguf_tensor_at(s->file, fi + p);
                if (next) { VG_TensorLease pl; if (vg_tensor_acquire(s, next->name, VG_TIER_HOST, &pl) == VG_OK) { vg_tensor_release(&pl); ++s->stats.prefetch_hits; } }
            }
            return VG_OK;
        }
    }
    return VG_E_INVALID;
}
