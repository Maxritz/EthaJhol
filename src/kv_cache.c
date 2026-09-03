#include "vg/kv_cache.h"
#include "vg/trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct VG_Page {
    void *data;
    size_t bytes;
    uint32_t refs;
    uint64_t age;
    VG_Tier tier;
    int spilled;         /* 1 if data is on disk, 0 if resident */
    uint64_t spill_id;   /* unique ID for spill filename */
} VG_Page;

typedef struct VG_RadixNode {
    uint64_t *tokens;
    size_t count;
    size_t cap;
    uint64_t semantic_fingerprint;
    int tainted;
    VG_KVSequence *snapshot;
    struct VG_RadixNode **child;
    size_t nchild;
    size_t cchild;
} VG_RadixNode;

struct VG_KVStore {
    VG_KVConfig cfg;
    VG_Page *pages;
    size_t page_count;
    size_t page_capacity;
    uint64_t clock;
    uint64_t next_spill_id;
    VG_KVStats stats;
};
struct VG_KVSequence {
    VG_KVStore *store;
    uint64_t *tokens;
    size_t length;
    size_t capacity;
    VG_Page **pages;
    size_t page_count;
    size_t page_capacity;
};
struct VG_Radix { VG_KVStore *store; VG_RadixNode *root; };

static size_t page_bytes(const VG_KVStore *s) {
    uint64_t n = (uint64_t)s->cfg.n_layers * s->cfg.kv_heads * s->cfg.head_dim * s->cfg.element_bytes * s->cfg.page_tokens;
    return n > SIZE_MAX ? 0 : (size_t)n;
}
static int spill_page(VG_KVStore *s, VG_Page *p) {
    if (!s->cfg.spill_dir || p->spilled || !p->data) return 0;
    char path[512]; snprintf(path, sizeof(path), "%s/spill_%lu_%lu.bin", s->cfg.spill_dir, (unsigned long)s->next_spill_id, (unsigned long)p->age);
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    size_t written = fwrite(p->data, 1, p->bytes, f); fclose(f);
    if (written != p->bytes) { remove(path); return 0; }
    p->spill_id = s->next_spill_id++;
    p->spilled = 1;
    free(p->data); p->data = NULL;
    vg_trace_emit("kv_page_spill", 0, 0, p->bytes, 0);
    return 1;
}
static int fault_page(VG_KVStore *s, VG_Page *p) {
    if (!p->spilled) return 1;
    char path[512]; snprintf(path, sizeof(path), "%s/spill_%lu_%lu.bin", s->cfg.spill_dir, (unsigned long)p->spill_id, (unsigned long)p->age);
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    p->data = malloc(p->bytes ? p->bytes : 1);
    if (!p->data) { fclose(f); return 0; }
    size_t rd = fread(p->data, 1, p->bytes, f); fclose(f);
    if (rd != p->bytes) { free(p->data); p->data = NULL; return 0; }
    p->spilled = 0;
    remove(path);
    vg_trace_emit("kv_page_fault", 0, 0, p->bytes, 0);
    return 1;
}
static void evict_pages(VG_KVStore *s) {
    if (!s->cfg.host_budget_bytes) return;
    while (s->stats.host_bytes > s->cfg.host_budget_bytes) {
        size_t victim = (size_t)-1; uint64_t age = UINT64_MAX;
        for (size_t i = 0; i < s->page_count; ++i) if (s->pages[i].refs == 0 && s->pages[i].age < age) { victim = i; age = s->pages[i].age; }
        if (victim == (size_t)-1) break;
        s->stats.host_bytes -= s->pages[victim].bytes;
        if (s->pages[victim].tier == VG_TIER_DEVICE) s->stats.device_bytes -= s->pages[victim].bytes;
        vg_trace_emit("kv_page_evict", 0, 0, s->pages[victim].bytes, 0);
        if (s->cfg.spill_dir) { spill_page(s, &s->pages[victim]); }
        else { free(s->pages[victim].data); }
        if (victim + 1 < s->page_count) memmove(&s->pages[victim], &s->pages[victim + 1], (s->page_count - victim - 1) * sizeof(*s->pages));
        --s->page_count; ++s->stats.page_evictions;
    }
}
static VG_Page *new_page(VG_KVStore *s, VG_Tier tier) {
    size_t bytes = page_bytes(s); if (!bytes) return NULL;
    evict_pages(s);
    if (tier == VG_TIER_DEVICE && s->cfg.device_budget_bytes && s->stats.device_bytes + bytes > s->cfg.device_budget_bytes) return NULL;
    void *p = calloc(1, bytes); if (!p) return NULL;
    if (s->page_count == s->page_capacity) { size_t c = s->page_capacity ? s->page_capacity * 2u : 64u; VG_Page *q = (VG_Page *)realloc(s->pages, c * sizeof(*q)); if (!q) { free(p); return NULL; } s->pages = q; s->page_capacity = c; }
    VG_Page *x = &s->pages[s->page_count++]; memset(x, 0, sizeof(*x)); x->data = p; x->bytes = bytes; x->refs = 1; x->age = ++s->clock; x->tier = tier; ++s->stats.page_allocs; s->stats.host_bytes += bytes; if (tier == VG_TIER_DEVICE) s->stats.device_bytes += bytes;
    vg_trace_emit("kv_page_alloc", 0, 0, bytes, 0);
    return x;
}
static void page_retain(VG_Page *p) { if (p) ++p->refs; }
static void page_drop(VG_Page *p) { if (p && p->refs) --p->refs; }
VG_Status vg_kv_open(const VG_KVConfig *cfg, VG_KVStore **out) {
    if (!cfg || !out || !cfg->page_tokens || !cfg->n_layers || !cfg->kv_heads || !cfg->head_dim || !cfg->element_bytes) return VG_E_INVALID;
    VG_KVStore *s = (VG_KVStore *)calloc(1, sizeof(*s)); if (!s) return VG_E_NOMEM; s->cfg = *cfg; if (!s->cfg.max_pages) s->cfg.max_pages = 16384; if (!page_bytes(s)) { free(s); return VG_E_RANGE; } *out = s; return VG_OK;
}
void vg_kv_close(VG_KVStore *s) { if (!s) return; for (size_t i = 0; i < s->page_count; ++i) free(s->pages[i].data); free(s->pages); free(s); }
VG_Status vg_kv_sequence_init(VG_KVStore *s, VG_KVSequence **out) { if (!s || !out) return VG_E_INVALID; VG_KVSequence *q = (VG_KVSequence *)calloc(1, sizeof(*q)); if (!q) return VG_E_NOMEM; q->store = s; *out = q; return VG_OK; }
void vg_kv_sequence_close(VG_KVSequence *q) { if (!q) return; for (size_t i = 0; i < q->page_count; ++i) page_drop(q->pages[i]); free(q->pages); free(q->tokens); free(q); }
static VG_Status seq_reserve(VG_KVSequence *q, size_t n) {
    if (n <= q->capacity) return VG_OK; size_t c = q->capacity ? q->capacity * 2u : 256u; while (c < n) { if (c > SIZE_MAX / 2u) return VG_E_RANGE; c *= 2u; }
    uint64_t *p = (uint64_t *)realloc(q->tokens, c * sizeof(*p)); if (!p) return VG_E_NOMEM; q->tokens = p; q->capacity = c; return VG_OK;
}
static VG_Status pages_reserve(VG_KVSequence *q, size_t n) {
    if (n <= q->page_capacity) return VG_OK; size_t c = q->page_capacity ? q->page_capacity * 2u : 16u; while (c < n) { if (c > SIZE_MAX / 2u) return VG_E_RANGE; c *= 2u; }
    VG_Page **p = (VG_Page **)realloc(q->pages, c * sizeof(*p)); if (!p) return VG_E_NOMEM; q->pages = p; q->page_capacity = c; return VG_OK;
}
VG_Status vg_kv_sequence_append(VG_KVSequence *q, uint64_t token) {
    if (!q) return VG_E_INVALID; VG_Status st = seq_reserve(q, q->length + 1u); if (st != VG_OK) return st;
    size_t page_tokens = q->store->cfg.page_tokens; size_t pi = q->length / page_tokens;
    if (pi >= q->page_count) { if (q->page_count >= q->store->cfg.max_pages) return VG_E_RANGE; st = pages_reserve(q, pi + 1u); if (st != VG_OK) return st; VG_Page *p = new_page(q->store, VG_TIER_HOST); if (!p) return VG_E_NOMEM; q->pages[q->page_count++] = p; }
    q->tokens[q->length++] = token; return VG_OK;
}
VG_Status vg_kv_sequence_fork(VG_KVSequence *src, size_t prefix, VG_KVSequence **out) {
    if (!src || !out || prefix > src->length) return VG_E_INVALID; VG_KVSequence *q; VG_Status st = vg_kv_sequence_init(src->store, &q); if (st != VG_OK) return st;
    st = seq_reserve(q, prefix); if (st == VG_OK && prefix) memcpy(q->tokens, src->tokens, prefix * sizeof(*q->tokens)); q->length = prefix;
    size_t np = (prefix + src->store->cfg.page_tokens - 1u) / src->store->cfg.page_tokens;
    if (st == VG_OK) st = pages_reserve(q, np);
    if (st == VG_OK) for (size_t i = 0; i < np; ++i) { q->pages[q->page_count++] = src->pages[i]; page_retain(src->pages[i]); }
    if (st != VG_OK) { vg_kv_sequence_close(q); return st; } *out = q; return VG_OK;
}
size_t vg_kv_sequence_length(const VG_KVSequence *q) { return q ? q->length : 0; }
const uint64_t *vg_kv_sequence_tokens(const VG_KVSequence *q) { return q ? q->tokens : NULL; }
VG_Status vg_kv_page_acquire(VG_KVSequence *q, uint32_t layer, size_t token, VG_Tier desired, void **data, size_t *bytes) {
    if (!q || !data || !bytes || layer >= q->store->cfg.n_layers || token >= q->length) return VG_E_INVALID; size_t pi = token / q->store->cfg.page_tokens; VG_Page *p = q->pages[pi];
    if (p->spilled && !fault_page(q->store, p)) return VG_E_IO;
    size_t layer_bytes = (size_t)q->store->cfg.kv_heads * q->store->cfg.head_dim * q->store->cfg.element_bytes * q->store->cfg.page_tokens; p->age = ++q->store->clock; p->tier = desired; *data = (unsigned char *)p->data + (size_t)layer * layer_bytes; *bytes = layer_bytes; ++q->store->stats.page_hits;
    vg_trace_emit("kv_page_hit", 0, 0, layer_bytes, 0);
    return VG_OK;
}
void vg_kv_page_release(VG_KVSequence *q, uint32_t layer, size_t token) { (void)q; (void)layer; (void)token; }

static VG_RadixNode *node_new(const uint64_t *tokens, size_t count) {
    VG_RadixNode *n = (VG_RadixNode *)calloc(1, sizeof(*n)); if (!n) return NULL; if (count) { n->tokens = (uint64_t *)malloc(count * sizeof(*n->tokens)); if (!n->tokens) { free(n); return NULL; } memcpy(n->tokens, tokens, count * sizeof(*n->tokens)); n->count = n->cap = count; } return n;
}
static void node_free(VG_RadixNode *n) { if (!n) return; for (size_t i = 0; i < n->nchild; ++i) node_free(n->child[i]); if (n->snapshot) vg_kv_sequence_close(n->snapshot); free(n->child); free(n->tokens); free(n); }
static void node_invalidate_all(VG_RadixNode *n) { if (!n) return; if (n->snapshot) { vg_kv_sequence_close(n->snapshot); n->snapshot = NULL; } for (size_t i = 0; i < n->nchild; ++i) node_invalidate_all(n->child[i]); }
VG_Status vg_radix_open(VG_KVStore *s, VG_Radix **out) { if (!s || !out) return VG_E_INVALID; VG_Radix *r = (VG_Radix *)calloc(1, sizeof(*r)); if (!r) return VG_E_NOMEM; r->store = s; r->root = node_new(NULL, 0); if (!r->root) { free(r); return VG_E_NOMEM; } *out = r; return VG_OK; }
void vg_radix_close(VG_Radix *r) { if (!r) return; node_free(r->root); free(r); }
static size_t common(const uint64_t *a, size_t an, const uint64_t *b, size_t bn) { size_t n = an < bn ? an : bn, i = 0; while (i < n && a[i] == b[i]) ++i; return i; }
static VG_RadixNode *child_find(VG_RadixNode *n, uint64_t token) { for (size_t i = 0; i < n->nchild; ++i) if (n->child[i]->count && n->child[i]->tokens[0] == token) return n->child[i]; return NULL; }
static VG_Status child_add(VG_RadixNode *n, VG_RadixNode *child) { if (n->nchild == n->cchild) { size_t c = n->cchild ? n->cchild * 2u : 4u; VG_RadixNode **p = (VG_RadixNode **)realloc(n->child, c * sizeof(*p)); if (!p) return VG_E_NOMEM; n->child = p; n->cchild = c; } n->child[n->nchild++] = child; return VG_OK; }
VG_Status vg_radix_lookup(VG_Radix *r, const uint64_t *tokens, size_t count, size_t *matched, VG_KVSequence **snapshot) {
    if (!r || (!tokens && count) || !matched || !snapshot) return VG_E_INVALID; *matched = 0; *snapshot = NULL; ++r->store->stats.prefix_lookups; VG_RadixNode *n = r->root; size_t pos = 0; VG_RadixNode *best = NULL; size_t best_pos = 0;
    while (pos < count) { VG_RadixNode *c = child_find(n, tokens[pos]); if (!c) break; size_t k = common(tokens + pos, count - pos, c->tokens, c->count); if (k != c->count) break; pos += k; n = c; if (n->snapshot && !n->tainted) { best = n; best_pos = pos; } }
    if (best) { *matched = best_pos; *snapshot = best->snapshot; ++r->store->stats.prefix_hits; r->store->stats.prefix_tokens_reused += *matched;
        vg_trace_emit("radix_hit", 0, 0, *matched * sizeof(uint64_t), 1);
    }
    return VG_OK;
}
VG_Status vg_radix_insert(VG_Radix *r, VG_KVSequence *seq, size_t token_count, uint64_t fp, int tainted) {
    if (!r || !seq || token_count > seq->length) return VG_E_INVALID; VG_RadixNode *n = r->root; size_t pos = 0;
    while (pos < token_count) { VG_RadixNode *c = child_find(n, seq->tokens[pos]); if (!c) { c = node_new(seq->tokens + pos, token_count - pos); if (!c) return VG_E_NOMEM; if (child_add(n, c) != VG_OK) { node_free(c); return VG_E_NOMEM; } n = c; pos = token_count; break; } size_t k = common(seq->tokens + pos, token_count - pos, c->tokens, c->count); if (k < c->count) { /* split the edge */ VG_RadixNode *tail = node_new(c->tokens + k, c->count - k); if (!tail) return VG_E_NOMEM; tail->snapshot = c->snapshot; tail->semantic_fingerprint = c->semantic_fingerprint; tail->tainted = c->tainted; c->snapshot = NULL; c->semantic_fingerprint = 0; c->tainted = 0; for (size_t j = 0; j < c->nchild; ++j) (void)child_add(tail, c->child[j]); free(c->child); c->child = NULL; c->nchild = c->cchild = 0; c->tokens[k] = 0; c->count = k; if (child_add(c, tail) != VG_OK) { node_free(tail); return VG_E_NOMEM; } if (k == token_count - pos) { n = c; pos = token_count; break; } VG_RadixNode *leaf = node_new(seq->tokens + pos + k, token_count - pos - k); if (!leaf || child_add(c, leaf) != VG_OK) { node_free(leaf); return VG_E_NOMEM; } n = leaf; pos = token_count; break; } pos += k; n = c; }
    if (n->snapshot) vg_kv_sequence_close(n->snapshot); VG_Status st = vg_kv_sequence_fork(seq, token_count, &n->snapshot); if (st != VG_OK) return st; n->semantic_fingerprint = fp; n->tainted = tainted != 0; return VG_OK;
}
void vg_radix_invalidate(VG_Radix *r, uint64_t model_identity, uint64_t adapter_identity) { (void)model_identity; (void)adapter_identity; if (!r) return; node_invalidate_all(r->root); }
void vg_kv_stats(const VG_KVStore *sc, VG_KVStats *out) { if (sc && out) *out = sc->stats; }
