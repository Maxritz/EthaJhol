#include "vg/tokenizer.h"
#include <stdlib.h>
#include <string.h>

/* Two tokenizer families appear in GGUF files and they are not interchangeable:
 *
 *   SentencePiece ("llama", "spm", "gemma", "bert")   words carry a U+2581 space marker
 *                                                     and pieces come from BPE merges
 *   GPT-2 byte-level BPE ("gpt2", "bpe", "qwen2")     words carry a U+0120 space marker
 *                                                     and bytes use a unicode remapping
 *
 * The previous implementation always used the GPT-2 marker and never applied merges, so on a
 * SentencePiece vocabulary ("▁capital" is the only spelling of that word) every word missed
 * both the prefixed and the bare lookup and fell through to per-character byte tokens. Feeding
 * a prompt shredded into single characters is what made generation incoherent.
 */

typedef struct { char *text; int32_t id; float score; } VG_VocabEntry;
typedef struct { char *a; char *b; int rank; } VG_Merge;

/* Open-addressed index tables. The old lookup scanned the whole vocabulary with strcmp for
 * every candidate string, which is O(vocab) per BPE merge step. */
typedef struct { uint32_t hash; int32_t index; } VG_Slot;

typedef enum VG_TokKind {
    VG_TOK_SPM = 0,   /* SentencePiece: U+2581 marker, merges over UTF-8 pieces */
    VG_TOK_BPE = 1    /* GPT-2 byte-level BPE: U+0120 marker, byte remap on decode */
} VG_TokKind;

struct VG_Tokenizer {
    VG_VocabEntry *vocab;
    size_t vocab_count;
    VG_Merge *merges;
    size_t merge_count;
    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;
    VG_TokKind kind;
    VG_Slot *vhash;
    size_t vhash_cap;
    VG_Slot *mhash;
    size_t mhash_cap;
    int32_t byte_tokens[256];   /* id of "<0xNN>", or -1 */
    int has_byte_fallback;
    int add_space_prefix;
    /* tokenizer.ggml.token_type (1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED 6=BYTE).
     * When present it decides which tokens are silent, instead of matching names. */
    unsigned char *types;
};

/* --- hashing -------------------------------------------------------------------- */
static uint32_t fnv_bytes(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}
static uint32_t fnv_pair(const char *a, size_t an, const char *b, size_t bn) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < an; ++i) { h ^= (unsigned char)a[i]; h *= 16777619u; }
    h ^= 0x1Fu; h *= 16777619u;
    for (size_t i = 0; i < bn; ++i) { h ^= (unsigned char)b[i]; h *= 16777619u; }
    return h;
}

static int build_vocab_index(VG_Tokenizer *tok) {
    if (tok->vocab_count == 0) return 0;
    size_t cap = 64;
    while (cap < tok->vocab_count * 2u) cap *= 2u;
    tok->vhash = (VG_Slot *)malloc(cap * sizeof(VG_Slot));
    if (!tok->vhash) return 0;
    tok->vhash_cap = cap;
    for (size_t i = 0; i < cap; ++i) tok->vhash[i].index = -1;
    const size_t mask = cap - 1u;
    for (size_t i = 0; i < tok->vocab_count; ++i) {
        const char *t = tok->vocab[i].text;
        const uint32_t h = fnv_bytes(t, strlen(t));
        size_t s = h & mask;
        for (size_t p = 0; p < cap; ++p) {
            if (tok->vhash[s].index < 0) {
                tok->vhash[s].hash = h;
                tok->vhash[s].index = (int32_t)i;
                break;
            }
            s = (s + 1u) & mask;
        }
    }
    return 1;
}

static int build_merge_index(VG_Tokenizer *tok) {
    if (tok->merge_count == 0) return 0;
    size_t cap = 64;
    while (cap < tok->merge_count * 2u) cap *= 2u;
    tok->mhash = (VG_Slot *)malloc(cap * sizeof(VG_Slot));
    if (!tok->mhash) return 0;
    tok->mhash_cap = cap;
    for (size_t i = 0; i < cap; ++i) tok->mhash[i].index = -1;
    const size_t mask = cap - 1u;
    for (size_t i = 0; i < tok->merge_count; ++i) {
        const char *a = tok->merges[i].a, *b = tok->merges[i].b;
        const uint32_t h = fnv_pair(a, strlen(a), b, strlen(b));
        size_t s = h & mask;
        for (size_t p = 0; p < cap; ++p) {
            if (tok->mhash[s].index < 0) {
                tok->mhash[s].hash = h;
                tok->mhash[s].index = (int32_t)i;
                break;
            }
            s = (s + 1u) & mask;
        }
    }
    return 1;
}

static int32_t vocab_lookup(const VG_Tokenizer *tok, const char *s, size_t n) {
    if (!tok->vhash || !tok->vhash_cap) return -1;
    const size_t mask = tok->vhash_cap - 1u;
    const uint32_t h = fnv_bytes(s, n);
    size_t i = h & mask;
    for (size_t p = 0; p < tok->vhash_cap; ++p) {
        const VG_Slot *sl = &tok->vhash[i];
        if (sl->index < 0) return -1;
        if (sl->hash == h) {
            const char *t = tok->vocab[sl->index].text;
            if (strlen(t) == n && memcmp(t, s, n) == 0) return sl->index;
        }
        i = (i + 1u) & mask;
    }
    return -1;
}

/* Lowest rank wins; -1 means the pair is not a merge. */
static int merge_rank(const VG_Tokenizer *tok, const char *a, size_t an, const char *b, size_t bn) {
    if (!tok->mhash || !tok->mhash_cap) return -1;
    const size_t mask = tok->mhash_cap - 1u;
    const uint32_t h = fnv_pair(a, an, b, bn);
    size_t i = h & mask;
    for (size_t p = 0; p < tok->mhash_cap; ++p) {
        const VG_Slot *sl = &tok->mhash[i];
        if (sl->index < 0) return -1;
        if (sl->hash == h) {
            const VG_Merge *m = &tok->merges[sl->index];
            if (strlen(m->a) == an && strlen(m->b) == bn &&
                memcmp(m->a, a, an) == 0 && memcmp(m->b, b, bn) == 0) return m->rank;
        }
        i = (i + 1u) & mask;
    }
    return -1;
}

/* --- small dynamic arrays ------------------------------------------------------- */
typedef struct { size_t off; size_t len; } VG_Sym;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} VG_Buf;

static int buf_reserve(VG_Buf *b, size_t extra) {
    if (b->len + extra <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 128;
    while (cap < b->len + extra) cap *= 2u;
    char *p = (char *)realloc(b->data, cap);
    if (!p) return 0;
    b->data = p;
    b->cap = cap;
    return 1;
}
static int buf_append(VG_Buf *b, const char *s, size_t n) {
    if (!buf_reserve(b, n)) return 0;
    memcpy(b->data + b->len, s, n);
    b->len += n;
    return 1;
}

static int sym_reserve(VG_Sym **syms, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    size_t c = *cap ? *cap : 64;
    while (c < need) c *= 2u;
    VG_Sym *p = (VG_Sym *)realloc(*syms, c * sizeof(VG_Sym));
    if (!p) return 0;
    *syms = p;
    *cap = c;
    return 1;
}

/* --- loading -------------------------------------------------------------------- */
static char *dup_range(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1u);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

/* Splits the array-text form returned by vg_gguf_meta ("[\"a\",\"b\"]") into owned strings.
 * The GGUF reader escapes '"' and '\', so those are unescaped here. */
static size_t split_quoted(const char *src, char ***out) {
    *out = NULL;
    if (!src) return 0;
    const char *p = src;
    if (*p == '[') ++p;
    size_t cap = 64, count = 0;
    char **items = (char **)malloc(cap * sizeof(char *));
    if (!items) return 0;
    while (*p && *p != ']') {
        while (*p == ',' || *p == ' ') ++p;
        if (*p != '"') break;
        ++p;
        char *tmp = NULL;
        size_t len = 0, tcap = 0;
        while (*p && *p != '"') {
            char c = *p++;
            if (c == '\\' && *p) c = *p++;
            if (len + 1u > tcap) {
                size_t nc = tcap ? tcap * 2u : 32u;
                char *nt = (char *)realloc(tmp, nc);
                if (!nt) { free(tmp); goto fail; }
                tmp = nt;
                tcap = nc;
            }
            tmp[len++] = c;
        }
        if (*p == '"') ++p;
        if (!tmp) { tmp = dup_range("", 0); if (!tmp) goto fail; }
        tmp[len] = 0;
        if (count == cap) {
            size_t nc = cap * 2u;
            char **ni = (char **)realloc(items, nc * sizeof(char *));
            if (!ni) { free(tmp); goto fail; }
            items = ni;
            cap = nc;
        }
        items[count++] = tmp;
    }
    *out = items;
    return count;
fail:
    for (size_t i = 0; i < count; ++i) free(items[i]);
    free(items);
    return 0;
}

static int parse_byte_token(const char *t, unsigned *byte_out) {
    if (t[0] != '<' || t[1] != '0' || t[2] != 'x') return 0;
    const char *h = t + 3;
    unsigned v = 0;
    int digits = 0;
    while (*h && *h != '>') {
        char c = *h++;
        unsigned d;
        if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A' + 10);
        else return 0;
        v = v * 16u + d;
        if (++digits > 2) return 0;
    }
    if (!digits || *h != '>') return 0;
    *byte_out = v;
    return 1;
}

VG_Status vg_tokenizer_load(const VG_GGUF *file, VG_Tokenizer **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_Tokenizer *tok = (VG_Tokenizer *)calloc(1, sizeof(*tok));
    if (!tok) return VG_E_NOMEM;
    for (int i = 0; i < 256; ++i) tok->byte_tokens[i] = -1;
    tok->bos_id = -1;
    tok->eos_id = -1;
    tok->unk_id = 0;

    const char *bos_s = vg_gguf_meta(file, "tokenizer.ggml.bos_token_id");
    const char *eos_s = vg_gguf_meta(file, "tokenizer.ggml.eos_token_id");
    if (bos_s) tok->bos_id = (int32_t)strtol(bos_s, NULL, 10);
    if (eos_s) tok->eos_id = (int32_t)strtol(eos_s, NULL, 10);

    const char *tok_model = vg_gguf_meta(file, "tokenizer.ggml.model");
    const char *add_prefix = vg_gguf_meta(file, "tokenizer.ggml.add_space_prefix");
    if (add_prefix) tok->add_space_prefix = (strtol(add_prefix, NULL, 10) != 0);

    char **tokens = NULL;
    const size_t n_tokens = split_quoted(vg_gguf_meta(file, "tokenizer.ggml.tokens"), &tokens);
    if (n_tokens == 0) { free(tok); return VG_E_FORMAT; }
    tok->vocab = (VG_VocabEntry *)calloc(n_tokens, sizeof(VG_VocabEntry));
    if (!tok->vocab) { for (size_t i = 0; i < n_tokens; ++i) free(tokens[i]); free(tokens); free(tok); return VG_E_NOMEM; }
    tok->vocab_count = n_tokens;
    for (size_t i = 0; i < n_tokens; ++i) {
        tok->vocab[i].text = tokens[i];
        tok->vocab[i].id = (int32_t)i;
        tok->vocab[i].score = 0.0f;
    }
    free(tokens);

    /* Scores give the unigram fallback ordering; optional. */
    {
        const char *scores_str = vg_gguf_meta(file, "tokenizer.ggml.scores");
        if (scores_str) {
            const char *p = scores_str;
            if (*p == '[') ++p;
            for (size_t i = 0; i < tok->vocab_count && *p && *p != ']'; ++i) {
                while (*p == ',' || *p == ' ') ++p;
                if (*p == ']') break;
                tok->vocab[i].score = (float)strtod(p, (char **)&p);
            }
        }
    }

    char **merges = NULL;
    const size_t n_merges = split_quoted(vg_gguf_meta(file, "tokenizer.ggml.merges"), &merges);
    if (n_merges) {
        tok->merges = (VG_Merge *)calloc(n_merges, sizeof(VG_Merge));
        if (!tok->merges) { for (size_t i = 0; i < n_merges; ++i) free(merges[i]); free(merges); vg_tokenizer_free(tok); return VG_E_NOMEM; }
        for (size_t i = 0; i < n_merges; ++i) {
            const char *m = merges[i];
            const char *sep = strchr(m, ' ');
            if (!sep) { tok->merges[i].a = dup_range(m, strlen(m)); tok->merges[i].b = dup_range("", 0); }
            else {
                tok->merges[i].a = dup_range(m, (size_t)(sep - m));
                tok->merges[i].b = dup_range(sep + 1, strlen(sep + 1));
            }
            free(merges[i]);
            if (!tok->merges[i].a || !tok->merges[i].b) { free(merges); vg_tokenizer_free(tok); return VG_E_NOMEM; }
            tok->merges[i].rank = (int)i;   /* file order is rank order */
        }
        free(merges);
        tok->merge_count = n_merges;
    }

    if (!build_vocab_index(tok)) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
    (void)build_merge_index(tok);

    /* Optional per-token type table, so control/byte tokens are identified by metadata. */
    {
        const char *types_str = vg_gguf_meta(file, "tokenizer.ggml.token_type");
        if (types_str) {
            unsigned char *types = (unsigned char *)calloc(tok->vocab_count, 1);
            if (types) {
                const char *p = types_str;
                if (*p == '[') ++p;
                size_t n = 0;
                while (*p && *p != ']' && n < tok->vocab_count) {
                    while (*p == ',' || *p == ' ') ++p;
                    if (*p == ']') break;
                    types[n++] = (unsigned char)strtoul(p, (char **)&p, 10);
                }
                if (n > 0) tok->types = types;
                else free(types);
            }
        }
    }

    /* Byte-fallback table: SentencePiece stores raw bytes as "<0xNN>" vocabulary entries. */
    for (size_t i = 0; i < tok->vocab_count; ++i) {
        unsigned b;
        const int is_byte_type = tok->types && tok->types[i] == 6u;
        if ((is_byte_type || !tok->types) && parse_byte_token(tok->vocab[i].text, &b) && b < 256u) {
            tok->byte_tokens[b] = (int32_t)i;
            tok->has_byte_fallback = 1;
        }
        if (tok->types && tok->types[i] == 2u) tok->unk_id = (int32_t)i;
        else if (!tok->types && strcmp(tok->vocab[i].text, "<unk>") == 0) tok->unk_id = (int32_t)i;
    }

    /* Family detection: trust the declared model, then fall back to vocabulary shape. */
    if (tok_model && (strcmp(tok_model, "gpt2") == 0 || strcmp(tok_model, "bpe") == 0)) {
        tok->kind = VG_TOK_BPE;
    } else if (tok_model && (strcmp(tok_model, "llama") == 0 || strcmp(tok_model, "spm") == 0 ||
                             strcmp(tok_model, "sentencepiece") == 0 || strcmp(tok_model, "gemma") == 0 ||
                             strcmp(tok_model, "gemma4") == 0 || strcmp(tok_model, "bert") == 0)) {
        tok->kind = VG_TOK_SPM;
    } else {
        int spm_marker = 0, gpt2_marker = 0;
        for (size_t i = 0; i < tok->vocab_count; ++i) {
            const unsigned char *t = (const unsigned char *)tok->vocab[i].text;
            if (t[0] == 0xE2 && t[1] == 0x96 && t[2] == 0x81) { ++spm_marker; if (spm_marker > 4) break; }
            else if (t[0] == 0xC4 && t[1] == 0xA0) { ++gpt2_marker; if (gpt2_marker > 4) break; }
        }
        tok->kind = (spm_marker >= gpt2_marker) ? VG_TOK_SPM : VG_TOK_BPE;
    }

    *out = tok;
    return VG_OK;
}

void vg_tokenizer_free(VG_Tokenizer *tok) {
    if (!tok) return;
    for (size_t i = 0; i < tok->vocab_count; ++i) free(tok->vocab[i].text);
    free(tok->vocab);
    for (size_t i = 0; i < tok->merge_count; ++i) { free(tok->merges[i].a); free(tok->merges[i].b); }
    free(tok->merges);
    free(tok->vhash);
    free(tok->mhash);
    free(tok->types);
    free(tok);
}

size_t vg_tokenizer_vocab_size(const VG_Tokenizer *tok) { return tok ? tok->vocab_count : 0; }
int32_t vg_tokenizer_bos_id(const VG_Tokenizer *tok) { return tok ? tok->bos_id : -1; }
int32_t vg_tokenizer_eos_id(const VG_Tokenizer *tok) { return tok ? tok->eos_id : -1; }

/* --- encoding ------------------------------------------------------------------- */

/* Length in bytes of the UTF-8 sequence starting at s, or 1 for an invalid lead byte. */
static size_t utf8_len(const unsigned char *s) {
    if (s[0] < 0x80u) return 1;
    if ((s[0] & 0xE0u) == 0xC0u) return 2;
    if ((s[0] & 0xF0u) == 0xE0u) return 3;
    if ((s[0] & 0xF8u) == 0xF0u) return 4;
    return 1;
}

typedef struct {
    int32_t *ids;
    size_t max_ids;
    size_t count;
} VG_Emit;

static void emit_id(VG_Emit *e, int32_t id) {
    if (e->ids && e->count < e->max_ids) e->ids[e->count] = id;
    ++e->count;
}

/* Recognises a literal reserved token (<s>, </s>, ...) starting at text[i]. Detection has to
 * work at any offset, not only at a fragment start: a sequence like "Einstein?</s>" keeps
 * </s> inside the word fragment, where a fragment-start-only check silently misses it and the
 * end-of-sequence token ends up encoded as '<', '/', 's', '>'. */
static int special_at(const VG_Tokenizer *tok, const char *text, size_t n, size_t i,
                      size_t *len_out, int32_t *id_out) {
    if (text[i] != '<') return 0;
    size_t j = i + 1u;
    while (j < n && text[j] != '>' && j - i <= 48u) ++j;
    if (j >= n || text[j] != '>') return 0;
    const int32_t id = vocab_lookup(tok, text + i, j - i + 1u);
    if (id < 0) return 0;
    *len_out = j - i + 1u;
    *id_out = id;
    return 1;
}

/* Applies BPE merges to one normalised fragment and emits the resulting ids. */
static void encode_fragment(const VG_Tokenizer *tok, const char *piece, size_t plen, VG_Emit *emit) {
    if (plen == 0) return;
    VG_Sym *syms = NULL;
    size_t cap = 0, nsym = 0;

    for (size_t i = 0; i < plen;) {
        const size_t l = utf8_len((const unsigned char *)piece + i);
        const size_t l2 = (i + l <= plen) ? l : 1u;
        if (!sym_reserve(&syms, &cap, nsym + 1u)) goto done;
        syms[nsym].off = i;
        syms[nsym].len = l2;
        ++nsym;
        i += l2;
    }

    if (tok->kind == VG_TOK_SPM && tok->merge_count) {
        for (;;) {
            int best = -1;
            size_t best_at = 0;
            for (size_t i = 0; i + 1u < nsym; ++i) {
                const int r = merge_rank(tok, piece + syms[i].off, syms[i].len,
                                         piece + syms[i + 1u].off, syms[i + 1u].len);
                if (r >= 0 && (best < 0 || r < best)) { best = r; best_at = i; }
            }
            if (best < 0) break;
            /* Merge every occurrence of the winning pair, left to right, in one pass. */
            size_t k = 0;
            for (size_t i = 0; i < nsym;) {
                if (i + 1u < nsym &&
                    merge_rank(tok, piece + syms[i].off, syms[i].len,
                               piece + syms[i + 1u].off, syms[i + 1u].len) == best) {
                    syms[k].off = syms[i].off;
                    syms[k].len = syms[i].len + syms[i + 1u].len;
                    ++k;
                    i += 2u;
                } else {
                    syms[k++] = syms[i++];
                }
            }
            nsym = k;
            if (nsym < 2u) break;
            (void)best_at;
        }
    }

    for (size_t i = 0; i < nsym; ++i) {
        const char *s = piece + syms[i].off;
        const size_t n = syms[i].len;
        const int32_t id = vocab_lookup(tok, s, n);
        if (id >= 0) { emit_id(emit, id); continue; }
        if (tok->has_byte_fallback) {
            for (size_t b = 0; b < n; ++b) {
                const unsigned char byte = (unsigned char)s[b];
                emit_id(emit, tok->byte_tokens[byte] >= 0 ? tok->byte_tokens[byte] : tok->unk_id);
            }
        } else {
            emit_id(emit, tok->unk_id);
        }
    }
done:
    free(syms);
}

size_t vg_tokenizer_encode(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids) {
    if (!tok || !text) return 0;
    const size_t n = strlen(text);
    if (n == 0) return 0;

    VG_Emit emit;
    emit.ids = ids;
    emit.max_ids = max_ids;
    emit.count = 0;

    VG_Buf piece = {0};
    size_t i = 0;
    int first_fragment = 1;
    int pending_space = 0;

    while (i < n) {
        size_t special_len = 0;
        int32_t special_id = -1;
        if (text[i] == '<' && special_at(tok, text, n, i, &special_len, &special_id)) {
            emit_id(&emit, special_id);
            i += special_len;
            pending_space = 0;
            first_fragment = 0;
            continue;
        }

        if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
            while (i < n && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
            pending_space = 1;
            continue;
        }

        const size_t start = i;
        while (i < n && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r') {
            /* Stop the fragment so a reserved token inside a word still reaches the scanner. */
            if (i > start && text[i] == '<') {
                size_t probe_len = 0;
                int32_t probe_id = -1;
                if (special_at(tok, text, n, i, &probe_len, &probe_id)) break;
            }
            ++i;
        }
        const size_t flen = i - start;

        piece.len = 0;
        const int with_marker = pending_space || (first_fragment && tok->add_space_prefix);
        if (with_marker) {
            if (tok->kind == VG_TOK_SPM) {
                if (!buf_append(&piece, "\xE2\x96\x81", 3)) break;   /* U+2581 */
            } else {
                if (!buf_append(&piece, "\xC4\xA0", 2)) break;       /* U+0120 */
            }
        }
        if (!buf_append(&piece, text + start, flen)) break;
        encode_fragment(tok, piece.data, piece.len, &emit);

        pending_space = 0;
        first_fragment = 0;
    }

    free(piece.data);
    return emit.count;
}

/* --- decoding ------------------------------------------------------------------- */

static int is_control_token(const char *t) {
    static const char *const skip[] = {
        "<s>", "</s>", "<unk>", "<pad>", "<mask>", "<|endoftext|>", "<|begin_of_text|>",
        "<|end_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
        "<|im_start|>", "<|im_end|>",
    };
    for (size_t i = 0; i < sizeof(skip) / sizeof(skip[0]); ++i)
        if (strcmp(t, skip[i]) == 0) return 1;
    return 0;
}

/* GPT-2 maps byte B to codepoint 0x100 + rank(B); this is the inverse order. */
static const unsigned char kGpt2ByteOrder[] = {
    0xC0,0xC1,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xCB,0xCC,0xCD,0xCE,0xCF,
    0xD0,0xD1,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xDB,0xDC,0xDD,0xDE,0xDF,
    0xE0,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xEB,0xEC,0xED,0xEE,0xEF,
    0xF0,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,0xF9,0xFA,0xFB,0xFC,0xFD,0xFE,0xFF,
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x7F,0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,
    0x8F,0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,
    0x9F,0xA0,
};

/* Renders one token into tmp; returns the number of bytes the piece needs. */
static size_t decode_piece(const VG_Tokenizer *tok, const char *text, char *tmp, size_t cap) {
    size_t o = 0;
    if (tok->kind == VG_TOK_SPM) {
        unsigned byte;
        if (parse_byte_token(text, &byte) && byte < 256u) {
            if (o < cap) tmp[o++] = (char)byte;
            return o;
        }
        const unsigned char *p = (const unsigned char *)text;
        while (*p) {
            if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {      /* U+2581 -> space */
                if (o < cap) tmp[o++] = ' ';
                p += 3;
            } else {
                if (o < cap) tmp[o++] = (char)*p;
                ++p;
            }
        }
        return o;
    }
    /* GPT-2 byte-level BPE: reverse the byte<->unicode remap. */
    const unsigned char *s = (const unsigned char *)text;
    while (*s) {
        const unsigned char c = *s;
        if (c < 0x80u) {
            if (o < cap) tmp[o++] = (char)*s;
            ++s;
        } else if ((c & 0xE0u) == 0xC0u && s[1]) {
            const unsigned int cp = ((unsigned int)(c & 0x1Fu) << 6) | (unsigned int)(s[1] & 0x3Fu);
            s += 2;
            if (cp >= 0x100u && cp - 0x100u < sizeof(kGpt2ByteOrder)) {
                if (o < cap) tmp[o++] = (char)kGpt2ByteOrder[cp - 0x100u];
            } else if (cp == 0x10A) { if (o < cap) tmp[o++] = '\n'; }
            else if (cp == 0x120) { if (o < cap) tmp[o++] = ' '; }
            else if (cp == 0x10D) { if (o < cap) tmp[o++] = '\r'; }
            else if (cp < 0x80u) { if (o < cap) tmp[o++] = (char)cp; }
        } else {
            if (o < cap) tmp[o++] = (char)*s;
            ++s;
        }
    }
    return o;
}

/* Returns the bytes written (never more than buf_len-1, always NUL terminated) or, when buf is
 * NULL, the required length. The previous version returned the full length even when it had
 * copied nothing, so callers that used the return value as a write length read past their
 * buffer for any token longer than their buffer. */
size_t vg_tokenizer_decode(const VG_Tokenizer *tok, int32_t id, char *buf, size_t buf_len) {
    if (buf && buf_len) buf[0] = 0;
    if (!tok || id < 0 || (size_t)id >= tok->vocab_count) return 0;
    const char *text = tok->vocab[id].text;
    if (tok->kind == VG_TOK_SPM) {
        /* Prefer the declared token type; fall back to the name list for files without it. */
        if (tok->types) {
            const unsigned char ty = tok->types[id];
            if (ty == 2u || ty == 3u) return 0;          /* UNKNOWN / CONTROL */
        } else if (is_control_token(text)) {
            return 0;
        }
    }

    char tmp[256];
    const size_t needed = decode_piece(tok, text, tmp, sizeof(tmp));
    if (!buf || buf_len == 0) return needed;
    const size_t c = needed < buf_len - 1u ? needed : buf_len - 1u;
    memcpy(buf, tmp, c);
    buf[c] = 0;
    return c;
}
