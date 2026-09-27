#include "vg/tokenizer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *text; int32_t id; float score; } VG_VocabEntry;
typedef struct { char *a; char *b; int rank; } VG_Merge;

/* Open-addressed hash index. Slots hold index+1 into the owning array; 0 means empty. */
typedef struct { size_t *slot; size_t mask; } VG_Index;

struct VG_Tokenizer {
    VG_VocabEntry *vocab;
    size_t vocab_count;
    VG_Index vocab_ix;
    VG_Merge *merges;
    size_t merge_count;
    VG_Index merge_ix;
    int32_t bos_id;
    int32_t eos_id;
    int     add_bos;
    int gpt2_byte_map;
    int add_dummy_prefix;      /* SentencePiece: prepend U+2581 to the input */
    int remove_extra_ws;       /* SentencePiece: collapse runs of spaces */
    int escape_merges;         /* GPT-2 merges arrive as "\uXXXX \uXXXX" */
    /* Tokens GGUF marks token_type 3 (CONTROL) or 4 (USER_DEFINED): chat
     * template markers and other reserved strings. encode matches the longest
     * of these ahead of byte-level BPE, so the array is scanned linearly. */
    char   **added_text;
    int32_t *added_ids;
    size_t   added_count;
    size_t   added_cap;
};

static uint64_t vg_hash_bytes(const char *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= (unsigned char)p[i]; h *= 1099511628211ull; }
    return h;
}

static void vg_index_free(VG_Index *ix) { free(ix->slot); ix->slot = NULL; ix->mask = 0; }

static int vg_index_build(VG_Index *ix, size_t n) {
    vg_index_free(ix);
    size_t cap = 16;
    while (cap < n * 2) cap *= 2;
    ix->slot = (size_t *)calloc(cap, sizeof(size_t));
    if (!ix->slot) return 0;
    ix->mask = cap - 1;
    return 1;
}

static void vg_index_insert(VG_Index *ix, size_t hash, size_t value) {
    if (!ix->slot) return;
    size_t i = (size_t)hash & ix->mask;
    while (ix->slot[i]) i = (i + 1) & ix->mask;
    ix->slot[i] = value + 1;
}

/* Append one codepoint as UTF-8. Returns bytes written (1..4). */
static size_t utf8_put(char *out, unsigned int cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static unsigned int hex4(const char *p) {
    unsigned int v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned int)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned int)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned int)(c - 'A' + 10);
    }
    return v;
}

/* Scan one double-quoted entry of a GGUF metadata array.
 * *pp must point at the opening quote. Writes the unescaped bytes into out
 * (out may be NULL to only measure) and advances *pp past the closing quote.
 * Never reads past the terminating NUL. Returns the unescaped length. */
static size_t scan_quoted(const char **pp, char *out, size_t out_cap) {
    const char *p = *pp;
    if (*p != '"') { *pp = p; return 0; }
    ++p;
    size_t n = 0;
    for (;;) {
        unsigned char c = (unsigned char)*p;
        if (c == 0) { *pp = p; return n; }      /* unterminated: stop, do not run past end */
        if (c == '"') { ++p; break; }
        if (c == '\\') {
            ++p;
            char e = *p;
            if (e == 0) { *pp = p; return n; }
            ++p;
            char tmp[4];
            size_t tl;
            switch (e) {
                case 'n': tmp[0] = '\n'; tl = 1; break;
                case 'r': tmp[0] = '\r'; tl = 1; break;
                case 't': tmp[0] = '\t'; tl = 1; break;
                case 'u': {
                    if (p[0] == 0 || p[1] == 0 || p[2] == 0 || p[3] == 0) { *pp = p; return n; }
                    unsigned int cp = hex4(p);
                    p += 4;
                    /* surrogate pair */
                    if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u' &&
                        p[2] && p[3] && p[4] && p[5]) {
                        unsigned int lo = hex4(p + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            p += 6;
                        }
                    }
                    tl = utf8_put(tmp, cp);
                    break;
                }
                default: tmp[0] = e; tl = 1; break;
            }
            if (out && n < out_cap) memcpy(out + n, tmp, tl < out_cap - n ? tl : out_cap - n);
            n += tl;
            continue;
        }
        if (out && n < out_cap) out[n] = (char)c;
        ++n;
        ++p;
    }
    *pp = p;
    return n;
}

/* Iterate a GGUF array-text string: skips separators, then scans one entry.
 * With advance == 0 the cursor is left untouched, so the caller can measure an
 * entry and then copy it on a second pass without skipping the next one. */
static int array_next(const char **pp, char *out, size_t cap, size_t *out_len, int advance) {
    const char *p = *pp;
    while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
    if (*p == '[') ++p;                       /* leading array marker, once */
    while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
    if (*p == 0 || *p == ']') { if (advance) *pp = p; return 0; }
    if (*p != '"') { if (advance) *pp = p; return 0; }
    size_t n = scan_quoted(&p, out, cap);
    if (advance) *pp = p;
    if (out_len) *out_len = n;
    return 1;
}

static int32_t find_token_n(const VG_Tokenizer *tok, const char *text, size_t len) {
    if (!tok->vocab_ix.slot) return -1;
    size_t i = (size_t)vg_hash_bytes(text, len) & tok->vocab_ix.mask;
    while (tok->vocab_ix.slot[i]) {
        size_t vi = tok->vocab_ix.slot[i] - 1;
        if (strlen(tok->vocab[vi].text) == len && !memcmp(tok->vocab[vi].text, text, len))
            return tok->vocab[vi].id;
        i = (i + 1) & tok->vocab_ix.mask;
    }
    return -1;
}

static int32_t find_token(const VG_Tokenizer *tok, const char *text) {
    return find_token_n(tok, text, strlen(text));
}

/* GGUF stores booleans as type 7, which vg_gguf_meta renders as "true"/"false",
 * while some writers emit plain "1"/"0". Accept both. */
static int meta_bool(const char *s, int dflt) {
    if (!s || !*s) return dflt;
    if (!strcmp(s, "true")  || !strcmp(s, "True"))  return 1;
    if (!strcmp(s, "false") || !strcmp(s, "False")) return 0;
    return strtol(s, NULL, 10) != 0;
}

VG_Status vg_tokenizer_load(const VG_GGUF *file, VG_Tokenizer **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_Tokenizer *tok = (VG_Tokenizer *)calloc(1, sizeof(*tok));
    if (!tok) return VG_E_NOMEM;

    tok->bos_id = -1; tok->eos_id = -1;
    const char *bos_s = vg_gguf_meta(file, "tokenizer.ggml.bos_token_id");
    const char *eos_s = vg_gguf_meta(file, "tokenizer.ggml.eos_token_id");
    if (bos_s) tok->bos_id = (int32_t)strtol(bos_s, NULL, 10);
    if (eos_s) tok->eos_id = (int32_t)strtol(eos_s, NULL, 10);

    /* BOS policy. Absent the key means "add BOS", matching the GGUF convention
     * used by stories15M, Llama3.2 and Smolcode. Qwen3 states add_bos_token=0 and
     * ships no bos_token_id, so prepending BOS would feed token id -1 into the
     * graph and index the embedding table out of bounds. A model with no usable
     * bos id can never have BOS prepended regardless of the flag. */
    tok->add_bos = 1;
    const char *add_bos_s = vg_gguf_meta(file, "tokenizer.ggml.add_bos_token");
    if (add_bos_s) tok->add_bos = meta_bool(add_bos_s, 1);
    if (tok->bos_id < 0) tok->add_bos = 0;

    /* Load vocab from tokenizer.ggml.tokens metadata array text representation.
     * GGUF stores the token list as a metadata array; we parse it from the
     * array-text form that vg_gguf_meta returns for array types. */
    /* Determine tokenizer model type for decode behavior.
     * Default to 0 (raw UTF-8 pass-through). Only enable GPT-2 byte-fallback
     * decoding if model is explicitly "gpt2"/"bpe" AND vocab contains byte-fallback tokens. */
    tok->gpt2_byte_map = 0;
    const char *tok_model = vg_gguf_meta(file, "tokenizer.ggml.model");
    if (tok_model && (!strcmp(tok_model, "gpt2") || !strcmp(tok_model, "bpe"))) {
        tok->gpt2_byte_map = 1;  /* provisional; will validate after vocab load */
    }

    const char *tokens_str = vg_gguf_meta(file, "tokenizer.ggml.tokens");
    if (!tokens_str) { vg_tokenizer_free(tok); return VG_E_FORMAT; }

    /* SentencePiece flags, read once so encode/decode agree on prefix handling. */
    tok->add_dummy_prefix = 1;
    tok->remove_extra_ws = 1;
    {
        const char *adp = vg_gguf_meta(file, "tokenizer.ggml.add_dummy_prefix");
        const char *rew = vg_gguf_meta(file, "tokenizer.ggml.remove_extra_whitespaces");
        if (adp) tok->add_dummy_prefix = (strtol(adp, NULL, 10) != 0);
        if (rew) tok->remove_extra_ws = (strtol(rew, NULL, 10) != 0);
    }

    /* Count tokens by counting '['-delimited entries separated by commas. */
    size_t cap = 256;
    tok->vocab = (VG_VocabEntry *)malloc(cap * sizeof(VG_VocabEntry));
    if (!tok->vocab) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
    tok->vocab_count = 0;

    /* Two passes: measure, then fill, so no per-entry growth bookkeeping is needed. */
    {
        size_t n = 0;
        const char *q = tokens_str;
        char scratch[1024];
        while (array_next(&q, scratch, sizeof(scratch), NULL, 1)) ++n;
        if (n == 0) { vg_tokenizer_free(tok); return VG_E_FORMAT; }
        cap = n;
        VG_VocabEntry *nv = (VG_VocabEntry *)realloc(tok->vocab, cap * sizeof(VG_VocabEntry));
        if (!nv) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
        tok->vocab = nv;
    }

    const char *p = tokens_str;
    while (tok->vocab_count < cap) {
        size_t need = 0;
        if (!array_next(&p, NULL, 0, &need, 0)) break;
        char *text = (char *)malloc(need + 1);
        if (!text) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
        if (!array_next(&p, text, need + 1, NULL, 1)) { free(text); break; }
        text[need] = 0;
        tok->vocab[tok->vocab_count].text = text;
        tok->vocab[tok->vocab_count].id = (int32_t)tok->vocab_count;
        tok->vocab[tok->vocab_count].score = 0.0f;
        ++tok->vocab_count;
    }
    if (tok->vocab_count == 0) { vg_tokenizer_free(tok); return VG_E_FORMAT; }

    /* Build the vocab hash so encode/decode-time lookups are O(1). */
    if (!vg_index_build(&tok->vocab_ix, tok->vocab_count)) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
    for (size_t i = 0; i < tok->vocab_count; ++i) {
        size_t l = strlen(tok->vocab[i].text);
        vg_index_insert(&tok->vocab_ix, (size_t)vg_hash_bytes(tok->vocab[i].text, l), i);
    }

    /* Collect CONTROL / USER_DEFINED tokens so encode can match them whole.
     * Without this, byte-level BPE shreds chat-template markers such as
     * "<|start_header_id|>" into nine ordinary pieces. tokenizer.ggml.token_type
     * is authoritative (llama.cpp uses the same rule); growers use
     * tok->added_cap so no scope-local capacity can desync from added_count. */
    {
        const char *types_str = vg_gguf_meta(file, "tokenizer.ggml.token_type");
        if (types_str) {
            const char *q = types_str;
            if (*q == '[') ++q;
            for (size_t i = 0; i < tok->vocab_count && *q && *q != ']'; ++i) {
                while (*q == ',' || *q == ' ') ++q;
                if (*q == ']') break;
                long ty = strtol(q, (char **)&q, 10);
                if ((ty == 3 || ty == 4) && tok->vocab[i].text[0] &&
                    strlen(tok->vocab[i].text) >= 2) {
                    if (tok->added_count == tok->added_cap) {
                        size_t ncap = tok->added_cap ? tok->added_cap * 2 : 16;
                        char **nt = (char **)realloc(tok->added_text, ncap * sizeof(char *));
                        if (!nt) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
                        tok->added_text = nt;
                        int32_t *ni = (int32_t *)realloc(tok->added_ids, ncap * sizeof(int32_t));
                        if (!ni) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
                        tok->added_ids = ni;
                        tok->added_cap = ncap;
                    }
                    tok->added_text[tok->added_count] = (char *)tok->vocab[i].text;
                    tok->added_ids[tok->added_count] = (int32_t)i;
                    ++tok->added_count;
                }
            }
        }
    }
    fprintf(stderr, "[vg] tokenizer: control tokens=%zu\n", tok->added_count);

    /* Load scores if available. */
    const char *scores_str = vg_gguf_meta(file, "tokenizer.ggml.scores");
    if (scores_str) {
        p = scores_str;
        if (*p == '[') ++p;
        for (size_t i = 0; i < tok->vocab_count && *p && *p != ']'; ++i) {
            while (*p == ',' || *p == ' ') ++p;
            if (*p == ']') break;
            tok->vocab[i].score = (float)strtod(p, (char **)&p);
        }
    }

    /* Validate GPT-2 byte-fallback assumption: scan vocab for any codepoint in the
     * byte-fallback range. Such codepoints occupy U+0100..U+0143, which serialize as
     * two-byte UTF-8 with lead byte 0xC4 or 0xC5 (never 0xC2/0xC3), and they routinely
     * appear mid-token (e.g. "Ġhave" = C4 A0 68 61 76 65), so the whole token must be
     * scanned rather than only its first codepoint. */
    if (tok->gpt2_byte_map) {
        int has_byte_fallback = 0;
        for (size_t i = 0; i < tok->vocab_count && !has_byte_fallback; ++i) {
            const unsigned char *t = (const unsigned char *)tok->vocab[i].text;
            if (!t) continue;
            for (const unsigned char *s = t; *s; ++s) {
                if ((*s & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
                    unsigned int cp = ((*s & 0x1Fu) << 6) | (s[1] & 0x3Fu);
                    if (cp >= 0x100 && cp <= 0x143) { has_byte_fallback = 1; break; }
                    s += 1;
                } else if ((*s & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
                    s += 2;
                } else if ((*s & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 &&
                           (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
                    s += 3;
                }
            }
        }
        if (!has_byte_fallback) tok->gpt2_byte_map = 0;
    }

    /* Load merges if available. Entries look like "a b". */
    const char *merges_str = vg_gguf_meta(file, "tokenizer.ggml.merges");
    if (merges_str) {
        size_t n = 0;
        const char *q = merges_str;
        char scratch[2048];
        while (array_next(&q, scratch, sizeof(scratch), NULL, 1)) ++n;
        if (n > 0) {
            VG_Merge *mg = (VG_Merge *)calloc(n, sizeof(VG_Merge));
            if (!mg) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
            tok->merges = mg;
            q = merges_str;
            for (size_t i = 0; i < n; ++i) {
                size_t need = 0;
                if (!array_next(&q, NULL, 0, &need, 0)) break;
                char *entry = (char *)malloc(need + 1);
                if (!entry) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
                if (!array_next(&q, entry, need + 1, NULL, 1)) { free(entry); break; }
                entry[need] = 0;
                /* Split on the first space; GPT-2 stores each half already escaped. */
                char *sp = strchr(entry, ' ');
                if (!sp) { free(entry); continue; }
                *sp = 0;
                size_t a_len = strlen(entry);
                size_t b_len = strlen(sp + 1);
                char *a = (char *)malloc(a_len + 1);
                char *b = (char *)malloc(b_len + 1);
                if (!a || !b) { free(entry); free(a); free(b); vg_tokenizer_free(tok); return VG_E_NOMEM; }
                memcpy(a, entry, a_len + 1);
                memcpy(b, sp + 1, b_len + 1);
                /* Some producers double-escape each half as \uXXXX text. */
                if (a_len == 6 && a[0] == '\\' && a[1] == 'u') {
                    char t[4]; size_t tl = utf8_put(t, hex4(a + 2));
                    memcpy(a, t, tl); a[tl] = 0;
                }
                if (b_len == 6 && b[0] == '\\' && b[1] == 'u') {
                    char t[4]; size_t tl = utf8_put(t, hex4(b + 2));
                    memcpy(b, t, tl); b[tl] = 0;
                }
                free(entry);
                mg[tok->merge_count].a = a;
                mg[tok->merge_count].b = b;
                mg[tok->merge_count].rank = (int)tok->merge_count;
                ++tok->merge_count;
            }
        }
        if (tok->merge_count > 0) {
            if (!vg_index_build(&tok->merge_ix, tok->merge_count)) { vg_tokenizer_free(tok); return VG_E_NOMEM; }
            for (size_t i = 0; i < tok->merge_count; ++i) {
                size_t al = strlen(tok->merges[i].a);
                size_t bl = strlen(tok->merges[i].b);
                uint64_t h = vg_hash_bytes(tok->merges[i].a, al);
                h = (h ^ vg_hash_bytes(tok->merges[i].b, bl)) * 1099511628211ull;
                vg_index_insert(&tok->merge_ix, (size_t)h, i);
            }
        }
    }

    *out = tok;
    return VG_OK;
}

void vg_tokenizer_free(VG_Tokenizer *tok) {
    if (!tok) return;
    free(tok->added_text);   /* borrowed pointers into vocab[], not owned */
    free(tok->added_ids);
    for (size_t i = 0; i < tok->vocab_count; ++i) free(tok->vocab[i].text);
    free(tok->vocab);
    vg_index_free(&tok->vocab_ix);
    for (size_t i = 0; i < tok->merge_count; ++i) { free(tok->merges[i].a); free(tok->merges[i].b); }
    free(tok->merges);
    vg_index_free(&tok->merge_ix);
    free(tok);
}

size_t vg_tokenizer_vocab_size(const VG_Tokenizer *tok) { return tok ? tok->vocab_count : 0; }
int32_t vg_tokenizer_bos_id(const VG_Tokenizer *tok) { return tok ? tok->bos_id : -1; }

int vg_tokenizer_add_bos(const VG_Tokenizer *tok) { return tok ? tok->add_bos : 0; }
int32_t vg_tokenizer_eos_id(const VG_Tokenizer *tok) { return tok ? tok->eos_id : -1; }
size_t vg_tokenizer_merge_count(const VG_Tokenizer *tok) { return tok ? tok->merge_count : 0; }
int vg_tokenizer_is_byte_bpe(const VG_Tokenizer *tok) { return tok ? tok->gpt2_byte_map : 0; }

const char *vg_tokenizer_token_text(const VG_Tokenizer *tok, int32_t id) {
    if (!tok || id < 0 || (size_t)id >= tok->vocab_count) return NULL;
    return tok->vocab[id].text;
}

/* Decode a whole token sequence. For SentencePiece vocabularies the first token
 * carries the add_dummy_prefix U+2581 that the encoder prepends, which would
 * otherwise surface as a stray leading space in user-visible text. The prefix is
 * stripped here at sequence level; per-token vg_tokenizer_decode() deliberately
 * keeps U+2581 -> ' ' because that is what the vocabulary entry means.
 * Caller frees *out. Returns 0 on success. */
int vg_tokenizer_decode_sequence(const VG_Tokenizer *tok, const int32_t *ids, size_t n, char **out) {
    if (!tok || !out) return -1;
    *out = NULL;
    if (!ids && n) return -1;

    size_t cap = 64;
    for (size_t i = 0; i < n; ++i) cap += 8;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t need = vg_tokenizer_decode(tok, ids[i], NULL, 0);
        if (at + need + 1 >= cap) {
            size_t ncap = (at + need + 1) * 2;
            char *nb = (char *)realloc(buf, ncap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
            cap = ncap;
        }
        at += vg_tokenizer_decode(tok, ids[i], buf + at, cap - at);
    }
    buf[at] = 0;

    if (!tok->gpt2_byte_map && at > 0 && buf[0] == ' ') {
        const char *t0 = vg_tokenizer_token_text(tok, ids[0]);
        if (t0 && (unsigned char)t0[0] == 0xE2 && (unsigned char)t0[1] == 0x96 &&
            (unsigned char)t0[2] == 0x81) {
            memmove(buf, buf + 1, at);
            --at;
        }
    }
    *out = buf;
    return 0;
}

char *vg_tokenizer_encode_decode(const VG_Tokenizer *tok, const char *text) {
    if (!tok || !text) return NULL;
    size_t cap = strlen(text) * 4 + 64;
    int32_t *ids = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!ids) return NULL;
    size_t n = vg_tokenizer_encode(tok, text, ids, cap);
    char *out = NULL;
    if (vg_tokenizer_decode_sequence(tok, ids, n, &out) != 0) { free(ids); return NULL; }
    free(ids);
    return out;
}

/* Canonical GPT-2 bytes_to_unicode() fallback set: the 68 bytes that are not
 * printable-Latin1, listed in ascending byte order. They map to U+0100..U+0143
 * by table index, and the reverse map in vg_tokenizer_decode() indexes the same
 * table, so the two directions cannot drift apart. */
static const unsigned char gpt2_byte_to_cp[68] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x20,0x7F,0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x8B,0x8C,0x8D,
    0x8E,0x8F,0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0x9B,0x9C,0x9D,
    0x9E,0x9F,0xA0,0xAD,
};

/* Index of a byte inside gpt2_byte_to_cp, or -1 if the byte keeps its own value. */
static int gpt2_fallback_index(unsigned char b) {
    if (b <= 0x20)             return (int)b;          /* 0x00..0x20 -> idx 0..32   */
    if (b >= 0x7F && b <= 0xA0) return 33 + (int)b - 0x7F; /* 0x7F..0xA0 -> idx 33..66 */
    if (b == 0xAD)              return 67;
    return -1;
}

/* Encode one raw byte as its GPT-2 forward-mapped character: bytes 33..126,
 * 161..172 and 174..255 keep their own codepoint, and the remaining 68 bytes take
 * U+0100..U+0143. Codepoints above U+007F are emitted as 2-byte UTF-8, which is
 * how the vocab stores them. Returns bytes written (1 or 2). */
static size_t encode_byte_fwd(unsigned char b, char *out) {
    int idx = gpt2_fallback_index(b);
    unsigned int cp = (idx >= 0) ? (0x100u + (unsigned int)idx) : (unsigned int)b;
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}


/* Find the lowest-rank merge rule matching the adjacent symbol pair (a,b).
 * Uses the merge hash index, so lookup is O(1) rather than O(merge_count). */
static size_t find_merge(const VG_Tokenizer *tok, const char *a, size_t a_len,
                         const char *b, size_t b_len) {
    if (!tok->merge_ix.slot) return SIZE_MAX;
    uint64_t h = vg_hash_bytes(a, a_len);
    h = (h ^ vg_hash_bytes(b, b_len)) * 1099511628211ull;
    size_t i = (size_t)h & tok->merge_ix.mask;
    size_t best = SIZE_MAX;
    while (tok->merge_ix.slot[i]) {
        size_t mi = tok->merge_ix.slot[i] - 1;
        if (strlen(tok->merges[mi].a) == a_len && strlen(tok->merges[mi].b) == b_len &&
            !memcmp(tok->merges[mi].a, a, a_len) && !memcmp(tok->merges[mi].b, b, b_len)) {
            if (best == SIZE_MAX || tok->merges[mi].rank < tok->merges[best].rank) best = mi;
        }
        i = (i + 1) & tok->merge_ix.mask;
    }
    return best;
}

/* One BPE "word": a byte span plus its per-byte forward-mapped pieces.
 * 'cap' bounds the piece arrays and 'bcap' bounds the byte buffer, so each can
 * grow independently; a piece is at most 2 bytes (U+0143 is the highest mapped
 * codepoint), so a piece slot can need more than one byte. */
typedef struct {
    char  *bytes;   /* concatenated forward-mapped pieces, NUL-terminated */
    size_t *off;    /* offset of each piece within bytes */
    size_t *len;    /* length of each piece */
    size_t  n;
    size_t  cap;    /* allocated slots in off/len */
    size_t  bcap;   /* allocated bytes in bytes, including the NUL */
} BpeWord;

static int bpe_word_push(BpeWord *w, const char *piece, size_t plen) {
    size_t at = w->bytes ? strlen(w->bytes) : 0;
    if (at + plen + 1 > w->bcap) {
        size_t ncap = w->bcap ? w->bcap * 2 : 64;
        while (ncap < at + plen + 1) ncap *= 2;
        char *nb = (char *)realloc(w->bytes, ncap);
        if (!nb) return 0;
        w->bytes = nb;
        w->bcap = ncap;
    }
    if (w->n == w->cap) {
        size_t ncap = w->cap ? w->cap * 2 : 16;
        size_t *no = (size_t *)realloc(w->off, ncap * sizeof(size_t));
        if (!no) return 0;
        w->off = no;
        size_t *nl = (size_t *)realloc(w->len, ncap * sizeof(size_t));
        if (!nl) return 0;
        w->len = nl;
        w->cap = ncap;
    }
    memcpy(w->bytes + at, piece, plen);
    w->bytes[at + plen] = 0;
    w->off[w->n] = at;
    w->len[w->n] = plen;
    ++w->n;
    return 1;
}

static void bpe_word_free(BpeWord *w) { free(w->bytes); free(w->off); free(w->len); }

/* Apply BPE merges to a word until no adjacent pair is mergeable. */
static int bpe_word_apply(const VG_Tokenizer *tok, BpeWord *w) {
    if (tok->merge_count == 0) return 1;
    for (;;) {
        size_t best_rank = SIZE_MAX, best_i = SIZE_MAX;
        for (size_t i = 0; i + 1 < w->n; ++i) {
            const char *a = w->bytes + w->off[i];
            size_t al = w->len[i];
            const char *b = w->bytes + w->off[i + 1];
            size_t bl = w->len[i + 1];
            size_t mi = find_merge(tok, a, al, b, bl);
            if (mi == SIZE_MAX) continue;
            if ((size_t)tok->merges[mi].rank < best_rank) { best_rank = (size_t)tok->merges[mi].rank; best_i = i; }
        }
        if (best_i == SIZE_MAX) break;
        const char *a = w->bytes + w->off[best_i];
        size_t al = w->len[best_i];
        const char *b = w->bytes + w->off[best_i + 1];
        size_t bl = w->len[best_i + 1];
        BpeWord nw; memset(&nw, 0, sizeof(nw));
        for (size_t i = 0; i < w->n; ++i) {
            if (i == best_i) {
                char merged[512];
                memcpy(merged, a, al); memcpy(merged + al, b, bl);
                if (!bpe_word_push(&nw, merged, al + bl)) { bpe_word_free(&nw); return 0; }
            } else if (i == best_i + 1) {
                continue;
            } else {
                if (!bpe_word_push(&nw, w->bytes + w->off[i], w->len[i])) { bpe_word_free(&nw); return 0; }
            }
        }
        bpe_word_free(w);
        *w = nw;
    }
    return 1;
}

/* Emit one finalized BPE word's pieces as token IDs. Pieces are stored back to
 * back in a shared buffer, so each must be looked up with an explicit length. */
static size_t emit_word(const VG_Tokenizer *tok, const BpeWord *w, int32_t *ids, size_t max_ids, size_t n) {
    for (size_t i = 0; i < w->n; ++i) {
        int32_t id = find_token_n(tok, w->bytes + w->off[i], w->len[i]);
        if (id < 0) id = 0;
        if (ids && n < max_ids) ids[n] = id;
        ++n;
    }
    return n;
}

/* Split raw text into GPT-2 pre-token words: optional leading space + a run of
 * letters/digits, or a run of non-space symbols. This mirrors the GPT-2 regex
 * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ closely
 * enough for byte-level BPE, and keeps contractions handled by the letter run. */
/* Byte classes for the GPT-2 pre-tokenizer. Bytes >= 0x80 are treated as letters,
 * which is a simplification of \p{L} but avoids pulling in Unicode tables. */
static int tc_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int tc_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int tc_alpha(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80;
}

static size_t vg_tokenizer_encode_body(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids) {
    if (!tok || !text) return 0;
    size_t n = 0;
    const unsigned char *p = (const unsigned char *)text;

    if (!tok->gpt2_byte_map) {
        /* SentencePiece: normalize spaces to U+2581, then greedily take the
         * longest vocab match at each position. */
        char *norm = (char *)malloc(strlen(text) * 3 + 8);
        if (!norm) return 0;
        size_t w = 0;
        if (tok->add_dummy_prefix) w += utf8_put(norm + w, 0x2581);
        int prev_space = 1;
        for (const unsigned char *q = (const unsigned char *)text; *q; ++q) {
            if (*q == ' ' && tok->remove_extra_ws && prev_space) continue;
            prev_space = (*q == ' ');
            if (*q == ' ') w += utf8_put(norm + w, 0x2581);
            else norm[w++] = (char)*q;
        }
        norm[w] = 0;

        char tmp[512];
        const char *s = norm;
        while (*s) {
            size_t remaining = strlen(s);
            size_t need_cp = 1;
            {
                unsigned char lead = (unsigned char)s[0];
                if (lead >= 0xF0) need_cp = 4; else if (lead >= 0xE0) need_cp = 3;
                else if (lead >= 0xC0) need_cp = 2;
            }
            size_t match_len = 0;
            int32_t match_id = -1;
            for (size_t len = need_cp; len <= remaining && len < sizeof(tmp); ++len) {
                memcpy(tmp, s, len);
                tmp[len] = 0;
                int32_t id = find_token(tok, tmp);
                if (id >= 0) { match_id = id; match_len = len; }
            }
            if (match_id < 0) { match_id = 0; match_len = need_cp; }
            if (ids && n < max_ids) ids[n] = match_id;
            ++n;
            s += match_len;
        }
        free(norm);
        return n;
    }

    /* GPT-2 pre-tokenization. Each alternative is a separate run, matching
     * ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+' | '\s+', so letters and
     * digits never merge into one word. */
    while (*p) {
        BpeWord w; memset(&w, 0, sizeof(w));

        /* A single leading space belongs to the following piece ("Ġword"). */
        int lead_space = (*p == ' ');
        if (lead_space) ++p;
        const unsigned char *wstart = p;

        if (tc_alpha(*p))       { while (tc_alpha(*p)) ++p; }
        else if (tc_digit(*p))  { while (tc_digit(*p)) ++p; }
        else if (!tc_space(*p)) { while (*p && !tc_space(*p) && !tc_alpha(*p) && !tc_digit(*p)) ++p; }
        else                    { while (tc_space(*p)) ++p; }

        size_t run = (size_t)(p - wstart);
        if (run == 0 && !lead_space) break;    /* end of input */

        if (lead_space) {
            char sp[4];
            size_t spl = encode_byte_fwd(' ', sp);
            if (!bpe_word_push(&w, sp, spl)) { bpe_word_free(&w); return n; }
        }
        for (size_t i = 0; i < run; ++i) {
            char piece[4];
            size_t pl = encode_byte_fwd(wstart[i], piece);
            if (!bpe_word_push(&w, piece, pl)) { bpe_word_free(&w); return n; }
        }
        if (!bpe_word_apply(tok, &w)) { bpe_word_free(&w); return n; }
        n = emit_word(tok, &w, ids, max_ids, n);
        bpe_word_free(&w);
    }
    return n;
}

/* Longest whole-token match for a CONTROL / USER_DEFINED token at p. Returns the
 * match length (0 = none). Scanning all candidates and keeping the longest makes
 * ordering independent of insertion order (e.g. "<|eot_id|>" beats "<|eot"|>"). */
static size_t match_added_token(const VG_Tokenizer *tok, const char *p, int32_t *id) {
    size_t best = 0;
    int32_t best_id = -1;
    for (size_t i = 0; i < tok->added_count; ++i) {
        const char *s = tok->added_text[i];
        if (!s) continue;
        size_t l = strlen(s);
        if (l > best && strncmp(p, s, l) == 0) { best_id = tok->added_ids[i]; best = l; }
    }
    if (id && best) *id = best_id;
    return best;
}

/* Added tokens are matched first, then each remaining span goes through the
 * normal SentencePiece / GPT-2 path. Control markers are ASCII, so stepping
 * one byte at a time can never split a multi-byte UTF-8 sequence. */
static size_t flush_segment(const VG_Tokenizer *tok, const char *seg, size_t seg_len,
                             int32_t *ids, size_t max_ids, size_t n) {
    if (seg_len == 0) return n;
    char *buf = (char *)malloc(seg_len + 1);
    if (!buf) return n;
    memcpy(buf, seg, seg_len);
    buf[seg_len] = 0;
    size_t cap = (ids && n < max_ids) ? (max_ids - n) : 0;
    int32_t *dst = (ids && n < max_ids) ? (ids + n) : NULL;
    size_t got = vg_tokenizer_encode_body(tok, buf, dst, cap);
    free(buf);
    return n + got;
}

static size_t vg_tokenizer_encode_impl(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids) {
    if (!tok || !text) return 0;
    if (!tok->added_count) return vg_tokenizer_encode_body(tok, text, ids, max_ids);

    size_t n = 0;
    const char *seg = text;
    const char *p = text;
    for (; *p; ) {
        int32_t aid = -1;
        size_t alen = match_added_token(tok, p, &aid);
        if (!alen) { ++p; continue; }
        size_t seg_len = (size_t)(p - seg);
        n = flush_segment(tok, seg, seg_len, ids, max_ids, n);
        if (ids && n < max_ids) ids[n] = aid;
        ++n;
        p += alen;
        seg = p;
    }
    n = flush_segment(tok, seg, (size_t)(p - seg), ids, max_ids, n);
    return n;
}

size_t vg_tokenizer_encode(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids) {
    return vg_tokenizer_encode_impl(tok, text, ids, max_ids);
}

/* Decode a token ID to its string piece. Returns length written, or required length if buf is NULL. */
size_t vg_tokenizer_decode(const VG_Tokenizer *tok, int32_t id, char *buf, size_t buf_len) {
    if (!tok || id < 0 || (size_t)id >= tok->vocab_count) { if (buf && buf_len > 0) buf[0] = 0; return 0; }
    const char *text = tok->vocab[id].text;

    /* SentencePiece / WPM: token text is final UTF-8, but U+2581 encodes a space
     * and must be rendered as one. Leading U+2581 on the first token of a
     * sequence is the dummy prefix; llama.cpp keeps it as a space on decode, so
     * we do the same to stay byte-faithful to the model output. */
    if (!tok->gpt2_byte_map) {
        if (!buf || buf_len == 0) {
            size_t total = 0;
            const unsigned char *s = (const unsigned char *)text;
            while (*s) {
                if ((s[0] == 0xE2) && (s[1] == 0x96) && (s[2] == 0x81)) { total += 1; s += 3; }
                else { total += 1; ++s; }
            }
            return total;
        }
        size_t out = 0;
        const unsigned char *s = (const unsigned char *)text;
        while (*s) {
            char emit;
            if ((s[0] == 0xE2) && (s[1] == 0x96) && (s[2] == 0x81)) { emit = ' '; s += 3; }
            else { emit = (char)*s; ++s; }
            if (out < buf_len) buf[out] = emit;
            ++out;
        }
        buf[out < buf_len ? out : buf_len - 1] = 0;
        return out;
    }

    /* GPT-2 byte-level BPE reverse map, matching llama.cpp's bytes_to_unicode().
     * Uses the shared gpt2_byte_to_cp[] table above so the forward and reverse
     * directions cannot drift apart. Decoding walks the token codepoint-by-codepoint
     * and re-emits the original byte for any codepoint in the byte-fallback range,
     * passing everything else through. */

    size_t out = 0;
    const unsigned char *s = (const unsigned char *)text;
    /* Two-pass: when buf is NULL only count required bytes. */
    int counting = (buf == NULL);
    for (;;) {
        unsigned char c0 = s[0];
        if (c0 == 0) break;
        unsigned int cp;
        int seq;
        if (c0 < 0x80)                            { cp = c0;         seq = 1; }
        else if ((c0 & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { cp = ((c0 & 0x1Fu) << 6) | (s[1] & 0x3Fu); seq = 2; }
        else if ((c0 & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) { cp = ((c0 & 0x0Fu) << 12) | ((s[1] & 0x3Fu) << 6) | (s[2] & 0x3Fu); seq = 3; }
        else if ((c0 & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) { cp = ((c0 & 0x07u) << 18) | ((s[1] & 0x3Fu) << 12) | ((s[2] & 0x3Fu) << 6) | (s[3] & 0x3Fu); seq = 4; }
        else                                       { cp = c0;         seq = 1; } /* malformed: raw byte */

        if (cp >= 0x100 && cp <= 0x143) {
            /* Byte-fallback codepoint: emit the original raw byte. */
            if (!counting && out < buf_len) buf[out] = (char)gpt2_byte_to_cp[cp - 0x100];
            ++out;
        } else if (cp >= 0xA1 && cp <= 0xFF) {
            /* bytes_to_unicode() maps bytes 161..172 and 174..255 to the codepoint
             * with the *same* value, so the reverse direction emits exactly one raw
             * byte equal to the codepoint -- not the 2-byte UTF-8 encoding of it. */
            if (!counting && out < buf_len) buf[out] = (char)cp;
            ++out;
        } else {
            /* Genuine multi-byte codepoint (CJK, emoji, ...): re-emit verbatim. */
            for (int k = 0; k < seq; ++k) {
                if (!counting && out < buf_len) buf[out] = (char)s[k];
                ++out;
            }
        }
        s += seq;
    }
    if (buf && buf_len > 0) buf[out < buf_len ? out : buf_len - 1] = 0;
    return out;
}
