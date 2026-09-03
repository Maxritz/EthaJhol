#include "vg/tokenizer.h"
#include <stdlib.h>
#include <string.h>

typedef struct { char *text; int32_t id; float score; } VG_VocabEntry;
typedef struct { char *a; char *b; int rank; } VG_Merge;

struct VG_Tokenizer {
    VG_VocabEntry *vocab;
    size_t vocab_count;
    VG_Merge *merges;
    size_t merge_count;
    int32_t bos_id;
    int32_t eos_id;
};

static int32_t find_token(const VG_Tokenizer *tok, const char *text) {
    for (size_t i = 0; i < tok->vocab_count; ++i)
        if (strcmp(tok->vocab[i].text, text) == 0) return tok->vocab[i].id;
    return -1;
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

    /* Load vocab from tokenizer.ggml.tokens metadata array text representation.
     * GGUF stores the token list as a metadata array; we parse it from the
     * array-text form that vg_gguf_meta returns for array types. */
    const char *tokens_str = vg_gguf_meta(file, "tokenizer.ggml.tokens");
    if (!tokens_str) { free(tok); return VG_E_FORMAT; }

    /* Count tokens by counting '['-delimited entries separated by commas. */
    size_t cap = 256;
    tok->vocab = (VG_VocabEntry *)malloc(cap * sizeof(VG_VocabEntry));
    if (!tok->vocab) { free(tok); return VG_E_NOMEM; }
    tok->vocab_count = 0;

    const char *p = tokens_str;
    if (*p == '[') ++p;
    while (*p && *p != ']') {
        while (*p == ',' || *p == ' ') ++p;
        if (*p == ']') break;
        if (*p != '"') break;
        ++p;
        const char *start = p;
        while (*p && *p != '"') { if (*p == '\\') ++p; ++p; }
        size_t len = (size_t)(p - start);
        if (*p == '"') ++p;
        if (tok->vocab_count >= cap) {
            cap *= 2;
            VG_VocabEntry *nv = (VG_VocabEntry *)realloc(tok->vocab, cap * sizeof(VG_VocabEntry));
            if (!nv) { vg_tokenizer_free(tok); *out = NULL; return VG_E_NOMEM; }
            tok->vocab = nv;
        }
        tok->vocab[tok->vocab_count].text = (char *)malloc(len + 1);
        if (!tok->vocab[tok->vocab_count].text) { vg_tokenizer_free(tok); *out = NULL; return VG_E_NOMEM; }
        memcpy(tok->vocab[tok->vocab_count].text, start, len);
        tok->vocab[tok->vocab_count].text[len] = 0;
        tok->vocab[tok->vocab_count].id = (int32_t)tok->vocab_count;
        tok->vocab[tok->vocab_count].score = 0.0f;
        ++tok->vocab_count;
    }

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

    /* Load merges if available. */
    const char *merges_str = vg_gguf_meta(file, "tokenizer.ggml.merges");
    if (merges_str) {
        size_t mcap = 256;
        tok->merges = (VG_Merge *)malloc(mcap * sizeof(VG_Merge));
        if (!tok->merges) { vg_tokenizer_free(tok); *out = NULL; return VG_E_NOMEM; }
        tok->merge_count = 0;
        p = merges_str;
        if (*p == '[') ++p;
        while (*p && *p != ']') {
            while (*p == ',' || *p == ' ') ++p;
            if (*p == ']') break;
            if (*p != '"') break;
            ++p;
            const char *start = p;
            while (*p && *p != '"') { if (*p == '\\') ++p; ++p; }
            size_t len = (size_t)(p - start);
            if (*p == '"') ++p;
            /* Split on ' ' separator */
            const char *sep = start;
            while (sep < start + len && *sep != ' ') ++sep;
            size_t a_len = (size_t)(sep - start);
            size_t b_len = (sep < start + len) ? (size_t)(start + len - sep - 1) : 0;
            if (tok->merge_count >= mcap) {
                mcap *= 2;
                VG_Merge *nm = (VG_Merge *)realloc(tok->merges, mcap * sizeof(VG_Merge));
                if (!nm) { vg_tokenizer_free(tok); *out = NULL; return VG_E_NOMEM; }
                tok->merges = nm;
            }
            tok->merges[tok->merge_count].a = (char *)malloc(a_len + 1);
            tok->merges[tok->merge_count].b = (char *)malloc(b_len + 1);
            if (!tok->merges[tok->merge_count].a || !tok->merges[tok->merge_count].b) {
                vg_tokenizer_free(tok); *out = NULL; return VG_E_NOMEM;
            }
            memcpy(tok->merges[tok->merge_count].a, start, a_len);
            tok->merges[tok->merge_count].a[a_len] = 0;
            memcpy(tok->merges[tok->merge_count].b, sep + 1, b_len);
            tok->merges[tok->merge_count].b[b_len] = 0;
            tok->merges[tok->merge_count].rank = (int)tok->merge_count;
            ++tok->merge_count;
        }
        /* Sort merges by rank for binary search during encoding. */
        /* Actually rank IS the order, so no sort needed. */
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
    free(tok);
}

size_t vg_tokenizer_vocab_size(const VG_Tokenizer *tok) { return tok ? tok->vocab_count : 0; }

/* GPT-2 word-level tokenizer: split on whitespace/punctuation boundaries, look up in vocab. */
size_t vg_tokenizer_encode(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids) {
    if (!tok || !text) return 0;
    size_t text_len = strlen(text);
    if (text_len == 0) return 0;

    size_t ntokens = 0;
    const char *p = text;

    /* Try whole-word matching first, fall back to byte-level. */
    while (*p) {
        /* Skip whitespace, emit space prefix token if mid-string */
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
            if (p > text) {
                /* Emit Ġ + next word approach: we'll handle it by trying the space-prefixed version */
            }
            ++p;
            continue;
        }

        /* Find word boundary: alphanumeric run or single punctuation */
        const char *word_start = p;
        int is_punct = (*p < 'A' || (*p > 'Z' && *p < 'a') || (*p > 'z')) && *p < '0x80';
        if (is_punct) {
            ++p;
        } else {
            while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') ++p;
        }
        size_t word_len = (size_t)(p - word_start);

        /* Try: space-prefixed word first (GPT-2 Ġ prefix for words after space) */
        char buf[256];
        int32_t id = -1;
        if (word_len < sizeof(buf) - 2) {
            buf[0] = '\xC4'; buf[1] = '\xA0'; /* Ġ = U+0120 = C4 A0 in UTF-8 */
            memcpy(buf + 2, word_start, word_len);
            buf[2 + word_len] = 0;
            id = find_token(tok, buf);
        }
        /* Try without prefix */
        if (id < 0 && word_len < sizeof(buf)) {
            memcpy(buf, word_start, word_len);
            buf[word_len] = 0;
            id = find_token(tok, buf);
        }
        /* Fall back to byte tokens */
        if (id < 0) {
            for (size_t b = 0; b < word_len; ++b) {
                char byte_str[2] = { word_start[b], 0 };
                int32_t bid = find_token(tok, byte_str);
                if (bid < 0) bid = 0;
                if (ids && ntokens < max_ids) ids[ntokens] = bid;
                ++ntokens;
            }
        } else {
            if (ids && ntokens < max_ids) ids[ntokens] = id;
            ++ntokens;
        }
    }
    return ntokens;
}

/* GPT-2 byte-level BPE decode: maps UTF-8 encoded Unicode back to original bytes. */
size_t vg_tokenizer_decode(const VG_Tokenizer *tok, int32_t id, char *buf, size_t buf_len) {
    if (!tok || id < 0 || (size_t)id >= tok->vocab_count) { if (buf && buf_len > 0) buf[0] = 0; return 0; }
    const char *text = tok->vocab[id].text;

    /* Decode GPT-2 byte-fallback tokens: map UTF-8 encoded Unicode back to bytes.
     * GPT-2 maps byte B to Unicode codepoint 0x100 + rank(B), where rank order is:
     *   0x20(sp),0x21(!)..0x7E(~),0xA1(¡),...,0xFF,0x01,0x02,...0x1F,0x7F,0x80..0xA0
     * We detect 2-byte UTF-8 (C2-C3 prefix) and decode back to the original byte. */
    size_t out = 0;
    const char *s = text;
    while (*s && out < buf_len - 1) {
        unsigned char c = (unsigned char)*s;
        if (c < 0x80) {
            /* ASCII: space is stored as 'Ġ' (2-byte), single chars are literal */
            buf[out++] = *s++;
        } else if ((c & 0xE0) == 0xC0) {
            /* 2-byte UTF-8: decode to codepoint */
            unsigned int cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
            s += 2;
            if (cp == 0x10A) buf[out++] = '\n';       /* Ċ = newline */
            else if (cp == 0x120) buf[out++] = ' ';    /* Ġ = space */
            else if (cp == 0x10D) buf[out++] = '\r';   /* č = carriage return */
            else if (cp == 0x10B) buf[out++] = '\t';   /* ċ = tab (some variants) */
            else if (cp >= 0x100 && cp <= 0x17F) {
                /* Generic GPT-2 byte mapping reverse:
                 * The mapping table is: bytes 0x21-0x7E map to themselves (no encoding),
                 * remaining bytes map to 0x100+ in a fixed order.
                 * Order: 0xC0-0xFF first, then 0x00-0x1F, 0x7F, 0x80-0xBF */
                static const unsigned char gpt2_byte_order[] = {
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
                unsigned int idx = cp - 0x100;
                if (idx < sizeof(gpt2_byte_order)) buf[out++] = (char)gpt2_byte_order[idx];
                else buf[out++] = '?';
            } else {
                /* Other multi-byte: pass through as-is */
                buf[out++] = (char)(cp > 0xFF ? '?' : cp);
            }
        } else if ((c & 0xF0) == 0xE0) {
            /* 3-byte UTF-8: pass through */
            buf[out++] = *s++;
            if (*s) { buf[out++] = *s++; }
            if (*s) { buf[out++] = *s++; }
        } else {
            buf[out++] = *s++;
        }
    }
    if (out < buf_len) buf[out] = 0;
    return out;
}
