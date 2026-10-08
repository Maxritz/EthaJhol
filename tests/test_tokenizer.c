#include "vg/tokenizer.h"
#include "vg/gguf.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Prompt used for the reference token-ID vectors. */
#define REF_PROMPT "What is the capital of Belgium?"

static int failures = 0;
static int skips = 0;

static void check(const char *name, const char *model, const char *what, int ok, const char *detail) {
    if (ok) {
        printf("  ok   %-24s %-18s %s\n", name, model, what);
    } else {
        printf("  FAIL %-24s %-18s %s\n", name, model, detail ? detail : "");
        ++failures;
    }
}

typedef struct { int32_t id; const char *want; } DecodeCase;

/* A GGUF token marked tokenizer.ggml.token_type 3 (CONTROL) or 4
 * (USER_DEFINED). These are chat-template markers and must encode atomically. */
typedef struct { const char *text; int32_t id; } SpecialCase;

typedef struct {
    const char  *label;
    const char  *path;
    /* Reference token IDs for REF_PROMPT, produced by an independent
     * implementation of the canonical algorithm (not by this code).
     * NULL when no independent reference exists for that tokenizer. */
    const int32_t *ref_ids;
    size_t         ref_count;
    const DecodeCase *decodes;
    size_t         decode_count;
    /* Expected value of tokenizer.ggml.add_bos_token after normalization.
     * Verified against raw GGUF metadata; see probe_bos.py. */
    int            add_bos;
    const SpecialCase *specials;
    size_t         special_count;
} ModelCase;

static const int32_t ref_llama[]  = { 3923, 374, 279, 6864, 315, 34061, 30 };
static const int32_t ref_qwen[]   = { 3838, 374, 279, 6722, 315, 32961, 30 };
static const int32_t ref_smol[]   = { 3838, 374, 279, 6722, 315, 32961, 30 };

static const DecodeCase dec_llama[] = {
    {  720, " \n"   },  /* C4A0 C48A   -> U+0120 U+010A */
    {  617, " have" },  /* C4A0 + "have" */
    { 1027, " been" },  /* C4A0 + "been" */
    {  198, "\n"    },  /* C48A         -> U+010A */
    {   40, "I"     },  /* 0x49 */
};
static const DecodeCase dec_qwen[] = {
    {  93283, " appending" },
    /* raw token text c3a7c4bbc2bdc3a9c4a7c4b4 reverse-maps to e7 99 bd e9 85 92 */
    { 106345, "\xe7\x99\xbd\xe9\x85\x92" },
};
static const DecodeCase dec_laya[] = {
    { 14392, " inquiry" },
};

/* CONTROL token IDs read straight from the GGUF token table of
 * Llama-3.2-1B-Instruct-Q8_0.gguf (token_type == 3, 256 such tokens). */
static const SpecialCase spec_llama[] = {
    { "<|start_header_id|>", 128006 },
    { "<|end_header_id|>",   128007 },
    { "<|eot_id|>",          128009 },
};

static const ModelCase MODELS[] = {
    /* stories15M: llama SP, bos=1, no add_bos_token key -> default true */
    { "stories15M",  "C:\\Users\\rr\\OneDrive\\Desktop\\vulk\\testdata\\stories15M-q4_0.gguf",
      NULL, 0, NULL, 0, 1, NULL, 0 },
    /* Llama3.2: gpt2 BPE, bos=128000, no add_bos_token key -> default true */
    { "Llama3.2-1B-Inst", "H:\\OLLAMA-Models\\GGUF\\Llama-3.2-1B-Instruct-Q8_0.gguf",
      ref_llama, sizeof(ref_llama) / sizeof(ref_llama[0]),
      dec_llama, sizeof(dec_llama) / sizeof(dec_llama[0]), 1,
      spec_llama, sizeof(spec_llama) / sizeof(spec_llama[0]) },
    /* Qwen3: gpt2 BPE, add_bos_token=0, no bos_token_id -> must NOT prepend */
    { "Qwen3-Q6_K",  "G:\\More-models\\Qwen3-4B-Instruct-2507-Q6_K.gguf",
      ref_qwen, sizeof(ref_qwen) / sizeof(ref_qwen[0]),
      dec_qwen, sizeof(dec_qwen) / sizeof(dec_qwen[0]), 0, NULL, 0 },
    /* Smolcode: gpt2 BPE, bos=151643, no add_bos_token key -> default true */
    { "Smolcode-Q4_K", "G:\\More-models\\smolcode-coder-cpp-1.5b-q4_k_m.gguf",
      ref_smol, sizeof(ref_smol) / sizeof(ref_smol[0]), NULL, 0, 1, NULL, 0 },
    /* ModernBERT WordPiece: no independent GPT-2 reference, invariants only.
     * Laya states add_bos_token=1 explicitly. */
    { "Laya-Q8_0",   "H:\\OLLAMA-Models\\GGUF\\laya-Q8_0.gguf",
      NULL, 0, dec_laya, sizeof(dec_laya) / sizeof(dec_laya[0]), 1, NULL, 0 },
};

/* Decode cases pinned to raw GGUF token bytes for this specific model. */
static void test_known_decodes(const char *label, const ModelCase *mc, const VG_Tokenizer *tok) {
    for (size_t i = 0; i < mc->decode_count; ++i) {
        char buf[128];
        memset(buf, 0, sizeof(buf));
        vg_tokenizer_decode(tok, mc->decodes[i].id, buf, sizeof(buf));
        char detail[256];
        snprintf(detail, sizeof(detail), "id=%d want=\"%s\" got=\"%s\"",
                 mc->decodes[i].id, mc->decodes[i].want, buf);
        check(label, "", "known decode", strcmp(buf, mc->decodes[i].want) == 0, detail);
    }
}

/* Token IDs must match the independent reference exactly, not merely round-trip. */
static void test_reference_ids(const char *label, const ModelCase *mc, const VG_Tokenizer *tok) {
    if (!mc->ref_ids) {
        printf("  skip %-24s %-18s no independent reference\n", label, "");
        ++skips;
        return;
    }
    int32_t ids[256];
    size_t n = vg_tokenizer_encode(tok, REF_PROMPT, ids, sizeof(ids) / sizeof(ids[0]));
    char detail[512];
    int ok = (n == mc->ref_count);
    size_t at = 0;
    at += (size_t)snprintf(detail + at, sizeof(detail) - at, "want");
    for (size_t i = 0; i < mc->ref_count && at < sizeof(detail) - 8; ++i)
        at += (size_t)snprintf(detail + at, sizeof(detail) - at, " %d", mc->ref_ids[i]);
    at += (size_t)snprintf(detail + at, sizeof(detail) - at, " got");
    for (size_t i = 0; i < n && at < sizeof(detail) - 8; ++i)
        at += (size_t)snprintf(detail + at, sizeof(detail) - at, " %d", ids[i]);
    for (size_t i = 0; ok && i < n && i < mc->ref_count; ++i)
        if (ids[i] != mc->ref_ids[i]) ok = 0;
    check(label, "", "reference token ids", ok, detail);
}

/* Size query must be safe, must report the required length, and must not overrun. */
static void test_size_query(const char *label, const VG_Tokenizer *tok) {
    size_t n = vg_tokenizer_vocab_size(tok);
    int bad = 0;
    char first[160];
    first[0] = 0;
    for (size_t id = 0; id < n; ++id) {
        size_t need = vg_tokenizer_decode(tok, (int32_t)id, NULL, 0);
        char small[8];
        memset(small, 0x5A, sizeof(small));
        size_t got = vg_tokenizer_decode(tok, (int32_t)id, small, 2);
        int clobbered = 0;
        for (size_t b = 2; b < sizeof(small); ++b)
            if ((unsigned char)small[b] != 0x5A) clobbered = 1;
        if (got != need || clobbered || small[1] != 0) {
            if (!bad)
                snprintf(first, sizeof(first), "id=%zu need=%zu got=%zu clobber=%d",
                         id, need, got, clobbered);
            bad = 1;
        }
    }
    check(label, "", "size query safe", !bad, first);
    vg_tokenizer_decode(tok, -1, NULL, 0);
    vg_tokenizer_decode(tok, INT32_MAX, NULL, 0);
    check(label, "", "invalid id safe", 1, NULL);
}

/* Encode -> decode must reproduce the input byte-for-byte. */
/* The engine must not prepend BOS unless the model says it should. Qwen3 ships
 * add_bos_token = 0 and no bos_token_id at all, so prepending it feeds token -1
 * into the graph. Verified against raw GGUF metadata. */
static void test_bos_policy(const char *label, const VG_Tokenizer *tok, int want) {
    int got = vg_tokenizer_add_bos(tok);
    char detail[96];
    snprintf(detail, sizeof(detail), "want=%d got=%d (bos_id=%d)", want, got,
             vg_tokenizer_bos_id(tok));
    check(label, "", "add_bos_token policy", got == want, detail);
}

static void test_roundtrip(const char *label, const VG_Tokenizer *tok) {
    /* Inputs every tokenizer in this suite can represent exactly. Whitespace runs
     * and edge padding are excluded: the SentencePiece normalizer collapses them by
     * design, so exact round-trip is not a property those models have. */
    static const char *core[] = {
        REF_PROMPT,
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "1234567890",
        "contractions don't can't it's they're",
        "punct: !@#$%^&*()_+-=[]{};':\",./<>?",
    };
    for (size_t i = 0; i < sizeof(core) / sizeof(core[0]); ++i) {
        char *back = vg_tokenizer_encode_decode(tok, core[i]);
        char detail[600];
        snprintf(detail, sizeof(detail), "input=\"%.70s\" got=\"%.70s\"",
                 core[i], back ? back : "(null)");
        check(label, "", "roundtrip", back && strcmp(back, core[i]) == 0, detail);
        free(back);
    }

    /* Full-byte-coverage inputs. Only GPT-2 byte-level BPE can round-trip these:
     * a SentencePiece vocabulary has no byte fallback, so tab/newline/emoji have no
     * entry and correctly encode to <unk>. Verified against raw GGUF vocabularies. */
    if (!vg_tokenizer_is_byte_bpe(tok)) {
        printf("  skip %-24s %-18s no byte fallback\n", label, "");
        ++skips;
        return;
    }
    static const char *bytes_in[] = {
        "tabs\tand\nnewlines",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "emoji: \xf0\x9f\x99\x82  cjk: \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",
        "  leading and trailing  ",
        "high: \xc3\xa9\xc2\xa0\xc2\xad soft\x7fhyphen",
    };
    for (size_t i = 0; i < sizeof(bytes_in) / sizeof(bytes_in[0]); ++i) {
        char *back = vg_tokenizer_encode_decode(tok, bytes_in[i]);
        char detail[600];
        snprintf(detail, sizeof(detail), "input=\"%.70s\" got=\"%.70s\"",
                 bytes_in[i], back ? back : "(null)");
        check(label, "", "byte-coverage roundtrip", back && strcmp(back, bytes_in[i]) == 0, detail);
        free(back);
    }
}

/* SentencePiece U+2581 must surface as a space, never as a literal glyph. */
static void test_sentencepiece_space(const char *label, const VG_Tokenizer *tok) {
    if (vg_tokenizer_is_byte_bpe(tok)) {
        printf("  skip %-24s %-18s not sentencepiece\n", label, "");
        ++skips;
        return;
    }
    size_t n = vg_tokenizer_vocab_size(tok);
    int leaked = 0, checked = 0;
    char first[128];
    first[0] = 0;
    for (size_t id = 0; id < n && !leaked; ++id) {
        char buf[128];
        memset(buf, 0, sizeof(buf));
        vg_tokenizer_decode(tok, (int32_t)id, buf, sizeof(buf));
        if (strlen(buf) == 0) continue;
        ++checked;
        if (strstr(buf, "\xe2\x96\x81")) {
            snprintf(first, sizeof(first), "id=%zu leaked U+2581: %s", id, buf);
            leaked = 1;
        }
    }
    check(label, "", "U+2581 renders as space", !leaked && checked > 0, first);
}

/* Every single-codepoint byte token must decode to exactly the one raw byte that
 * bytes_to_unicode() mapped it from. This is the real invariant behind the old
 * "no glyph leak" check: scanning decoded output for U+0100..U+0143 is invalid
 * once decoding is correct, because arbitrary raw bytes can coincidentally form
 * byte pairs that look like those codepoints. */
static void test_byte_map(const char *label, const VG_Tokenizer *tok) {
    if (!vg_tokenizer_is_byte_bpe(tok)) {
        printf("  skip %-24s %-18s not byte-bpe\n", label, "");
        ++skips;
        return;
    }
    static const unsigned char fallback[68] = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
        0x20,0x7F,0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x8B,0x8C,0x8D,
        0x8E,0x8F,0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0x9B,0x9C,0x9D,
        0x9E,0x9F,0xA0,0xAD,
    };
    size_t n = vg_tokenizer_vocab_size(tok);
    int bad = 0, checked = 0;
    char first[160];
    first[0] = 0;
    for (size_t id = 0; id < n; ++id) {
        const char *txt = vg_tokenizer_token_text(tok, (int32_t)id);
        if (!txt) continue;
        /* Decode the single UTF-8 codepoint the vocab stores. */
        const unsigned char *s = (const unsigned char *)txt;
        unsigned int cp; int seq;
        if (s[0] < 0x80)                            { cp = s[0];         seq = 1; }
        else if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) { cp = ((s[0] & 0x1Fu) << 6) | (s[1] & 0x3Fu); seq = 2; }
        else { continue; }  /* not a single 1-2 byte codepoint; covered elsewhere */
        if (s[seq] != 0) continue;                  /* more than one codepoint */

        int want = -1;
        if (cp >= 0x100 && cp <= 0x143) want = fallback[cp - 0x100];
        else if (cp >= 0xA1 && cp <= 0xFF) want = (int)cp;
        else if (cp >= 0x21 && cp <= 0x7E) want = (int)cp;
        if (want < 0) continue;
        ++checked;

        char buf[8];
        memset(buf, 0, sizeof(buf));
        size_t got = vg_tokenizer_decode(tok, (int32_t)id, buf, sizeof(buf));
        if (got != 1 || (unsigned char)buf[0] != (unsigned char)want) {
            if (!bad)
                snprintf(first, sizeof(first),
                         "id=%zu cp=U+%04X want byte %02X got len=%zu byte %02X",
                         id, cp, (unsigned char)want, got, (unsigned char)buf[0]);
            bad = 1;
        }
    }
    /* Laya is BERT WordPiece and carries 243 of the 256 byte pieces rather than all
     * of them, so the floor is a sanity check that real byte tokens were exercised,
     * not an exact count. Any actual forward/reverse mismatch is reported above. */
    snprintf(first + (bad ? strlen(first) : 0), sizeof(first) - (bad ? strlen(first) : 0),
             " (checked %d)", checked);
    check(label, "", "byte map identity", !bad && checked >= 200, first);
}

/* Byte-level BPE would shred "<|start_header_id|>" into nine ordinary pieces
 * ("<", "|", "start", ...), so a chat template fed through it produces noise.
 * CONTROL / USER_DEFINED tokens must be matched atomically instead. */
static void test_added_tokens(const char *label, const ModelCase *mc, const VG_Tokenizer *tok) {
    for (size_t i = 0; i < mc->special_count; ++i) {
        int32_t ids[8];
        size_t n = vg_tokenizer_encode(tok, mc->specials[i].text, ids, 8);
        char detail[192];
        snprintf(detail, sizeof(detail), "\"%s\" -> %zu token(s), first=%d; want 1 token = %d",
                 mc->specials[i].text, n, n ? ids[0] : -1, mc->specials[i].id);
        check("added_atomic", label, mc->specials[i].text,
              (n == 1 && ids[0] == mc->specials[i].id), detail);
    }

    /* A concatenated chat-template string must keep each separator atomic and
     * only BPE-split the actual words "user"/"What"/"France". */
    if (mc->special_count >= 3) {
        const char *tmpl = "<|start_header_id|>user<|end_header_id|>What is the capital of France?<|eot_id|>";
        size_t need = vg_tokenizer_encode(tok, tmpl, NULL, 0);
        int32_t *ids = (int32_t *)malloc(need * sizeof(int32_t));
        size_t n = ids ? vg_tokenizer_encode(tok, tmpl, ids, need) : 0;
        int has_bos = 0, has_start = 0, has_endhdr = 0, has_eot = 0;
        for (size_t i = 0; i < n; ++i) {
            if (ids[i] == 128000) has_bos = 1;
            if (ids[i] == 128006) has_start = 1;
            if (ids[i] == 128007) has_endhdr = 1;
            if (ids[i] == 128009) has_eot = 1;
        }
        char d[160]; snprintf(d, sizeof(d), "tmpl n=%zu need=%zu start=%d endhdr=%d eot=%d",
                               n, need, has_start, has_endhdr, has_eot);
        check("added_template", label, "specials present",
              (n == need && has_start && has_endhdr && has_eot), d);
        free(ids);
    }
}

static void run(const ModelCase *mc) {
    VG_GGUF *file = NULL;
    if (vg_gguf_open(mc->path, &file) != VG_OK) {
        printf("SKIP %s: cannot open %s\n", mc->label, mc->path);
        ++skips;
        return;
    }
    VG_Tokenizer *tok = NULL;
    if (vg_tokenizer_load(file, &tok) != VG_OK) {
        printf("SKIP %s: tokenizer load failed\n", mc->label);
        vg_gguf_close(file);
        ++skips;
        return;
    }
    printf("%s: vocab=%zu merges=%zu byte_bpe=%d bos=%d eos=%d\n",
           mc->label, vg_tokenizer_vocab_size(tok), vg_tokenizer_merge_count(tok),
           vg_tokenizer_is_byte_bpe(tok),
           vg_tokenizer_bos_id(tok), vg_tokenizer_eos_id(tok));

    test_known_decodes(mc->label, mc, tok);
    test_reference_ids(mc->label, mc, tok);
    test_size_query(mc->label, tok);
    test_roundtrip(mc->label, tok);
    test_sentencepiece_space(mc->label, tok);
    test_byte_map(mc->label, tok);
    test_bos_policy(mc->label, tok, mc->add_bos);
    test_added_tokens(mc->label, mc, tok);

    vg_tokenizer_free(tok);
    vg_gguf_close(file);
}

int main(void) {
    for (size_t i = 0; i < sizeof(MODELS) / sizeof(MODELS[0]); ++i) run(&MODELS[i]);
    printf("\n%s: %d failure(s), %d skip(s)\n",
           failures ? "FAILED" : "PASSED", failures, skips);
    return failures ? 1 : 0;
}
