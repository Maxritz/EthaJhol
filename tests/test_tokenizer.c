#include "fixture.h"
#include "vg/gguf.h"
#include "vg/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* These checks are written with explicit comparisons rather than assert(): in a Release build
 * NDEBUG removes assert() bodies, and earlier test files lost every side effect with them. */
static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
} while (0)

#define CHECK_EQ_INT(actual, expected) do { \
    long long a_ = (long long)(actual), e_ = (long long)(expected); \
    if (a_ != e_) { fprintf(stderr, "FAIL %s:%d: %s == %lld, expected %lld\n", \
                             __FILE__, __LINE__, #actual, a_, e_); ++failures; } \
} while (0)

#define CHECK_EQ_STR(actual, expected) do { \
    const char *a_ = (actual), *e_ = (expected); \
    if (!a_ || strcmp(a_, e_) != 0) { \
        fprintf(stderr, "FAIL %s:%d: %s == \"%s\", expected \"%s\"\n", \
                __FILE__, __LINE__, #actual, a_ ? a_ : "(null)", e_); ++failures; } \
} while (0)

static void encode_expect(VG_Tokenizer *tok, const char *text, const int32_t *want, size_t n_want) {
    const size_t n = vg_tokenizer_encode(tok, text, NULL, 0);
    CHECK_EQ_INT(n, n_want);
    if (n != n_want) return;
    int32_t *ids = (int32_t *)malloc(n * sizeof(int32_t));
    if (!ids) { ++failures; return; }
    const size_t written = vg_tokenizer_encode(tok, text, ids, n);
    CHECK_EQ_INT(written, n_want);
    for (size_t i = 0; i < n_want; ++i) {
        if (ids[i] != want[i]) {
            fprintf(stderr, "FAIL: \"%s\" token %zu == %d, expected %d\n", text, i, ids[i], want[i]);
            ++failures;
        }
    }
    free(ids);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "fixture_spm.gguf";
    if (vg_fixture_write_spm(path) != 0) { fprintf(stderr, "cannot write fixture\n"); return 2; }

    VG_GGUF *file = NULL;
    if (vg_gguf_open(path, &file) != VG_OK) { fprintf(stderr, "cannot open fixture\n"); return 2; }
    VG_Tokenizer *tok = NULL;
    if (vg_tokenizer_load(file, &tok) != VG_OK) { fprintf(stderr, "cannot load tokenizer\n"); return 2; }

    CHECK_EQ_INT(vg_tokenizer_vocab_size(tok), 24);

    /* 1. Merges must actually be applied: without them "capital" encodes as c/a/p/i/t/a/l. */
    {
        const int32_t want[] = { VG_FIX_SPM_CAPITAL };
        encode_expect(tok, "capital", want, 1);
    }
    /* 2. The SentencePiece space marker must be chosen, not the GPT-2 one: the marker form
     *    "▁capital" is a distinct vocabulary entry from bare "capital". */
    {
        const int32_t want[] = { VG_FIX_SPM_THE, VG_FIX_SPM_CAPITAL_MARKED };
        encode_expect(tok, "The capital", want, 2);
    }
    {
        const int32_t want[] = { VG_FIX_SPM_CAPITAL_MARKED };
        encode_expect(tok, " capital", want, 1);
    }
    /* 3. Byte fallback for a character that only exists as <0xNN> entries. */
    {
        const int32_t want[] = { VG_FIX_SPM_BYTE_E2, VG_FIX_SPM_BYTE_82, VG_FIX_SPM_BYTE_AC };
        encode_expect(tok, "\xE2\x82\xAC", want, 3);
    }
    /* 4. Reserved special tokens are emitted whole instead of being merged into pieces. */
    {
        const int32_t want[] = { VG_FIX_SPM_BOS };
        encode_expect(tok, "<s>", want, 1);
    }

    /* 5. Decode must translate the marker to a space and <0xNN> to a raw byte. */
    {
        char buf[64];
        size_t n = vg_tokenizer_decode(tok, VG_FIX_SPM_CAPITAL_MARKED, buf, sizeof(buf));
        CHECK_EQ_INT(n, 8);
        CHECK_EQ_STR(buf, " capital");

        n = vg_tokenizer_decode(tok, VG_FIX_SPM_BYTE_0A, buf, sizeof(buf));
        CHECK_EQ_INT(n, 1);
        CHECK_EQ_INT((unsigned char)buf[0], 0x0Au);

        n = vg_tokenizer_decode(tok, VG_FIX_SPM_THE, buf, sizeof(buf));
        CHECK_EQ_STR(buf, "The");

        /* Control tokens contribute no text. */
        n = vg_tokenizer_decode(tok, VG_FIX_SPM_BOS, buf, sizeof(buf));
        CHECK_EQ_INT(n, 0);
        CHECK_EQ_STR(buf, "");
    }
    /* 6. The decoder must never report more bytes than it stored. */
    {
        char small[3];
        memset(small, 'Z', sizeof(small));
        const size_t n = vg_tokenizer_decode(tok, VG_FIX_SPM_CAPITAL_MARKED, small, sizeof(small));
        CHECK_EQ_INT(n, 2);
        CHECK(small[2] == 0);
        CHECK(small[0] == ' ' && small[1] == 'c');
    }
    /* 7. Round trip through encode -> decode (only words the fixture has merge chains for). */
    {
        const char *text = "The capital capital";
        const size_t n = vg_tokenizer_encode(tok, text, NULL, 0);
        CHECK(n > 0);
        int32_t *ids = (int32_t *)malloc((n ? n : 1) * sizeof(int32_t));
        if (ids) {
            vg_tokenizer_encode(tok, text, ids, n);
            char joined[128];
            size_t used = 0;
            joined[0] = 0;
            for (size_t i = 0; i < n && used + 1 < sizeof(joined); ++i) {
                const size_t got = vg_tokenizer_decode(tok, ids[i], joined + used, sizeof(joined) - used);
                used += got;
            }
            joined[used < sizeof(joined) ? used : sizeof(joined) - 1] = 0;
            CHECK_EQ_STR(joined, text);
            free(ids);
        } else {
            ++failures;
        }
    }
    /* 8. A reserved token is recognised even when it is glued to the end of a word, which is
     * exactly how chat templates write it ("...?</s>"). Detecting only at fragment starts
     * silently encoded </s> as '<', '/', 's', '>'. */
    {
        const int32_t want[] = { VG_FIX_SPM_THE, VG_FIX_SPM_EOS, VG_FIX_SPM_CAPITAL };
        encode_expect(tok, "The</s>capital", want, 3);
    }

    /* 9. A short buffer in encode must not be overrun. */
    {
        const int32_t want[] = { VG_FIX_SPM_THE, VG_FIX_SPM_CAPITAL_MARKED };
        int32_t one[1] = { -1 };
        const size_t n = vg_tokenizer_encode(tok, "The capital", one, 1);
        CHECK_EQ_INT(n, 2);
        CHECK_EQ_INT(one[0], want[0]);
    }

    vg_tokenizer_free(tok);
    vg_gguf_close(file);
    remove(path);

    if (failures) { fprintf(stderr, "vg-tokenizer: %d failure(s)\n", failures); return 1; }
    puts("vg-tokenizer: all tests passed");
    return 0;
}
