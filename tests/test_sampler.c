#include "vg/full_engine.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(const char *name, int ok, const char *detail) {
    printf("%s %-28s %s\n", ok ? "ok  " : "FAIL", name, ok ? "" : detail);
    if (!ok) ++failures;
}

int main(void) {
    /* 1. Zero penalty -> no-op (logits untouched). */
    float a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float a0[8]; memcpy(a0, a, sizeof(a));
    int32_t h1[1] = {2};
    vg_apply_repetition_penalty(a, 8, h1, 1, 1.0f);
    int same = 1; for (int i = 0; i < 8; ++i) same &= (a[i] == a0[i]);
    check("no-op @ penalty 1.0", same, "logits changed with penalty 1.0");

    /* 2. penalty > 1 only lowers tokens present in history; untouched tokens
     *    stay identical. */
    float b[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b0[8]; memcpy(b0, b, sizeof(b));
    int32_t h3[3] = {0, 3, 7};
    vg_apply_repetition_penalty(b, 8, h3, 3, 1.2f);
    check("history token penalized down",
          (b[0] < b0[0]) && (b[3] < b0[3]) && (b[7] < b0[7]), "penalized token not reduced");
    check("non-history token unchanged",
          (b[1] == b0[1]) && (b[2] == b0[2]) && (b[4] == b0[4]) && (b[5] == b0[5]) && (b[6] == b0[6]),
          "a non-repeated token was altered");
    check("repeated token penalized twice",
          (b[3] == b0[3] - 1.2f) && (b[0] == b0[0] - 1.2f) && (b[7] == b0[7] - 1.2f),
          "expected single subtract per occurrence");

    /* 3. Out-of-range ids are ignored, must not touch any logit / not crash. */
    float c[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float c0[4]; memcpy(c0, c, sizeof(c));
    int32_t bad[4] = {-1, 100, 4, -5};
    vg_apply_repetition_penalty(c, 4, bad, 4, 2.0f);
    int ok3 = (c[0] == c0[0]) && (c[1] == c0[1]) && (c[2] == c0[2]) && (c[3] == c0[3]);
    check("out-of-range ids ignored", ok3, "an out-of-range id altered logits");

    /* 4. NULL/zero-length history is a no-op. */
    float d[4] = {9, 9, 9, 9};
    float d0[4]; memcpy(d0, d, sizeof(d));
    int32_t h0[1] = {0};
    vg_apply_repetition_penalty(d, 4, NULL, 0, 1.5f);
    vg_apply_repetition_penalty(d, 4, h0, 0, 1.5f);
    int ok4 = 1; for (int i = 0; i < 4; ++i) ok4 &= (d[i] == d0[i]);
    check("null/zero history no-op", ok4, "degraded with empty history");

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
