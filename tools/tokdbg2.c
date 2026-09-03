#include "vg/gguf.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
int main(void) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open("/home/rr/models/llama-3.2-1b-q8.gguf", &g) != VG_OK) { fprintf(stderr, "open failed\n"); return 1; }
    /* Check all meta keys */
    printf("tensor_count: %llu\n", (unsigned long long)vg_gguf_tensor_count(g));
    /* Try to find tokenizer keys */
    const char *keys[] = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "general.architecture",
        NULL
    };
    for (int i = 0; keys[i]; ++i) {
        const char *v = vg_gguf_meta(g, keys[i]);
        if (v) printf("meta[%s]: len=%zu val=%.100s\n", keys[i], strlen(v), v);
        else printf("meta[%s]: NULL\n", keys[i]);
    }
    vg_gguf_close(g);
    return 0;
}
