#include "vg/gguf.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    VG_GGUF *g = NULL;
    if (vg_gguf_open("/home/rr/models/llama-3.2-1b-q8.gguf", &g) != VG_OK) return 1;
    const char *v = vg_gguf_meta(g, "tokenizer.ggml.tokens");
    if (v) { printf("len=%zu\n", strlen(v)); printf("first 500 chars: %.500s\n", v); }
    else printf("NULL\n");
    vg_gguf_close(g);
    return 0;
}
