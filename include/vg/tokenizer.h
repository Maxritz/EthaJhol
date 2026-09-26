#ifndef VG_TOKENIZER_H
#define VG_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_Tokenizer VG_Tokenizer;

/* Build a tokenizer from GGUF metadata (tokenizer.ggml.tokens, .scores, .merges, .model). */
VG_Status vg_tokenizer_load(const VG_GGUF *file, VG_Tokenizer **out);
void vg_tokenizer_free(VG_Tokenizer *tok);

/* Encode text to token IDs. Returns number of tokens written, or required count if out is NULL. */
size_t vg_tokenizer_encode(const VG_Tokenizer *tok, const char *text, int32_t *ids, size_t max_ids);

/* Decode a token ID to its string piece. Returns length written, or required length if buf is NULL. */
size_t vg_tokenizer_decode(const VG_Tokenizer *tok, int32_t id, char *buf, size_t buf_len);

/* Number of tokens in the vocabulary. */
size_t vg_tokenizer_vocab_size(const VG_Tokenizer *tok);

/* BOS/EOS token IDs. */
int32_t vg_tokenizer_bos_id(const VG_Tokenizer *tok);
int32_t vg_tokenizer_eos_id(const VG_Tokenizer *tok);

#ifdef __cplusplus
}
#endif
#endif
