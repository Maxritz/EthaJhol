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

/* 1 when the model wants BOS prepended to the prompt, 0 when it does not.
 * Driven by tokenizer.ggml.add_bos_token, forced to 0 when no bos id exists. */
int vg_tokenizer_add_bos(const VG_Tokenizer *tok);
int32_t vg_tokenizer_eos_id(const VG_Tokenizer *tok);

/* Number of BPE merge rules loaded (0 means no merges available). */
size_t vg_tokenizer_merge_count(const VG_Tokenizer *tok);

/* 1 if the vocabulary uses GPT-2 byte-level BPE encoding, 0 otherwise. */
int vg_tokenizer_is_byte_bpe(const VG_Tokenizer *tok);

/* Raw vocabulary text for a token ID (the stored piece, before any decoding).
 * Returns NULL for out-of-range IDs. */
const char *vg_tokenizer_token_text(const VG_Tokenizer *tok, int32_t id);

/* Decode a whole token sequence into one string, stripping the SentencePiece
 * add_dummy_prefix space. Caller frees *out. Returns 0 on success. */
int vg_tokenizer_decode_sequence(const VG_Tokenizer *tok, const int32_t *ids, size_t n, char **out);

/* Encode and return the exact decoded string for `text`. Caller frees. Returns NULL on failure. */
char *vg_tokenizer_encode_decode(const VG_Tokenizer *tok, const char *text);

#ifdef __cplusplus
}
#endif
#endif
