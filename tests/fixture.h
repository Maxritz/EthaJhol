#ifndef VG_TEST_FIXTURE_H
#define VG_TEST_FIXTURE_H

#include <stdint.h>

/* Synthetic GGUF models used by the engine tests. They are deterministic: every weight is
 * generated from a fixed pattern, so two runs must produce identical logits. */

#define VG_FIX_DENSE_LAYERS 2
#define VG_FIX_DENSE_EMBD   64
#define VG_FIX_DENSE_HEADS  4
#define VG_FIX_DENSE_KV     2
#define VG_FIX_DENSE_FF     128
#define VG_FIX_DENSE_VOCAB  32
#define VG_FIX_DENSE_CTX    64

#define VG_FIX_MOE_LAYERS 1
#define VG_FIX_MOE_EMBD   64
#define VG_FIX_MOE_HEADS  4
#define VG_FIX_MOE_KV     2
#define VG_FIX_MOE_FF     64
#define VG_FIX_MOE_VOCAB  32
#define VG_FIX_MOE_EXPERTS 4
#define VG_FIX_MOE_USED    2

/* BitNet b1.58 shape. The embedding width is a multiple of 256 because TQ2_0 blocks hold
 * 256 weights, which is also what makes the ternary kernels' block loop meaningful. */
#define VG_FIX_BITNET_LAYERS 1
#define VG_FIX_BITNET_EMBD   256
#define VG_FIX_BITNET_HEADS  4
#define VG_FIX_BITNET_KV     2
#define VG_FIX_BITNET_FF     256
#define VG_FIX_BITNET_VOCAB  32
#define VG_FIX_BITNET_CTX    64

/* Dense llama-style model: Q4_0 attention/FFN, F32 embedding and output head. */
int vg_fixture_write_dense(const char *path);
/* Same graph, but one required FFN tensor is stored as IQ2_XXS so the probe must reject it
 * and name that exact tensor. */
int vg_fixture_write_unsupported(const char *path);
/* MoE model with stacked expert tensors and a router projection. */
int vg_fixture_write_moe(const char *path);
/* Same MoE fixture with every expert weight zeroed (regression probe for the FFN path). */
int vg_fixture_write_moe_zero(const char *path);

/* BitNet-style model: "bitnet" architecture, TQ2_0 ternary projections, attn/ffn sub-norms,
 * and one per-tensor output scale, so the whole 1.58-bit code path is exercised end to end. */
int vg_fixture_write_bitnet(const char *path);
/* LoRA adapter targeting that model: two complete targets (one with `.scaling`) and one
 * A-without-B pair that must be ignored. */
int vg_fixture_write_lora(const char *path);

/* Tokenizer-only GGUF: a SentencePiece-style vocabulary (U+2581 word marker), a real merge
 * chain, <0xNN> byte-fallback tokens and one reserved special token. The dense/moe fixtures
 * use 32 single-character ASCII tokens with no merges, so they cannot detect a tokenizer that
 * ignores SentencePiece markers or never applies merges. */
int vg_fixture_write_spm(const char *path);

/* Vocabulary ids in that fixture, so the tests can assert exact output. */
enum {
    VG_FIX_SPM_UNK = 0, VG_FIX_SPM_BOS = 1, VG_FIX_SPM_EOS = 2, VG_FIX_SPM_THE = 3,
    VG_FIX_SPM_CAPITAL_MARKED = 4, VG_FIX_SPM_CAPITAL = 5, VG_FIX_SPM_BYTE_0A = 6,
    VG_FIX_SPM_OF = 7, VG_FIX_SPM_FRANCE = 8, VG_FIX_SPM_IS = 9, VG_FIX_SPM_MARKER = 10,
    VG_FIX_SPM_BYTE_E2 = 17, VG_FIX_SPM_BYTE_82 = 18, VG_FIX_SPM_BYTE_AC = 19,
    VG_FIX_SPM_PARIS = 23
};

#endif
