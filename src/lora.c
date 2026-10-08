#include "vg/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *my_strdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *d = (char *)malloc(len);
    if (d) memcpy(d, s, len);
    return d;
}

typedef struct VG_LoraTensor {
    char *name;
    void *data;
    size_t bytes;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
} VG_LoraTensor;

struct VG_LoraAdapter {
    VG_GGUF *file;
    VG_LoraTensor *tensors;
    size_t tensor_count;
    uint32_t rank;
};

VG_Status vg_lora_load(VG_GGUF *file, VG_LoraAdapter **out) {
    if (!file || !out) return VG_E_INVALID;
    *out = NULL;
    VG_LoraAdapter *a = (VG_LoraAdapter *)calloc(1, sizeof(*a));
    if (!a) return VG_E_NOMEM;
    a->file = file;

    /* Find all loraA/loraB tensors and determine rank */
    uint64_t tc = vg_gguf_tensor_count(file);
    size_t cap = 64;
    a->tensors = (VG_LoraTensor *)calloc(cap, sizeof(VG_LoraTensor));
    if (!a->tensors) { free(a); return VG_E_NOMEM; }

    for (uint64_t i = 0; i < tc; ++i) {
        const VG_GGUF_Tensor *t = vg_gguf_tensor_at(file, i);
        if (!t) continue;
        /* Check if it's a LoRA tensor (contains loraA, loraB, or scaling) */
        const char *name = t->name;
        if (strstr(name, ".loraA") || strstr(name, ".loraB") || strstr(name, ".scaling")) {
            if (a->tensor_count >= cap) { cap *= 2; VG_LoraTensor *nt = (VG_LoraTensor *)realloc(a->tensors, cap * sizeof(VG_LoraTensor)); if (!nt) break; a->tensors = nt; }
            VG_LoraTensor *lt = &a->tensors[a->tensor_count];
            lt->name = my_strdup(name);
            lt->n_dims = t->n_dims;
            memcpy(lt->dims, t->dims, sizeof(t->dims));
            lt->ggml_type = t->ggml_type;
            lt->bytes = t->nbytes;
            lt->data = malloc(t->nbytes ? t->nbytes : 1);
            if (lt->data) {
                VG_Status st = vg_gguf_read(file, t, 0, lt->data, t->nbytes);
                if (st != VG_OK) { free(lt->data); lt->data = NULL; }
            }
            ++a->tensor_count;
        }
    }

    /* Determine rank from first loraA tensor: dims[0] is rank */
    for (size_t i = 0; i < a->tensor_count; ++i) {
        if (strstr(a->tensors[i].name, ".loraA") && a->tensors[i].n_dims >= 2) {
            a->rank = (uint32_t)a->tensors[i].dims[0];
            break;
        }
    }

    *out = a;
    return VG_OK;
}

void vg_lora_free(VG_LoraAdapter *a) {
    if (!a) return;
    for (size_t i = 0; i < a->tensor_count; ++i) { free(a->tensors[i].name); free(a->tensors[i].data); }
    free(a->tensors);
    free(a);
}

uint32_t vg_lora_rank(const VG_LoraAdapter *a) { return a ? a->rank : 0; }

const void *vg_lora_tensor(const VG_LoraAdapter *a, const char *name, size_t *out_size) {
    if (!a || !name) return NULL;
    for (size_t i = 0; i < a->tensor_count; ++i) {
        if (strcmp(a->tensors[i].name, name) == 0) {
            if (out_size) *out_size = a->tensors[i].bytes;
            return a->tensors[i].data;
        }
    }
    return NULL;
}

VG_Status vg_lora_apply(const VG_LoraAdapter *adapter, const char *tensor_name,
                        float *weight, uint32_t out_dim, uint32_t in_dim, float scale) {
    if (!adapter || !tensor_name || !weight) return VG_E_INVALID;
    /* Find A and B matrices */
    char name_a[256], name_b[256], name_s[256];
    snprintf(name_a, sizeof(name_a), "%s.loraA", tensor_name);
    snprintf(name_b, sizeof(name_b), "%s.loraB", tensor_name);
    snprintf(name_s, sizeof(name_s), "%s.scaling", tensor_name);

    size_t a_size = 0, b_size = 0;
    const float *lora_a = (const float *)vg_lora_tensor(adapter, name_a, &a_size);
    const float *lora_b = (const float *)vg_lora_tensor(adapter, name_b, &b_size);
    if (!lora_a || !lora_b) return VG_E_INVALID;

    const float *scaling = (const float *)vg_lora_tensor(adapter, name_s, NULL);
    float actual_scale = scale * (scaling ? *scaling : 1.0f);

    uint32_t rank = adapter->rank;
    if (rank == 0) return VG_E_FORMAT;

    /* weight += scale * B @ A
     * A: [rank, in_dim], B: [out_dim, rank]
     * For each output row i, for each input col j: weight[i][j] += scale * sum_k(B[i][k] * A[k][j]) */
    for (uint32_t i = 0; i < out_dim; ++i) {
        for (uint32_t j = 0; j < in_dim; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < rank; ++k) {
                acc += lora_b[i * rank + k] * lora_a[k * in_dim + j];
            }
            weight[i * in_dim + j] += actual_scale * acc;
        }
    }
    return VG_OK;
}
