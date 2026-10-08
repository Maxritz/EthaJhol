# ARCHITECTURE_GAP.md — Intended vs Actual Architecture

## Intended Architecture

```text
GGUF
 ↓
GGUF metadata
 ↓
Tensor manifest
 ↓
File-range storage
 ↓
Host residency (bounded LRU)
 ↓
Async IO
 ↓
VRAM residency (bounded LRU)
 ↓
Vulkan tensor execution
 ↓
Dense execution
 ↓
MoE router
 ↓
Expert residency lookup
 ↓
Expert prefetch
 ↓
Expert dispatch
 ↓
KV cache (paged)
 ↓
Scheduler (batched)
 ↓
Token generation
```

## Actual Implementation Status

### Layer 1: GGUF File Format
**Status: COMPLETE**

- `gguf.c` correctly parses GGUF v2/v3 headers, tensor descriptors (64-bit offsets/sizes), and metadata
- Validates tensor ranges: `offset + size <= file_size` checked at gguf.c:264
- Tensor type layout table supports all standard GGUF quant types (0-30, 39-42)

```text
GGUF file
    ↓
gguf.c: vg_gguf_open → parse_index → 64-bit offsets validated
    ↓
VG_GGUF struct: tensor_count, tensors[], meta[], hash tables
    ↓
vg_gguf_find_tensor() — O(1) hash lookup by name
    ↓
vg_gguf_read() — seeks to data_base + data_offset + off → fread
    ↓
vg_gguf_map() — mmap entire file for VG_TIER_STORAGE
```

### Layer 2: Tensor Manifest
**Status: COMPLETE**

- `VG_GGUF_Tensor` struct holds: name, n_dims, dims[4], ggml_type, data_offset, nbytes
- Tensor hash table provides O(1) lookup by name

### Layer 3: File-Range Storage
**Status: COMPLETE**

- `vg_gguf_read()` uses 64-bit `seek64` to read arbitrary ranges
- `vg_gguf_map()` memory-maps the file for zero-copy storage tier access

### Layer 4: Host Residency (bounded)
**Status: COMPLETE**

- `tensor_source.c` implements LRU cache with `host_budget_bytes`
- Thread-safe via mutex
- Eviction only removes tensors with ref==0
- Oversized tensors get transient lease

### Layer 5: Async IO
**Status: PARTIAL**

- `VG_TensorSourceConfig.io_workers` field exists but no async IO thread pool is created
- `vg_tensor_prefetch` — synchronous prefetch
- `vg_tensor_stream()` — synchronous streaming read loop
- No background IO threads

### Layer 6: VRAM Residency (bounded)
**Status: FIXED**

- `vulkan_backend.cpp` has `device_budget_bytes`, `bytes_device` tracking
- `vk_matmul` in model_graph.c dispatches matvec to GPU
- `native_engine.c` now calls `vg_vk_open()` + `vg_model_graph_set_vulkan()` when config->use_gpu=1

### Layer 7: Vulkan Tensor Execution
**Status: COMPLETE**

- q_matmul() in model_graph.c supports F32, F16, Q8_0 on GPU
- Q4_0 dispatch missing in vk_matmul (line 574 returns VG_E_UNSUPPORTED)

### Layer 8: Dense Execution
**Status: COMPLETE**

- `vg_model_graph_decode()` implements full transformer forward: QKV proj → attention → FFN → logits
- Supports GQA, RMS norm, RoPE, SwiGLU

### Layer 9: MoE Router
**Status: MISSING**

- No MoE routing logic exists

### Layer 10: Expert Residency
**Status: MISSING**

### Layer 11: Expert Prefetch
**Status: MISSING**

### Layer 12: KV Cache (paged)
**Status: PARTIAL**

- `kv_cache.c` has `vg_kv_*` APIs (KVStore, KVSequence, Radix)
- BUT model_graph.c uses inline kv_k[], kv_v[] arrays, not the paged API

### Layer 13: Scheduler (batched)
**Status: STUB**

- threadpool.c exists but not used in inference path

## Gap Summary

| Gap | Location | Impact |
| --- | --- | --- |
| Async IO not implemented | tensor_source.c:io_workers unused | IO is synchronous |
| Q4_0 GPU dispatch missing | model_graph.c:574 | Only F32/F16/Q8_0 on GPU |
| No MoE support | model_graph.c | Cannot route MoE experts |
| KV cache not paged | kv_cache.c vs model_graph.c | Inline arrays, no radix |
| Tokenizer mismatch | tokenizer.c:38-265 | Garbled output for non-GPT-2 |
| vg_generate returns VG_OK on failure | native_engine.c:181 | Exit code 0 even on error |

## Critical Path (Dependency-Ordered)

```text
Phase 6 (FIXED: Vulkan wiring) — native_engine.c now calls vg_vk_open
  ↓
Phase 4 (Async IO) — io_workers field exists, needs thread pool implementation
  ↓
Phase 13 (End-to-end) — fix tokenizer type detection + error propagation
```
