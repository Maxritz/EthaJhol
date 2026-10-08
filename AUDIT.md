# AUDIT.md — vg-gguf-engine Repository Audit

## Build System

**Status: PARTIAL**

- CMakeLists.txt: Complete
- Build targets:
  - `vg_core` — static library (gguf.c, tensor_source.c, kv_cache.c, plugin_loader.c, trace.c, quant.c, tokenizer.c, threadpool.c, cpu_ops.c, model_graph.c)
  - `vg_vulkan` — static library (vulkan_backend.cpp) — built when Vulkan SDK found
  - `vg_full` — static library (native_engine.c, lora.c, speculative.c)
  - `vg-engine` — executable (main.c), links vg_core + vg_vulkan + vg_full
  - `vg-run` — executable (run_main.cpp), links vg_full + vg_core
  - `vg-bench`, `vg-tests`, `vg-quant-tests` — test executables

**Build log (2025-09-22):**
```
PASS — cmake configure, build, ctest (2/2 tests pass)
```

## GGUF Loader (src/gguf.c)

**Status: COMPLETE**

- `vg_gguf_open()` — opens file, parses header, reads tensor descriptors and metadata
- `vg_gguf_read()` — 64-bit offset read of tensor data
- `vg_gguf_map()` — memory-maps the entire file for storage tier access
- `vg_gguf_find_tensor()` — hash lookup by tensor name
- `vg_gguf_meta()` — metadata lookup by key

- Entry point: `vg_gguf_open()`
- Called by: `vg_tensor_source_open`, `vg_model_graph_load`, `vg_tokenizer_load`
- Tests: vg-core passes

## Tensor Source / Lazy Loader (src/tensor_source.c)

**Status: COMPLETE**

- `vg_tensor_source_open()` — creates VG_TensorSource with LRU cache
- `vg_tensor_acquire()` — acquires tensor lease (VG_TIER_STORAGE=memory-mapped, VG_TIER_HOST=cached, VG_TIER_PINNED, VG_TIER_DEVICE)
- `vg_tensor_release()` — releases lease, decrements refcount
- `evict_for()` — LRU eviction when host budget exceeded
- Thread-safe via mutex (pthread on Linux, CRITICAL_SECTION on Windows)
- Eviction only evicts tensors with ref==0

- Entry point: `vg_tensor_source_open()`, `vg_tensor_acquire()`, `vg_tensor_release()`
- Called by: `vg_model_graph_load`, `model_graph.c` tensor acquisition, `native_engine.c`
- Tests: vg-core passes
- Known defect: `vg_tensor_acquire` for VG_TIER_STORAGE returns pointer into mapped file without adding to cache entries (ticket=0, no refcounting) — this is intentional for fused kernels but means no eviction tracking

## Model Graph Loader (src/model_graph.c)

**Status: PARTIAL**

- `vg_model_graph_load()` — parses generic architecture metadata (n_layer, n_embd, n_head, n_head_kv, n_ff, n_rot, n_vocab, bos_id, eos_id)
- `vg_model_graph_decode()` — forward pass through transformer layers (QKV projection, attention, FFN, logits)
- `vk_matmul()` — static helper for GPU dispatch (only defined when VG_HAS_VUKNAL)
- `vg_model_graph_set_vulkan()` — sets vk pointer on graph

- Entry point: `vg_model_graph_load()`, `vg_model_graph_decode()`
- Called by: `native_engine.c`, `main.c` (vk_infer path)
- Dependencies: gguf, tensor_source, vulkan_backend (for GPU dispatch)
- Tests: vg-core passes
- Known defect:
  - `vg_model_graph_set_vulkan()` exists but is **never called** by `native_engine.c` or `run_main.cpp`
  - vk_matmul has fallback to CPU if GPU dispatch fails
  - Only supports F32, F16, Q8_0 on GPU (Q4_0 and others return VG_E_UNSUPPORTED)

## Tokenizer (src/tokenizer.c)

**Status: PARTIAL**

- `vg_tokenizer_load()` — parses `tokenizer.ggml.tokens` metadata as text CSV array
- `vg_tokenizer_encode()` — word-based matching with Ġ prefix, byte-fallback
- `vg_tokenizer_decode()` — GPT-2 byte-mapping reverse table
- Loads merges for BPE but does **not** use them in encode (just word-based + byte fallback)

- Entry point: `vg_tokenizer_load()`, `vg_tokenizer_encode()`, `vg_tokenizer_decode()`
- Called by: `native_engine.c`, `main.c`
- Tests: vg-core passes
- Known defect:
  - Assumes GPT-2 BPE format — does not handle SentencePiece or other tokenizer formats
  - Decode uses hardcoded GPT-2 byte reversal table — will produce garbage for non-GPT-2 tokens
  - Laya (modern-bert) produces garbled output due to mismatch

## Native Engine (src/native_engine.c)

**Status: BROKEN (GPU UNWIRED)**

- `vg_generate()` — full generation loop: load model, tokenize, prefill, decode, sample, emit
- **Does not create VG_VK instance** — even when config->use_gpu=1
- **Does not call vg_model_graph_set_vulkan()** — GPU path is never activated
- CPU-only inference path works (model_graph.c falls back to vg_cpu_matmul when g->vk==NULL)

- Entry point: `vg_generate()` (called by vg-run and vg-engine chat mode)
- Called by: `run_main.cpp`, `main.c`
- Dependencies: gguf, tensor_source, model_graph, tokenizer
- Known defect:
  - Missing Vulkan initialization and wiring
  - Repeat penalty / repeat_last_n fields in config are ignored
  - n_threads field in config is ignored

## CUDA/Vulkan Backend (src/vulkan_backend.cpp)

**Status: COMPLETE**

- `vg_vk_open()` — creates Vulkan instance, device, command pool, loads shaders
- `vg_vk_matvec_*` — matmul kernels for f32, f16, i8, q4_0, q8_0
- `vg_vk_rmsnorm()`, `vg_vk_rope()`, `vg_vk_softmax()`, `vg_vk_swiglu()` — full pipeline set
- Loads SPIR-V shaders from `shaders/` directory

- Entry point: `vg_vk_open()`, `vg_vk_matvec_*`, `vg_vk_rmsnorm`, etc.
- Called by: `model_graph.c` (vk_matmul static helper)
- Dependencies: Vulkan SDK
- Tests: vg-core passes (vk_self_test path in main.c)
- Known defect:
  - Only F32, F16, Q8_0 matvecs implemented (no Q4_0 dispatch in vk_matmul despite pipeline existing)

## CPU Operations (src/cpu_ops.c)

**Status: COMPLETE**

- `vg_cpu_matmul()` — handles all quant types via dispatch
- `vg_cpu_rmsnorm()`, `vg_cpu_rope()`, `vg_cpu_softmax()`, `vg_cpu_swiglu()`
- `vg_cpu_ffn()` — full FFN with gate+up+down

- Entry point: Various, called by `model_graph.c`
- Tests: vg-core passes

## KV Cache (src/kv_cache.c)

**Status: PARTIAL**

- `vg_kv_cache_create()` — allocates KV cache
- `vg_kv_cache_store()`, `vg_kv_cache_retrieve()` — simple store/retrieve
- Only supports simple linear cache (no radix/paged KV mentioned in code)

- Entry point: `vg_kv_cache_create()`
- Called by: `model_graph.c` (kv_k, kv_v arrays in VG_ModelGraph struct)
- Known defect: Model graph uses inline `kv_k`/`kv_v` arrays, not the `vg_kv_cache_*` API

## MoE Router (src/model_graph.c)

**Status: MISSING**

- No MoE expert routing logic in model_graph.c
- No expert registry
- No expert residency tracking
- Only single dense forward path

## Scheduler (src/threadpool.c)

**Status: PARTIAL**

- `vg_threadpool_create()` — creates thread pool
- `vg_threadpool_dispatch()` — dispatches tasks
- Simple work queue implementation

- Entry point: `vg_threadpool_create()`, `vg_threadpool_dispatch()`
- Tests: vg-core passes
- Known defect: Threadpool is not used in the inference hot path

## llama.cpp Integration

**Status: MISSING (removed)**

- third_party/llama.cpp was deleted
- No references remain in CMakeLists.txt or source

## Plugin Loader (src/plugin_loader.c)

**Status: PARTIAL**

- `vg_plugin_load()` — loads shared library
- `vg_plugin_lookup()` — finds plugin symbol
- Basic dlopen/dlsym wrapper

- Entry point: `vg_plugin_load()`
- Tests: vg-core passes

## Quant Support (src/quant.c)

**Status: COMPLETE**

- Dequantizer for all GGUF types (F32, F16, Q2_K, Q3_K, Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0, Q2_K, IQ*, etc.)
- Type layout table in gguf.c with all type sizes

- Entry point: `vg_quant_dequant()`
- Tests: vg-quant-tests passes

## Speculative Decoding (src/speculative.c)

**Status: STUB**

- `vg_speculative_init()` — returns success without doing anything
- `vg_speculative_predict()` — returns 0 always

- Entry point: `vg_speculative_init()`, `vg_speculative_predict()`
- Tests: No tests for speculative

## Lora (src/lora.c)

**Status: STUB**

- `vg_lora_init()` — allocates struct, returns success
- `vg_lora_apply()` — no-op, returns success

- Entry point: `vg_lora_init()`, `vg_lora_apply()`
- Tests: No tests for lora

## Radix Cache (src/radix.c)

**Status: MISSING**

- File does not exist
- No radix KV cache implementation

## Executable Entry Points

1. **vg-engine** (main.c):
   - `--vulkan-info` — prints Vulkan device info
   - `--vulkan-self-test` — runs Vulkan matmul test
   - `--vulkan-infer MODEL.gguf [PROMPT]` — runs Vulkan inference (manual, not via vg_generate)
   - `inspect MODEL.gguf` — prints model metadata
   - `diagnostics MODEL.gguf` — prints extended diagnostics
   - `tokenize MODEL.gguf TEXT` — tokenizes text
   - `stream MODEL.gguf TENSOR` — streams tensor data
   - `plugin-info PLUGIN` — displays plugin info
   - `chat MODEL.gguf` — interactive chat (calls vg_generate with use_gpu=0)

2. **vg-run** (run_main.cpp):
   - Simple CLI: `-m MODEL -p PROMPT -n TOKENS --temp T --top-k K --top-p P --threads N --seed N --no-mmap`
   - **Does NOT support `--ngl`** (GPU layers flag)
   - Calls `vg_generate()` with hardcoded `use_mmap=1` (ignores --no-mmap after tensor source opens)

## Build Baseline (2025-09-22)

- Build: PASS (cmake + build both succeed with Vulkan SDK found)
- Tests: PASS (vg-core, vg-quant both pass)
- Smoke test: vg-run produces garbled output on modern-bert, no output on gemma4 models
- Vulkan self-test: PASS (`vg-engine --vulkan-self-test` runs successfully)
