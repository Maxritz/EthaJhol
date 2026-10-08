# Feature-Complete Gap Assessment

The existing alpha is a substrate with approximately 1,166 lines across the public headers, C99 core, C++ Vulkan wrapper, one compute shader, and smoke tests. It does not yet execute a transformer graph or generate text. The feature-complete target therefore needs a staged runtime above the existing contracts.

| Required capability | Current state | Completion work |
| --- | --- | --- |
| GGUF parsing | Header, scalar/array metadata text, tensor index, common byte-size validation. Hash tables for O(1) lookup, O_DIRECT read path. | Add all GGML quantization layouts, alignment/endian checks, metadata typing, tokenizer extraction, and model-family metadata normalization. |
| Lazy loading | mmap storage leases, range reads, host LRU, transient large-tensor leases, synchronous streaming. | Add asynchronous overlapped I/O, aligned direct-I/O paths, tile/subtensor requests, read-ahead scheduler, pinned staging, and safe concurrent eviction. |
| Tokenization | Native tokenizer from GGUF metadata (tokenizer.ggml.tokens/scores/merges). | Implement SentencePiece/BPE tokenizer loading from GGUF metadata and special-token/chat-template handling. |
| Model graph | Missing. | Add graph IR and model adapters for Llama/Qwen/Mistral/Gemma/DeepSeek-style dense, GQA/MLA, and hybrid/recurrent variants. |
| CPU inference | Missing. | Add F32/F16/BF16/int8/int4 kernels, RMSNorm/RoPE/attention/SwiGLU/MoE, multithreaded scheduling, and numerically checked reference paths. |
| Vulkan inference | One headless int8 matvec dispatch. | Add device-local buffers, staging queues, descriptor arenas, pipeline cache, tiled quantized GEMM, fused pointwise kernels, attention, KV page tables, grouped MoE, and device residency/eviction. |
| Long context | Basic paged KV allocation, sequence fork, radix snapshot, page spill/fault to file-backed storage. | Add per-layer page residency, chunked prefill, hybrid recurrent snapshots, identity namespaces, concurrent radix eviction, and context limits driven by budgets. |
| Generation | Missing. | Add logits pipeline, repetition/frequency/presence penalties, temperature/top-k/top-p/min-p/typical sampling, grammar/stop sequences, streaming callbacks, and deterministic seeds. |
| MoE | No routing or expert execution. | Add router/top-k/top-p policy, disk-backed expert table, expert slot cache, batched/grouped execution, and CPU/Vulkan overlap. |
| Plugins | Versioned loader and example. | Add model/backend/quantizer/storage/sampler interfaces with capability discovery, lifecycle tests, and ABI compatibility rules. |
| Adapters | Missing. | Add LoRA/QLoRA and prompt-adapter hooks with lazy adapter tensors and cache identity isolation. |
| Speculation | Missing. | Add draft-model API, verification batching, acceptance accounting, and fallback behavior. |
| Tooling | CLI inspect/stream/self-test/benchmark, tokenize, diagnostics. | Add `run`, `chat`, `convert`, `bench`, JSON metrics, model diagnostics, and Python/C++ bindings. |
| Release readiness | Linux smoke-tested only. | Add Windows/MSVC presets and CI, Linux GCC/Clang CI, Vulkan validation layers, golden logits, quantization parity tests, fuzzing, packaging, and hardware matrix. |

Feature-complete means at least one dense model family can load a real GGUF file, tokenize a prompt, prefill and decode on CPU, decode through Vulkan when available, stream generated tokens, reuse a radix prefix, and survive a model larger than the configured host/device working sets through lazy tensor residency. Additional model families and exotic GGML types must fail with explicit diagnostics rather than silently producing incorrect output.
