# Architecture Contract

## Purpose

The engine is organized around a strict separation between **what the model is**, **where bytes live**, and **how bytes are executed**. A GGUF file is an indexed immutable source. A tensor source turns indexed ranges into leases or streams. A scheduler chooses when a range is needed. A backend consumes packed bytes and activations. No layer is allowed to assume that a complete model, complete expert bank, or complete context fits in one memory tier.

## Components

| Component | Stable responsibility | Deliberate non-responsibility |
| --- | --- | --- |
| `VG_GGUF` | Validate header, metadata, tensor dimensions, offsets, and quantized byte sizes. | Dequantize weights or construct a model graph. |
| `VG_TensorSource` | Provide storage/host/pinned leases, range reads, streaming callbacks, prefetch, and residency statistics. | Decide model routing or KV eviction. |
| `VG_KVStore` | Own fixed-size KV pages, sequence page references, forks, and page accounting. | Select tokens or implement attention math. |
| `VG_Radix` | Match exact token prefixes and share immutable sequence snapshots. | Treat semantic inputs as token IDs when their payload differs. |
| Backend plugin | Discover devices, allocate buffers, dispatch kernels, and report budgets. | Open model files or own global scheduler state. |
| Model plugin | Translate a GGUF metadata/tensor namespace into graph operations. | Bypass leases and silently materialize all weights. |
| Scheduler | Batch requests, chunk prefill, route experts, issue prefetch, and coordinate page faults. | Change quantization or router semantics without an explicit policy. |

## Oversized model execution

A large model uses a tiered working set:

```text
immutable GGUF / filesystem
        |
        +-- mmap or direct range reads
        |
        +-- host cache with leases and LRU eviction
        |
        +-- pinned transfer buffers
        |
        +-- Vulkan device-local working set
```

The GGUF index is small enough to remain resident. Tensor payloads remain packed. The scheduler first issues read-ahead for the next layer or likely routed experts, then acquires a lease for the current operation. A lease pins the source entry against eviction until the operation releases it. If a requested tensor is larger than the host budget, the source may return a transient lease or stream chunks; it must not fail merely because caching is impossible.

Production storage providers should implement both a random range path and a sequential stream path. The random path is needed for routed experts and tensor slices. The sequential path is needed for dense layer tiles and for filesystems where mapped readahead outperforms many small reads. Linux can add O_DIRECT when alignment and filesystem support are confirmed; Windows can use overlapped reads or a memory mapping. Both remain optimizations over the same offset/length contract.

## Long context

The context window is represented as pages, not a single allocation. A page contains a fixed token span and one or more layer-specific KV planes. The scheduler can therefore:

1. prefill in bounded chunks;
2. retain only the pages referenced by active sequences;
3. share immutable prefix pages across forks;
4. spill cold pages to host or storage tiers;
5. fault pages back before attention consumes them; and
6. report page faults and residency as first-class latency metrics.

The `page_tokens` value is a tuning parameter. It trades page-table overhead against wasted tail space and transfer granularity. The model plugin supplies KV shape and element type; the scheduler supplies the memory budgets.

## Radix prefix reuse

A radix node stores token IDs, a reference to an immutable sequence snapshot, a semantic fingerprint, and a taint bit. A lookup returns only a prefix that is both token-identical and semantically compatible. Images, audio, external tool results, and other payloads that are not fully represented by token IDs set taint and cannot be reused by token comparison alone.

The cache namespace must include model identity, tokenizer identity, adapter identity, RoPE/position policy, and any graph-affecting execution options. Invalidating any of these identities clears or segregates the affected radix tree. The snapshot’s recorded fed-token sequence is authoritative; caller counters are not sufficient because sampled-token feedback and chunk boundaries vary between engines.

Hybrid architectures require two validity domains: attention-page validity and recurrent-state snapshot validity. A prefix may be token-identical but unusable if its recurrent state snapshot is missing or stale. The scheduler should cap the reusable prefix at the deepest node whose required state is live.

## Vulkan backend contract

The backend is headless and compute-only in the alpha. It reports device identity and memory budget, exposes buffer upload/readback, and dispatches a packed int8 matvec. Future kernels should extend this API with:

| Kernel family | Required property |
| --- | --- |
| quantized GEMV/GEMM | dequantize in registers and accumulate without an intermediate float weight matrix; |
| fused gate/up/SwiGLU | read activation once and keep hidden values on device; |
| attention | consume page tables directly and avoid repacking long-context KV; |
| MoE | accept routed expert IDs and issue one grouped dispatch; |
| normalization and sampling | fuse pointwise stages where graph capture and precision policy allow it. |

The backend must support a CPU fallback with identical tensor-source and scheduler semantics. Device-local memory is a cache, not a correctness requirement. When the GPU budget is exceeded, the scheduler evicts device working-set entries and reloads from host or storage.

## Plugin ABI

The C ABI is intentionally opaque. Public structs begin with version and size fields where extensibility matters. Plugins must not depend on private struct layout or process-global singleton state. A C++ backend may use RAII, templates, and specialized kernels internally, but exports only the C ABI. Python bindings can be generated later against the same headers.

## Correctness guardrails

Every load path validates tensor ranges before access. Prepared mmap mode must refuse a format that needs load-time conversion rather than silently allocating an unbounded converted copy. Prefix reuse refuses tainted or identity-mismatched state. Cache-pressure policy may reduce speed, batch size, or residency, but it must not silently change precision, routing, or token semantics.
