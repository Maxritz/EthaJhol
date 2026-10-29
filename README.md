# vg-gguf-engine

A portable GGUF inference engine: a C99 systems layer (GGUF parsing, tiered tensor storage,
paged KV cache, quantized kernels) and a complete transformer graph evaluator. It is
self-contained — no llama.cpp, no Python, no external runtime. Vulkan is used for the
quantized matrix-vector work when a device and an SDK are available.

## What actually runs

| Capability | Status |
| --- | --- |
| GGUF v2/v3 parsing, metadata, typed tensor geometry | Yes — `vg-engine inspect` |
| Dense llama-family models (llama, mistral, qwen2/3, gemma, smollm) | Yes — `vg-run` |
| BitNet b1.58 / ternary models (`bitnet` arch, `attn_sub_norm` + `ffn_sub_norm`, per-tensor `.scale`) | Yes — TQ2_0/TQ1_0/Q2_0/Q1_0 kernels, probe-validated |
| MoE models (mixtral / qwen-moe blocks: router + stacked experts) | Yes — top-k routing with routing weights and an LRU expert cache |
| CPU inference | Yes, row-parallel across all cores (`--threads N`, default = hardware) |
| Vulkan inference | Q4_0, Q8_0 (dp4a), Q4_K, Q6_K, I8, F16, F32 matvec plus fused dense/MoE layer paths — see [docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md) |
| Large models | mmap + lazy paging; tensors are never copied whole |
| Quantized weights | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, **TQ1_0, TQ2_0, Q1_0, Q2_0** — see [docs/QUANTIZATION.md](docs/QUANTIZATION.md) |
| Unsupported quants (IQ1/IQ2/IQ3/IQ4_XS/MXFP4/...) | Rejected up front by the probe, naming the tensor |
| LoRA adapters | `--lora ADAPTER.gguf [--lora-scale S]`; delta applied inside matmul as `y += s·B·(A·x)`, no requantization |
| Speculative decoding | `vg-engine spec MODEL [DRAFT] N`; distribution-correct `min(1, p/q)` acceptance with residual sampling |
| Long context | Model context length, paged KV with optional sliding window |
| Sampling | temperature, top-k, top-p, repeat penalty, deterministic seed |
| Model validation | `vg-engine probe MODEL` — arch, geometry, quant types, required ops, exact failure |
| Tokenizers | SentencePiece BPE (llama/spm/gemma/bert, U+2581 marker, ranked merges, `<0xNN>` byte fallback) and GPT-2 byte-level BPE (U+0120 marker, byte remap) |

### Tokenizer notes

The tokenizer family is chosen from `tokenizer.ggml.model`, falling back to the vocabulary
shape. SentencePiece vocabularies store words with a U+2581 marker (`▁capital`) and build
pieces from the `tokenizer.ggml.merges` chain, so both the marker and the merges are required:
a GPT-2-style U+0120 prefix or an encoder that skips merges shreds every word into single
characters. Decode uses `tokenizer.ggml.token_type` to keep CONTROL/UNKNOWN tokens silent, maps
`▁` back to a space, and expands `<0xNN>` entries to raw bytes. Reserved tokens (`<s>`, `</s>`)
are recognised even when glued to a word, as in `...?</s>`.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Vulkan is compiled only when a SDK is present; otherwise the build uses `src/vulkan_stub.c`
and stays CPU-only. For a sanitizer build:

```sh
cmake -S . -B build-asan -DVG_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan && ctest --test-dir build-asan --output-on-failure
```

Sanitizer flags are applied directory-wide (`add_compile_options`/`add_link_options`), so every
target is instrumented and links the runtime — including `vg_full`, which a per-target list used
to miss and which therefore failed to link.

## Run

```sh
# Dense model, CPU, 2 threads, 256 tokens
./build/vg-run -m model.gguf -p "Write a short story" -n 256 --temp 0.8 --threads 2

# With a LoRA adapter
./build/vg-run -m model.gguf --lora adapter.gguf --lora-scale 1.0 -p "Hello" -n 64

# Speculative decoding (omit DRAFT to use the model as its own draft)
./build/vg-engine spec model.gguf draft.gguf 32

# GPU path (only the ops listed in docs/GPU_SUPPORT.md are offloaded)
./build/vg-run -m model.gguf -p "..." --ngl 1

# What does this model need, and can this build run it?
./build/vg-engine probe model.gguf
./build/vg-engine inspect model.gguf
./build/vg-engine --vulkan-self-test
./build/vg-engine --cpu-infer model.gguf "prompt"

# Matvec throughput for the packed-weight kernels (the decode bottleneck)
./build/vg-mmbench 4096 4096 10
```

`vg_generate()` and `--cpu-infer` both call the probe first and refuse a model with a precise
diagnosis instead of emitting garbage:

```
result: NOT SUPPORTED: tensor 'blk.0.attn_q.weight' uses IQ2_XXS, which has no CPU dequantizer
```

## Tests

`ctest` runs eight suites. The engine suite generates its own deterministic GGUF fixtures at
runtime (dense, MoE, and an intentionally unsupported one), so the runner needs no external
model download:

| Test | Covers |
| --- | --- |
| `vg-core` | paged KV allocation, sequence fork, radix prefix lookup |
| `vg-quant` | legacy packed-block dequantization, Q4_K/Q6_K block math, fused-vs-reference matmul round-trips, ternary golden vectors |
| `vg-tokenizer` | SentencePiece + GPT-2 byte-level BPE, special/control tokens, decode round-trip |
| `vg-sampler` | temperature/top-k/top-p and the additive repetition penalty |
| `vg-tensor` | tensor ranges, read bounds, leases, budget eviction, mmap address stability, chunked streaming, async prefetch incl. error reporting, graph load + MoE router |
| `vg-engine` | half→float conversion, every supported quant type, fused-vs-reference matmul parity, threaded determinism, thread-count resizing, thread-pool completion, probe (accept + exact rejection), dense decode determinism, MoE routing and expert-cache behaviour, in-process `vg_generate()` determinism and probe-gated rejection |
| `vg-full-smoke` | end-to-end `vg-run` generation on a synthetic model, asserting it prints tokens |
| `vg-full-probe` | `vg-engine probe` accepts the same model and reports `SUPPORTED` |

`vg-engine` also writes the dense, MoE and unsupported fixtures that `vg-tensor`, `vg-full-smoke`
and `vg-full-probe` consume (a CTest fixture), so no model download is needed. The tests use
explicit checks rather than `assert()`, because `NDEBUG` in release builds would silently delete
both the assertions and every call made inside one.

## Performance

Measured with `vg-mmbench` on the 2-core x86-64 container used for this change (Release build,
4096×4096 matvec, packed weights):

| Type | bytes/token | ms/op (best, 2 threads) | packed bytes/s |
| --- | --- | --- | --- |
| **TQ2_0** | **4.12 MiB** | **3.40** | 1.19 GiB/s |
| Q4_0 | 9.00 MiB | 5.54 | 1.59 GiB/s |
| Q2_0 | 4.50 MiB | 5.39 | 0.81 GiB/s |
| Q1_0 | 2.25 MiB | 5.80 | 0.38 GiB/s |
| Q8_0 | 17.00 MiB | 6.49 | 2.56 GiB/s |
| Q4_K | 9.00 MiB | 7.07 | 1.24 GiB/s |
| TQ1_0 | 3.38 MiB | 7.77 | 0.42 GiB/s |
| Q6_K | 13.12 MiB | 10.66 | 1.20 GiB/s |
| F32 | 64.00 MiB | 8.74 | 7.15 GiB/s |
| F16 | 32.00 MiB | 17.38 | 1.80 GiB/s |

`vg-mmbench` reports the **minimum** of N iterations: this box is a shared 2-vCPU VM where the
same Q4_0 workload varied between 5.8 and 8.2 ms/op across identical runs, enough to hide a real
15% kernel change. Threading is a genuine 1.84× on this workload (Q4_0: 5.54 ms at 2 threads vs
10.20 ms at 1, Q6_K: 10.66 vs 19.61 ms).

Kernel work in this revision: TQ2_0 went 7.86 → 3.40 ms/op (2.3×) by loading each packed byte
once and running four accumulator chains from it instead of re-reading the byte per bit-plane;
Q1_0 went 8.44 → 6.26 ms by hoisting a per-element divide/modulo out of the bit loop; Q2_0 went
8.92 → 5.39 ms the same way. Per-row arithmetic order is unchanged, so the fused-vs-reference
parity test still passes bit-for-bit.

Decode is memory-bound, so the number that predicts token rate is the bytes a model must stream
per token. A 4-bit 7B model reads ≈3.9 GiB per token, i.e. roughly one token per 2.5 s per GiB/s
of packed-weight throughput on hardware like this. Do not extrapolate these figures to a
GPU-backed run: no device was available here (see `docs/GPU_SUPPORT.md`).

## Design notes

* **Weights stay packed.** Every quantized type is consumed block-by-block; nothing is
  dequantized to a float matrix. The generic fallback expands one *row* per matmul, so a
  4096×4096 Q5_K tensor needs a 16 KiB buffer, not 64 MiB.
* **One output head pass.** Logits are produced by a single fused matmul straight on the mmap
  pointer. The previous implementation re-read the head row-by-row for every token.
* **Paged KV.** 64-token pages, allocated on demand, optionally evicted by a sliding window.
* **Expert cache.** `(layer, expert, gate|up|down)` → expert slice pointer, LRU eviction, hit
  and miss counters exposed through `vg_model_graph_expert_stats()`.
* **Threading.** `vg_threadpool_parallel_for()` splits output rows across workers; the
  reference path (`vg_cpu_matmul_ref`) is single-threaded and produced bit-identical results in
  the test suite. `vg_cpu_set_threads()` resizes (or tears down) the pool when the requested
  count changes and reports the count actually in use.
* **No per-token allocation.** Attention scores, the attention projection, the FFN gate/up/down
  buffers and the MoE top-k id list are owned by the graph and reused for every token.
* **Sampling is O(V log V).** One sort of the probability vector feeds top-k, top-p and the draw;
  the previous top-p loop rescanned the vocabulary per selected token (O(V²) per generated token).

## License

MIT for the engine files; see `LICENSE` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Measured test/build results, and an explicit list of what could not be verified here |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | Every quant type, ternary decode order, kernel strategy |
| [docs/GPU_SUPPORT.md](docs/GPU_SUPPORT.md) | Which ops the Vulkan backend executes, residency/dispatch model, device status |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layer boundaries and storage contracts |
