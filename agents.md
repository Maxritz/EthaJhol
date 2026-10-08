# Agents — vg-gguf-engine

## Architecture (current: CPU-only)

All inference is native CPU. The Vulkan compute backend and its shader
pipeline were removed (see session findings 2026-09-29). A ROCm/HIP
backend is the planned replacement for GPU acceleration but is not yet
present. There is no GPU dispatch, no `--ngl` flag, and no `VG_HAS_VULKAN`
guard anywhere in the tree.

## Mandatory Rules

1. **Inference must work with GGUF** — every model path loaded via `VG_GGUF` must complete end-to-end token generation. No model is "supported" until a full forward pass runs and produces output.

2. **High speed for PP and TG** — prefill (prompt processing) and token-generation throughput must both be measurable. Track time-to-first-token (TTFT) and tokens/sec for decode (TG). Use `VG_Time` instrumentation. Bottleneck = any kernel spending >30% frame on a single op.

3. **Lazy / disk-based loading for MOE experts** — only the active expert weights are paged into host memory. Unused experts stay on disk. Verify via `vg-engine inspect` showing `lazy source: host_budget=`.

4. **Every component evaluated** — produce a project plan gap-analysis table (see below) mapping each subsystem to: Done / Stub / Missing / Fake / Mock. Update on every commit.

5. **All missing/incomplete/partial/demo/placeholder/fake/mock code noted** — any file containing `TODO`, `stub`, `placeholder`, `fake`, `mock`, `not implemented`, or `NYI` gets logged in the TODO ledger below with severity P0/P1/P2.

6. **Done = inferencing runs correctly for all quants** — Q4_0, Q4_K_M, Q5_K_S, Q8_0, Q2_K, IQ4_XS, and any new quant type discovered during testing. Must pass before closing any task.

7. **Models remain same quant in memory as on disk** — no implicit upcasting or downcasting during graph load or tensor lease. Verify via `vg-engine inspect` showing tensor type codes matching the file.

8. **Ethajhol tweaks optimized and redone** — any hack, workaround, or special-case that bypasses standard GGUF metadata must be re-examined. Either remove it or replace it with a standards-compliant path. Document the original shortcut and the replacement.

---

## Component Evaluation Table

| Component | Status | Notes |
| --- | --- | --- |
| GGUF header parser | Done | `src/gguf.c` — reads metadata, tensor descriptors |
| Tensor source / lazy loader | Done | `src/tensor_source.c` — mmap-backed lazy pager with host-budget LRU + spill |
| Model graph loader | Done | `src/model_graph.c` — parses arch metadata generically |
| Native tokenizer | Done | `src/tokenizer.c` — added-token matching for chat templates |
| CPU inference kernels | Done | `src/cpu_ops.c` |
| Sampling (greedy, top-k, top-p) | Done | `native_engine.c` `sample_next_token`; shared `vg_apply_repetition_penalty` |
| Native full engine | Done | `src/native_engine.c` — `vg_generate()` |
| CLI runner | Done | `src/run_main.cpp` (vg-run), `src/main.c` (vg-engine) |
| MoE expert sharding | Partial | Lazy load via `acquire_weight`→`vg_tensor_acquire`; unload via `release_weight`→`vg_tensor_release`→`tensor_source.c` LRU. No MoE-specific spill test exists. |
| Quant type support matrix | Partial | Q4_0, Q4_K_M, Q5_K_S, Q6_K, Q8_0 verified in `test_quant.c`. Q2_K, IQ4_XS untested. |
| Radix KV cache | Done | `src/kv_cache.c` — radix tree with page refcounts, spill, LRU eviction. |
| Thread pool scheduler | Partial | `src/threadpool.c` — pool is real and cross-platform; NOT yet wired into prefill/decode to parallelize matmul (single-threaded inference path still in `native_engine.c`). |
| GPU backend (ROCm/HIP) | Missing | Vulkan backend removed 2026-09-29. No GPU acceleration path currently exists. |

## Session findings (2026-09-27)

Verified against an independent numpy reference:
- Engine forward pass matches oracle (logit 15.31 vs 15.24 at the capital-of-France completion; rank-identical; top-1 = "Paris").
- Engine gen-token matches oracle argmax on the 18-token Llama-3.2-1B-Instruct chat prompt.
- **Fixed: double-BOS.** `vg_generate` prepended BOS even when the template already started with `<|begin_of_text|>`, shifting every KV position by one. Guard added: skip BOS when `prompt_tokens[0] == bos`.
- **Fixed: repetition-penalty history.** Now seeded with the prompt's last `repeat_last_n` tokens so the penalty is active from token 1; consolidated onto the shared additive `vg_apply_repetition_penalty` (`full_engine.h`), unit-tested in `tests/test_sampler.c`.
- Added auto chat-template framing for Instruct models in `vg-run` (`-p` is wrapped via the Llama-3 template; `--system` overrides the default system turn).
- Note: Llama-3.2-1B-Instruct itself produces weak continuations for trivial factual Q&A even when correctly framed (model quality, not an engine bug); larger instruct models answer correctly.

## Session findings (2026-09-29) — Vulkan removal, CPU-first

The Vulkan compute backend was fully removed. Rationale: the dense
single-CB-per-token path produced correct tokens only for a narrow
subset (Q8_0, no q/k-norm, no bias), and the Q4_K/Q6_K raw-block dense
path could not be made correct fast enough; bias handling (Qwen2.5 has
attn q/k/v biases) plus the MoE path were not on the single-CB scheme.
Decision: strip to a clean CPU engine first, make CPU fast and coherent
across all quants and families, then add a ROCm/HIP backend that
implements the full dense + MoE + bias paths rather than patching Vulkan.

Changes:
- Deleted `src/vulkan_backend.cpp`, `src/vulkan_stub.c`,
  `include/vg/vulkan_backend.h`, and `shaders/*.comp` (16 files).
- Stripped all `#ifdef VG_HAS_VULKAN` blocks (36) from `model_graph.c`,
  `native_engine.c`, `main.c`, `model_graph.h`.
- Removed dead GPU flags (`qkv_gpu`, `done_gpu`, `gpu_done`,
  `gpu_logits_done_local`), the `--ngl` CLI flag, `use_gpu`/`ngl_layers`
  config fields, and `supported_vulkan` quant field.
- `CMakeLists.txt` no longer references `VG_ENABLE_VULKAN`, glslang,
  SPIRV, or the `vg_vulkan` target.
- CPU path now always taken: `vg_model_graph_decode` runs the
  per-layer CPU loop directly.

## TODO Ledger

### P1 — Inaccuracy / partial implementation
- **Thread pool parallelism** — pool exists but not wired into prefill/decode matmul (inference is single-threaded). (Open — performance.)
- **End-to-end prompt-vs-oracle regression** — no ctest that runs `vg-run` and compares emitted tokens against an independent forward pass. (Open — add `test_prompt.c` on `stories15M-q4_0`.)
- **MoE lazy expert paging test** — no MoE-specific spill test. (Open.)

### P2 — Performance / missing features
- **GPU backend (ROCm/HIP)** — no GPU acceleration after Vulkan removal. Target: RDNA2 (gfx1030) and RDNA4 (gfx1201) kernels via HSA/ROCm, replacing the removed single-CB dense path for Q4_K/Q6_K raw blocks + MoE + biases.
- **CPU matmul optimization** — current `vg_cpu_matmul` is single-threaded and F32; the threadpool is unwired. This is the primary CPU speed lever.

## Quant Verification Matrix (CPU)

| Quant | Type Code | Tested Model | TG Tokens/sec | Status |
| --- | --- | --- | --- | --- |
| Q8_0 | type=8 | laya-Q8_0.gguf | ~12 t/s | ✅ Working |
| Q4_0 | type=2 | stories15M-q4_0.gguf | ~15 t/s | ✅ Working |
| Q4_K_M | type=12 | smolcode-cpp-1.5b-q4_k_m.gguf | ~0.6 t/s | ✅ Working (CPU) |
| Q6_K | type=14 | Qwen3-4B-Instruct-2507-Q6_K.gguf | — | ✅ Working (round-trip in test_quant.c) |
| IQ4_XS | type=23 | gemma-4-12B-IQ4_XS.gguf | — | ⏳ Untested |

## Repro Tests

```sh
# Build (CPU-only; no Vulkan flags)
cmake -S . -B build -DVG_BUILD_FULL_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Test
ctest --test-dir build --output-on-failure

# Smoke run (replace MODEL)
./build/vg-run -m MODEL.gguf -p "Hello" -n 8 --temp 0 --threads 4 --no-mmap
```

## Model locations (this machine)

- `G:\More-models\` — mixed dense/MoE GGUFs (smolcode-coder-cpp-1.5b-q4_k_m, qwen2.5-coder, Qwen3.8 variants, Tiel-Coder, etc.)
- `H:\OLLAMA-Models\GGUF\` — Ollama cache (Qwen3.5/3.6/3.8, K2-Horizon, olmoe-1b, gpt-oss-120b, etc.)
- `C:\Users\rr\` — acrux-500m.gguf (Q8_0+Q6_K), tinyllm.gguf

Last updated: 2026-09-29
