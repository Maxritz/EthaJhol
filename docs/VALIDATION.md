# Validation record

Every line below was produced by running the command in this repository on the Linux x86-64
build described at the bottom. Anything that could not be executed here is listed under
**Not verified** rather than claimed.

## Build matrix

| Build | Command | Result |
| --- | --- | --- |
| Release, CPU + full engine | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVG_ENABLE_VULKAN=OFF` | builds clean, 0 warnings from project sources |
| Sanitizers | `cmake -S . -B build-asan -DVG_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug -DVG_ENABLE_VULKAN=OFF` | 8/8 suites pass, no leaks; six family fixtures also run clean under LSan |
| Vulkan backend | `g++ -c src/vulkan_backend.cpp` against Vulkan-Headers 1.3.290 with the full warning set | compiles, 0 warnings |
| Core-only | `-DVG_BUILD_FULL_ENGINE=OFF` | probe/inspect/tokenize/stream targets build |

Warning flags (`-Wall -Wextra -Wconversion -Wshadow -Wpedantic`) are applied at directory
scope, so `vg_full` (the generation engine) is checked too — a per-target list used to miss it.

## Test suites

| Suite | Covers | Result |
| --- | --- | --- |
| `vg-tests` | GGUF parse, tensor ranges, KV/radix, plugins | pass |
| `vg-quant-tests` | Legacy + K-quant dequantization, **ternary and 1-bit golden vectors** | pass |
| `vg-tokenizer-tests` | SentencePiece BPE, GPT-2 byte-level, specials, byte fallback | pass (ids match an independent Python BPE) |
| `vg-tensor-tests` | Bounded leases, LRU eviction, mmap, async, concurrent prefetch, streaming | pass |
| `vg-engine-tests` | Matvec parity vs reference for 18 types, probe, dense/MoE/BitNet decode, LoRA, speculative decoding, end-to-end generation | pass |
| `vg-sampler-tests` | Temperature/top-k/top-p and the additive repetition penalty | pass |
| `vg-full-smoke` | `vg-run` on a synthetic dense model (exit status is the contract) | pass |
| `vg-full-probe` | `vg-engine probe` reports SUPPORTED for that model | pass |

## Model-family matrix

Each family below was generated as a small GGUF from that architecture's real contract
(metadata keys, tensor names, feature tensors, tokenizer kind), probed, decoded and
generated from, then deleted — nothing is committed. Geometry is held constant
(`n_embd=256`, 4 heads, 2 KV heads, head_dim 64, n_ff 256, 2 layers, 8 experts x 2 used),
so a failure isolates the family's tensor contract rather than its size.

| Family | Probe | Logits | Generates | Feature wiring |
| --- | --- | --- | --- | --- |
| llama-2 (SPM tokenizer) | SUPPORTED | finite | yes | — |
| llama-3 (GPT-2 byte BPE) | SUPPORTED | finite | yes | — |
| mistral (sliding window) | SUPPORTED | finite | yes | — |
| tinyllama | SUPPORTED | finite | yes | — |
| gemma | SUPPORTED | finite | yes | — |
| gemma2 | SUPPORTED | finite | yes | attention + final softcap, post-norms: verified |
| gemma3 | SUPPORTED | finite | yes | QK norm, post-norms: verified |
| qwen2 | SUPPORTED | finite | yes | attention bias: verified (delta 4.0e-1) |
| qwen3 | SUPPORTED | finite | yes | per-head QK norm: verified (6.0e-2) |
| smollm | SUPPORTED | finite | yes | attention bias: verified (4.0e-1) |
| bitnet b1.58 (TQ2_0) | SUPPORTED | finite | yes | attn/ffn sub-norms: verified (1.1e-2) |
| mixtral (8 experts, top-2) | SUPPORTED | finite | yes | router + experts: verified (7.9e-2) |
| qwen3-moe | SUPPORTED | finite | yes | router: verified (2.1e-2) |
| qwen2-moe (shared expert) | SUPPORTED | finite | yes | router and shared expert: verified (4.7e-2) |
| deepseek-moe (shared expert) | SUPPORTED | finite | yes | router and shared expert: verified (4.7e-2) |
| phi3 | rejected | — | — | names the unsupported piece |
| phi3, fused QKV + fused gate/up | rejected | — | — | `fused QKV projection (attn_qkv.weight) is not implemented` |
| gpt2 | rejected | — | — | names the unsupported architecture |

**Feature wiring** is a behavioural check, not a code reading: a second fixture perturbs
exactly one feature tensor (attention bias, QK-norm weights, sub-norm weights, the MoE
router, the shared-expert FFN) with *non-uniform* values and the logits must move by more
than 1e-3 relative. Uniform scaling is deliberately avoided because the following RMSNorm
cancels it, which would make an unimplemented feature look wired.

Weight-type sweep on the same geometry (probe, finite logits and generation all pass):
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q4_K, Q6_K, TQ2_0, Q1_0, Q2_0, F16.

### Defects this matrix found and fixed

* **MoE experts were multiplied by a zeroed vector.** `moe_ffn_layer` normalised the block
  input into `hidden`, then `memset(hidden, 0)` before the expert matvecs, so every expert
  computed `W * 0` and the FFN block degenerated to the residual. The block input now lives
  in its own `work_moe_in` buffer. Proof: routing logits changed with the router weights
  while top-5 logits stayed bit-identical, and a zeroed-expert model produced exactly the
  same output as the real one.
* **MoE errors were swallowed.** `ffn_layer` discarded the status of `moe_ffn_layer`
  (`(void)`), continuing on a half-written activation. `ffn_layer` now returns `VG_Status`
  and `vg_model_graph_decode` propagates it.
* **Attention bias was added after RoPE.** `attn_q.bias`/`attn_k.bias` were added to the
  rotated vectors, which rotates the bias as well. Both are now added inside the projection,
  before QK-norm and RoPE (qwen2/smollm were affected).
* **gemma2/3 were reported SUPPORTED but not executed.** The sandwich post-norms
  (`attn_post_norm`, `ffn_post_norm`) and the attention/final logit softcaps were ignored.
  All three are implemented, and the probe now reports `post_norm`/`logit_softcap` in its
  operator list.
* **Shared experts were ignored.** `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` were
  never read, so qwen2-moe and deepseek-moe ran with only the routed experts. The
  sigmoid-gated shared FFN is now part of the MoE block.
* **Sliding window used the wrong metadata key.** Only `gemma4|llama|attention.sliding_window`
  was read; the arch-prefixed key (`mistral.attention.sliding_window`) was missed.
* **The probe accepted layouts the evaluator cannot execute.** Fused QKV (`attn_qkv`), fused
  gate/up (`ffn_up` with `2*n_ff` outputs), MLA projections, learned positional embeddings
  and convolutional attention are now refused by name instead of silently computing a
  different model. The architecture message also lists the real supported set.
* **Regression test added**: `vg-engine` writes a zero-expert MoE fixture and requires the
  logits to differ from the real one. Before the fix they were bit-identical.

### What the engine tests actually assert

* **Matvec parity**: every fused kernel is compared against `vg_cpu_matmul_ref` (single
  threaded) *and* against dequantize-row-then-dot, so a threading bug and a kernel bug are
  distinguishable. Covers F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K,
  Q5_K, Q6_K, Q8_K, TQ1_0, TQ2_0, Q1_0, Q2_0.
* **Ternary golden vectors**: block bytes are decoded by the library and compared with values
  produced by an independent Python port of ggml's `dequantize_row_tq1_0/tq2_0/q1_0/q2_0`,
  including the position-weighted sum so a wrong *element order* fails even when the multiset
  is right.
* **BitNet end to end**: a synthetic `bitnet` model (TQ2_0 projections, `attn_sub_norm`,
  `ffn_sub_norm`, a per-tensor `attn_q.scale`) is probed, decoded and generated from, and the
  probe report names `sub_norm` in its required ops.
* **LoRA**: the adapter delta is compared against materializing the delta into a dense weight
  and multiplying (two independent code paths), applied during generation (logits and tokens
  change), and detaching restores the base logits bit-for-bit.
* **Speculative decoding**: with draft == target every proposal is accepted and the emitted
  sequence is identical to plain greedy decoding.
* **Rejection path**: a model whose weights use an unimplemented quant type is refused before
  any weight is read, and the message names the exact tensor (`blk.0.attn_q.weight`) and type.

## Real-model run

```
./build/vg-run -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
               -p "The capital of France is" -n 16 -c 256 --temp 0.0 --seed 42 --threads 2
... a city in the region of France. It is located on the River Seine-
[vg] TG: 16 tokens in 8.303s = 1.8 t/s
```

`vg-engine probe` on that file reports `arch llama`, tensors in `Q4_0,Q6_K,F32`, and
`SUPPORTED`; so the real-data coverage of the K-quant path here is Q6_K, not Q4_K.

## Performance (2 vCPU, 4096x4096 matvec, best of 8)

| Type | Bytes streamed | ms/op | GiB/s |
| --- | --- | --- | --- |
| Q4_0 | 9.00 MiB | 5.54 | 1.59 |
| Q8_0 | 17.00 MiB | 6.49 | 2.56 |
| Q4_K | 9.00 MiB | 7.07 | 1.24 |
| Q6_K | 13.12 MiB | 10.66 | 1.20 |
| **TQ2_0** | **4.12 MiB** | **3.40** | 1.19 |
| TQ1_0 | 3.38 MiB | 7.77 | 0.42 |
| Q2_0 | 4.50 MiB | 5.39 | 0.82 |
| Q1_0 | 2.25 MiB | 5.80 | 0.38 |
| F32 | 64.00 MiB | 8.74 | 7.15 |
| F16 | 32.00 MiB | 17.38 | 1.80 |

Single-thread, same shape: Q4_0 10.20 ms, Q6_K 19.61 ms — so threading is a 1.84x on Q4_0 and a
1.84x on Q6_K.

Caveats that matter when reading this table:

* The box is a shared 2-vCPU VM. Identical runs of Q4_0 varied between 5.8 and 8.2 ms/op, so
  `vg-mmbench` reports the **minimum** of N iterations; the mean is printed next to it.
* Threading is a genuine 1.84x on this workload (Q4_0 5.54 ms at 2 threads vs 10.20 ms at 1).
* TQ2_0 is the format BitNet and PrismML ship, and it is the fastest per token here because it
  streams 45% of the bytes Q4_0 does. TQ1_0 is the legacy 1.6875 bpw layout: its radix-3
  unpack costs more than the bandwidth it saves on this CPU.

## Not verified

| Item | Why |
| --- | --- |
| GPU execution of any kernel | No Vulkan ICD exists in this environment and the SDK cannot be installed, so the Vulkan build here is a compile-only check against Khronos Vulkan-Headers; the CPU build uses `vulkan_stub.c` and `--ngl` reports "no Vulkan backend, running on CPU". |
| Shader compilation (SPIR-V) | `glslangValidator` is not installed in this environment, so the 15 shaders were not recompiled here. |
| Windows / MSVC build | No toolchain available here. |
| Q2_K / Q3_K / Q4_K / Q5_K on real files | Only synthetic fixtures and golden vectors exercise them; no real GGUF of those types was available for download. |
| >256 MiB tensor path (transient lease) | The real model's largest tensor is 53 MiB, so the host-budget overflow path in `tensor_source.c` is never entered. |
| Real BitNet 2B4T weights | A synthetic `bitnet` fixture is used; the architecture, sub-norms, per-tensor scale and ternary decode are all exercised, but not Microsoft's released file. |

## Environment

Linux x86-64, GCC 12, CMake (from the sandbox toolchain), Ninja, 2 vCPU, no GPU, no Vulkan SDK.
