# vg-gguf-engine

`vg-gguf-engine` is a portable GGUF inference engine built as a compact C99 systems layer plus a complete C++17 model-generation layer. It is self-contained: no external LLM runtime is required to load a GGUF file and generate tokens. The engine is CPU-first and CPU-explicit by design: large-model operation is made visible through mmap/lazy loading, bounded tensor residency, paged KV state with radix prefix reuse, streaming hooks, and a traced dispatch path. ROCm/HIP is the planned GPU path and exists today as an opt-in verification build; the quantized decode path is CPU-only.

The native layer provides GGUF metadata inspection, typed tensor-geometry validation, disk-backed tensor leases, a bounded host LRU cache, async prefetch and streaming hooks, paged KV allocation, immutable sequence forks, radix prefix snapshots, tracing, and runtime plugin interfaces. The full layer exposes a stable C streaming API and a CLI that performs real model tokenization, prompt evaluation, decoding, sampling, and token-piece streaming through the engine's own transformer graph evaluator.

This README describes the current engine as it actually is on this branch and machine: a CPU-first GGUF engine with a planned AMD ROCm/HIP GPU path, not a Vulkan or graphics-shader backend engine.

## What the engine is

- A real GGUF loader and transformer evaluator in C99, with the generation loop and sampling in C++17.
- A native GGUF/metadata layer (`include/vg/gguf.h`, `src/gguf.c`), a tensor-source/lazy-loader layer (`include/vg/tensor_source.h`, `src/tensor_source.c`), a KV cache layer (`include/vg/kv_cache.h`, `src/kv_cache.c`), threading (`src/threadpool.c`), quantization (`include/vg/quant.h`, `src/quant.c`), attention/FFN execution math (`src/model_graph.c`, `src/model_graph_internal.h`, `src/layer_math.c`, `src/arch_spec.c`), and CPU matmul/activation kernels (`include/vg/cpu_ops.h`, `src/cpu_ops.c`).
- A full engine layer for tokenization, generation, sampling, and streaming (`include/vg/full_engine.h`, `src/native_engine.c`, `src/run_main.cpp`, `src/tokenizer.c`, `src/tensor_source.c`).
- A top-level CLI (`src/main.c`) that exposes `vg-engine inspect|tokenize|logits|diagnostics|stream ...`, plus the full runner `vg-run` when the full engine is built.
- A plugin ABI for model/quantizer/storage/backend/sampler extensions (`include/vg/plugin.h`, `include/vg/plugin_loader.h`, `src/plugin_loader.c`, `tools/example_plugin.c`).

The build is CMake-driven, C99 for the systems layer and C++17 for the generation layer, with MSVC and clang-cl supported on Windows and a POSIX path included. On this machine the primary validated platform is Windows 11.

## What the engine is not

This is not a Vulkan backend engine. There is no Vulkan dispatch in the current tree, and the README no longer claims `-ngl 99`, a software Vulkan device, or bundled third-party runtime code as current capabilities. The older Vulkan shader assets and stubs that remain in the repo are legacy scratch from an earlier pass; they are not part of the documented engine surface.

ROCm/HIP is the planned GPU vehicle because it is the path in the plan that can land compute on the AMD compute engine rather than a graphics/3D ring. The current HIP code is a correctness-and-probe gate, not a production acceleration path for quantized models yet.

## Current limitations of this README

This README is written from the tracked source on this branch, not from a fresh build or model run performed while writing it. The build knobs, targets, and CLI verbs are checked against the tracked CMake and `src/main.c` files, and the capability status reflects the current CPU-first / HIP-verification-only shape of the project. It is not a run-time report from a just-built binary or a just-run model.

## Capabilities

| Capability | Status | Notes |
| --- | --- | --- |
| Real GGUF prompt completion | Available | `vg-run` tokenizes, prefills, decodes, samples, and streams pieces. |
| CPU inference | Available | AVX2 + thread pool; per-type quantized GEMV/GEMM kernels; row-parallel dispatch. |
| GPU inference (ROCm/HIP) | Opt-in verification only | F16/F32 GEMV+batch exist and match CPU on the probed shapes; all K-quant/IQ types remain CPU-only. |
| Dense model loading | Available | mmap + lazy loading by default; `--no-mmap` disables mapping for controlled comparisons. |
| MoE model loading | Available | Lazy hot-expert paging under a host budget; unrouted experts stay on disk. |
| Long context | Available | Configurable context, paged KV/radix contracts, runtime KV management. |
| Sampling | Available | Temperature, top-k, top-p, repeat penalty, deterministic seed; streaming callbacks. |
| Discrete model families | Supported | Unsupported files fail with diagnostics rather than silently running wrong math. |
| Extension model | Available | Versioned C ABI for model, quantizer, storage, backend, and sampler plugins. |
| Operating systems | Windows 11 (primary) | C99/C++17, CMake, mmap, Win32 mapping and dynamic-loading branches included. Linux path included; shipped validation is Windows. |
| Quant types verified end-to-end | Q4_0, Q4_K_M, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, Q5_0, Q5_1, IQ4_NL, IQ4_XS | Bitwise dequant round-trips against the reference `gguf` package plus end-to-end oracle/logit parity on the real quantized files present in the tree. |

### Native core tooling

- `vg-engine inspect model.gguf` — GGUF architecture, tensor geometry, and storage information.
- `vg-engine stream model.gguf TENSOR` — bounded lazy tensor lease exercise.
- `vg-engine diagnostics model.gguf` — file sizing, tensor-type distribution, tokenizer info, KV sizing estimate, and a storage-source acquire test.
- `vg-engine tokenize model.gguf "text"` — encode/decode through the engine's tokenizer.
- `vg-engine logits model.gguf [prompt]` — run a prefill and print the top-5 greedy logits and pieces.
- `vg-engine plugin-info plugin.dll` — load and print a runtime plugin descriptor.
- `vg-tests` — paged KV allocation, sequence fork behavior, and radix lookup.
- `vg-quant-tests` — scalar conversion and quant dequantizer round-trips.
- `vg-tensor-tests` — tensor-source model tests when a test model is present.
- `vg-tokenizer-tests` — tokenizer unit tests.
- `vg-sampler-tests` — sampling and repetition-penalty unit tests.
- `vg-bench`, `vg-gemv-bench`, `vg-gemm-bench` — page/radix and matmul microbenchmarks.
- `vg-fused-kernel-test` — fused quantized GEMV/batch exactness gates.
- `vg-quant-mem-probe` — peak-working-set probe proving a matmul did not materialize F32 weights.
- `vg-hip-probe` and `vg-hip-matmul-test` — present only when `VG_ENABLE_HIP=ON`.
- `vg-run` — the full runner: tokenization, prefill, decode, sampling, streaming.
- `vg-conventions` — ctest oracle gate: the engine's own top-8 must be reproducible by an independent forward pass, and the matching convention must be the one the architecture specifies.

## Build the complete engine

The default build includes the native inference engine and the optional HIP probe when requested:

```sh
cmake -S . -B build -DVG_BUILD_FULL_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

For a minimal embeddable systems-only build without the full graph runtime:

```sh
cmake -S . -B build-core -DVG_BUILD_FULL_ENGINE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --config Release
ctest --test-dir build-core --output-on-failure
```

To build with the HIP probe and HIP matmul objects enabled (opt-in; fails loudly if `hipcc` is missing):

```sh
cmake -S . -B build-hip -DVG_BUILD_FULL_ENGINE=ON -DVG_ENABLE_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip --config Release
```

The HIP build defines `VG_HAVE_HIP` on the core library and produces `vg-hip-probe` and the HIP matmul test. It does not change the CPU kernels and does not make quantized types run on the device.

## Run a real model

The full runner uses mmap plus lazy model loading by default. The following streams generated pieces to stdout:

```sh
./build/vg-run -m model.gguf -p "Write a short story about a village" -n 128 -c 8192 --temp 0.8 --top-k 40 --top-p 0.95
```

Useful knobs:

- `--threads N` sets CPU generation threads. The accepted range is `auto` or `1..physical-cores`. The default is a fraction of physical cores; requesting more than physical cores is rejected.
- `--seed N` for deterministic output.
- `-c N` to select the context window.
- `--f32pack` to dequantize weights to F32 once and run AVX2 F32 GEMVs. Deterministic and useful for controlled math comparisons; off by default so lazy MoE paging and low-RAM operation stay intact.
- `--no-mmap` to disable mmap and force the malloc+read pager against the host-budget LRU.
- `VG_DEBUG=1`, `VG_PROFILE=1`, `VG_RESOURCES=1` for diagnostics; see the profiling section.

The C API in `include/vg/full_engine.h` lets a host application receive each token piece through a streaming callback.

On Windows 11, build through the same CMake project with Visual Studio or clang-cl and run `vg-run.exe` directly.

## How a real decode run actually looks

The engine's execution map is small and traceable:

- **Load.** Dense models are fully resident in host RAM before the first token, so generation never pages weights from disk. MoE models keep a lazy hot-expert budget (default 256 MiB host budget, overridable with `VG_HOST_BUDGET_MB`); only routed experts are paged in, and unrouted experts stay on disk.
- **Prefill.** The prompt is processed as a batched GEMM over the prompt tokens so each weight byte streams from RAM once per batch instead of once per token. Activation phases (norms, RoPE, attention rows, SwiGLU, residuals) keep the exact serial scalar order.
- **Decode.** Each generated token runs a per-layer serial path: one matmul per projection per layer, plus the output projection (lm_head), sampling over the full vocab, and detokenization. There is one call per projection per layer per token. For a large all-Q4_K model that is many matmul calls per token, and each call streams a large packed weight block from DDR.
- **Profiling.** `vg-prof` records per-op host time, bytes, dispatch shape, and device time where a GPU kernel actually ran. Device time is reported separately from host wall time and is N/A for pure-CPU ops.

The practical consequence for large dense quantized models on this hardware: decode is DDR-bandwidth bound. The FFN matmul alone is the majority of per-token weight traffic, and saturating DDR with the row-parallel kernel is the ceiling. That is why more threads stop helping once DDR is saturated, and why the current per-call HIP path is slower than CPU for the shapes in use.

## How correctness is decided

Correctness in this tree is decided by independent evidence, not by determinism alone. Two separate gates are used:

1. **Bitwise dequant round-trips against the reference `gguf` package.** The engine's dequantizers are compared against an independent Python `gguf`-package dequantizer on the real quantized files. That catches the Q5_0/Q5_1 high-nibble index bug and confirms the IQ4_NL and fused Q4_K/Q5_K/IQ4_XS layouts by construction.

2. **End-to-end oracle/logit parity.** The engine's prefill and generated tokens are compared against an independent forward pass written in Python (`tools/oracle_forward.py`, `tools/oracle_moe.py`, `tools/conv_matrix.py`, `tools/oracle_gate.py`). The oracle shares no engine code, so when the two disagree the prior is "which one is wrong," not "the engine must be wrong." This gate is what caught the qwen2, olmoe, gemma3, gemma4, and starcoder2 convention bugs.

Determinism and bit-identity are necessary but not sufficient. A bit-stable engine can still implement the wrong convention. The convention oracle gate is therefore registered in CMake as `vg-conventions` and is skipped only when python/gguf/model are absent, never treated as passed.

## Performance realities documented in this tree

- **Throughput is measurable for both prefill and decode.** Prefill is batched and benefits from more threads up to the DDR saturation point; decode is per-token and becomes DDR-bandwidth bound. Track time-to-first-token and decode tokens/sec with `VG_Time` instrumentation.
- **Dense = RAM resident; MoE = hot experts only.** Dense models fully load into RAM at load time; MoE models page only the routed experts under an explicit host budget.
- **Quant types stay in the same precision in memory as on disk.** No implicit upcasting/downcasting during graph load or tensor lease.
- **Debug builds are tested, not assumed.** Debug segfault fixes (for example forced inlining of AVX2 helpers) are preserved so the Debug path stays runnable.

### Why more CPU threads do not always help decode

The machine used for profiling has 16 physical / 32 logical cores. The thread-control policy deliberately expresses thread counts in physical cores and caps `--threads` at `1..physical-cores`; requesting more than physical cores is rejected as invalid. That policy is intentional: the row-parallel GEMV already saturates DDR on the large matrix shapes the engine uses, so logical/SMT siblings contend for the same memory channels and do not raise the bandwidth ceiling.

For the large all-Q4_K_M model present in this tree, the measured reality at max CLI threads versus default threads was:

- prefill: faster with more threads, with diminishing returns once DDR is saturated;
- decode: flat within noise;
- top-8 logits: identical across runs, so correctness is preserved.

The bottleneck is not core count. The bottleneck is the rate at which packed weight bytes can be read from DDR for each per-token matmul. For that model the FFN projections dominate per-token weight traffic, and the row-parallel Q4_K kernel already reaches roughly the DDR bandwidth ceiling on the large shapes. Adding threads past that point does not raise throughput.

### What would move the decode number

Only moving the weight reads off DDR helps. On this machine's GPU, GDDR6 bandwidth is far higher than DDR bandwidth, so a device-resident quantized matmul path would remove the per-call host read and change the ceiling from DDR to VRAM. That is the reason the GPU path exists in the plan. But the current HIP path is not yet that path: it re-copies weights every call and synchronizes, so it is currently slower than CPU for the shapes in use. Closing that gap is a kernel+runtime change, not a CLI or thread-count change.

## Profiling and investigation

The engine has a built-in per-op profiler and resource reporter, and the repo includes harness and reference tools used during investigation:

- `VG_PROFILE=1 VG_RESOURCES=1 VG_PROF_OUT=<prefix>` writes per-op text/CSV/JSON plus a per-invocation samples file. Host and device time are separate; device time is N/A when no GPU kernel ran.
- `vg-engine inspect model.gguf` prints GGUF architecture, tensor geometry, and storage information.
- `vg-engine stream model.gguf TENSOR` exercises a bounded lazy tensor lease.
- The `bench/` and `tools/` trees include matmul and kernel benches, quant round-trip checks, and independent forward-pass oracles used to validate correctness during changes.

Investigation in this tree follows measurement, not guesswork: decompose the execution and data flow, add traps where visibility is insufficient, rank bottlenecks by cumulative cost, form disprovable hypotheses, change one meaningful variable at a time, and retest against a measured baseline with correctness re-verified.

## GPU path status

ROCm/HIP is the plan for AMD acceleration on RDNA4 gfx1201 first and RDNA2 gfx1031 second. Today:

- `VG_ENABLE_HIP=ON` builds the HIP probe and the HIP matmul objects, links the HIP backend into the core, and defines `VG_HAVE_HIP`.
- F16/F32 GEMV and batched GEMM route through HIP only when the corresponding `VG_HIP_F16=1` / `VG_HIP_F32=1` gates are set; any HIP error falls back to CPU instead of faking success.
- The probe passes on this machine (gfx1201), and the F16/F32 path matches CPU logits/tokens for the shapes tested.
- The measured reality on this hardware is that the HIP path is currently slower than CPU for the matmul shapes the engine uses, because each call re-copies the weight from the host and synchronizes. Therefore the GPU path is verification-only until device-resident weights or stream capture remove that per-call transfer+launch+sync cost and until the full acceptance bar is met.

Quantized types are CPU-only until a per-type parity gate is met. The next GPU kernel candidate is a per-type K-quant GEMV behind the same parity gate, not a narrow special case.

### GPU acceptance bar

For any kernel to count as done, all of the following must hold:

1. Generated tokens/logits on a real model run match the CPU engine — same emitted token sequence and top-1/rank-identical top-8 logits to 4 decimal places. Bit-identical logits are not required where the HIP accumulation order differs from the CPU order; ~1e-6 elementwise maxabsdiff is the expected magnitude for F16/F32.
2. Batched GEMM matches n separate GEMV calls for the same accumulation order the CPU oracle requires.
3. The HIP runtime probe passes on this machine's GPU.
4. The dispatch demonstrates it lands on the compute engine rather than the graphics engine.
5. There is a measured per-call speedup over the CPU kernels on the same shapes the engine uses.

Until every item holds, the CPU engine stays the production path. Today: items 1–3 pass for the existing F16/F32 path; item 4 is not yet demonstrated; item 5 currently fails because each call re-copies W/X from the host.

## Architecture

The storage path is tiered: metadata stays resident, tensor payloads are addressed by file range, host cache entries are bounded and leased, and device backends may stage only the tiles required by the active graph. The long-context path treats KV pages as immutable after publication, so forked sequences and radix prefixes can share pages safely. Prefix identity includes model and adapter state so cached pages cannot cross incompatible execution configurations.

The performance policy is practical and measured: quantized weights stay packed until the fused backend consumes them; hot loops are laid out for SIMD and branch-light execution; read-ahead is explicit; intermediate buffers are minimized; and trace hooks expose storage, page, cache, and dispatch costs before tuning. AMD RDNA-specific compute variants belong in backend plugins or the HIP path so the portable core does not acquire device-specific assumptions. GGUF metadata and tensor conventions are honored directly rather than bypassed with per-model hacks.

## Limitations and open work

- GPU acceleration for quantized models is not production yet. Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, Q2_K, Q3_K, Q5_0, Q5_1, IQ4_NL, IQ4_XS are CPU-only.
- The current HIP speedup bar fails for the existing F16/F32 path because of per-call host-device copies and sync; that is a known, measurable gap, not a hidden one.
- For very large dense models on this machine, decode is DDR-bandwidth bound and thread scaling flattens once DDR is saturated.
- Some model families or arch-specific conventions are still being brought up to parity; unsupported or unknown archs fail with diagnostics rather than running silently wrong math.

## Active investigation artifacts in the repo

This tree currently carries the live measurement and investigation artifacts alongside tracked source and docs:

- Tracked project docs and status: `README.md`, `AGENTS.md`, `STATUS.md`, `TASKS.md`, `ARCHITECTURE_GAP.md`, `AUDIT.md`, `THIRD_PARTY_NOTICES.md`, and the `docs/` set (`FEATURE_COMPLETE.md`, `FEATURE_GAP.md`, `VALIDATION.md`, `Validation Record`, `Feature-Complete Engine Profile`, `vg-gguf-engine.md`, `Third-Party Notices`).
- Tracked source and headers: `src/` and `include/vg/`, plus the investigation-visible parts `src/layer_math.c`, `src/cpu_ops.c`, `include/vg/prof.h`, `include/vg/hip_ops.h`, `src/model_graph.c`, `src/model_graph_internal.h`, `src/arch_spec.c`, `src/vg_prof.c`, and `tools/hip_matmul.hip`.
- Investigation harnesses and oracles in `tools/`: `hip_probe.hip`, `hip_matmul.hip`, `hip_matmul_test.c`, `hipcc_probe.sh`, `gemm_bench.c`, `gemv_bench.c`, `quant_mem_probe.c`, `fused_quant_kernel_test.c`, `gt_alltypes.py`, `oracle_forward.py`, `oracle_moe.py`, `oracle_gate.py`, `conv_matrix.py`, `audit_models.py`, and the Q35/qwen2 diagnostic oracles.
- Benchmarks and captures: `bench/` and the raw run captures such as `benchmark_tables_raw.txt`, `gemma4_bench.txt`, `gemma4_profile2.txt`, `smolcode_bench.txt`, `smolcode_fox121_prompt.txt`, `stories_f16_bench.txt`, and the HIP profile writes `vgprof_hip_f16.{txt.txt,txt.csv,txt.json,txt.samples.csv}`.
- The TileLang ROCm investigation tree: `tilelang-rocm-copy/` (the TileLang ROCm copy used for codegen), `tilelang_ab/` (the A/B harness, manifest, generated HSACOs, and run logs), and the gfx1201 probe/build scripts and objects.
- Port analysis: `ports/`.
- Untracked build/scratch trees: `build-dbg/`, `build-hip/`, `build-rel/`, `obj/`, `exe/`, `tools/__pycache__/`, `.agents/`, `.freebuff/`, `.scratch/`, and assorted `*.obj`/`*.exe` intermediates. These are investigation leftovers, not part of the documented engine surface.

Stale scratch, failed caps, and old Vulkan artifacts from earlier passes remain in the tree but are not part of the documented engine surface; the README describes the current CPU-first reality, not those older paths.

## References

- [1] https://github.com/FlashML-org/FreeToken — FreeToken repository
- [2] https://github.com/JustVugg/colibri — colibri repository
- [3] https://github.com/patcarter883/minisglang-rdna4 — mini-sglang RDNA4 repository

## License

The engine files are MIT licensed. Consult `LICENSE` for details.
