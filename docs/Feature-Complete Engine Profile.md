# Feature-Complete Engine Profile

The project now ships two layers. The native `vg_core` layer remains a small C99 library for GGUF indexing, bounded tensor leases, paged KV state, radix prefix reuse, tracing, plugins, and the Vulkan self-test. The `vg_full` layer is a C ABI streaming generation facade backed by the bundled llama.cpp graph runtime. This keeps the public extension surface small while providing a real execution path rather than a mock transformer.

| Area | Delivered behavior |
| --- | --- |
| Real GGUF execution | The bundled bridge loads real GGUF models, tokenizes prompts from model vocabulary metadata, evaluates prompt and decode batches, samples tokens, converts pieces, and invokes a streaming C callback. |
| CPU path | Full llama.cpp CPU graph and quantized kernels are available through `-ngl 0`. |
| Vulkan path | The bundled GGML Vulkan backend is enabled by default in full builds; `-ngl 99` offloads model layers and KV state when a Vulkan device is available. |
| Oversized models | The full runner defaults to mmap plus lazy model loading. The native C99 tensor source provides bounded host LRU leases, range reads, prefetch hooks, and storage tiers. |
| Long context | Context size is configurable with `-c`; the native layer provides paged KV and radix prefix contracts, while the full backend uses the mature model-runtime KV implementation. |
| Sampling | Temperature, top-k, top-p, repeat penalty, deterministic seed support in the C ABI, and streamed token callbacks are wired. |
| Model families and quantization | The bundled runtime covers the model-family and quantization surface supported by the pinned llama.cpp tree, including dense, MoE, recurrent/hybrid variants, and current GGML quantized formats. |
| Plugins | The versioned C ABI supports model, quantizer, storage, backend, and sampler extension kinds without private-core linkage. |
| Cross-platform | The core uses C99/CMake with POSIX and Win32 branches. The full layer uses C++17 and the same Vulkan API on Linux and Windows 11. |

## Build

A clean checkout builds the core and bundled full engine with:

```sh
cmake -S . -B build -DVG_ENABLE_VULKAN=ON -DVG_BUILD_FULL_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

To create a smaller core-only build, use `-DVG_BUILD_FULL_ENGINE=OFF`. To use a separately maintained llama.cpp checkout rather than the bundled copy, use `-DVG_LLAMA_CPP_PATH=/path/to/llama.cpp`.

## Run

The full runner uses lazy mmap loading by default:

```sh
./build/vg-run -m model.gguf -p "Write a short story" -n 128 -c 8192 -ngl 99
```

Use `-ngl 0` for CPU-only execution. Use `--eager` to disable lazy tensor loading, `--no-mmap` to disable memory mapping, `--threads N` to select CPU generation threads, and `--temp`, `--top-k`, and `--top-p` to control sampling.

On Linux, shared libraries produced by the bundled CMake subproject may require the build output directories in the runtime search path:

```sh
LD_LIBRARY_PATH=build/llama.cpp/bin:build/Release ./build/vg-run -m model.gguf -p "Hello" -n 32
```

On Windows, the corresponding DLL directory should be added to `PATH` before launching `vg-run.exe`.

## Performance contract

The hot path follows the supplied demoscene guidance: packed quantized data stays packed until the fused backend consumes it; the storage layer exposes bounded leases instead of whole-model materialization; pages are immutable when shared by forked sequences; and tracing is available for read latency, page faults, cache events, and dispatch timing. The next hardware-tuning step is platform-specific shader specialization for AMD RDNA2/RDNA3/RDNA4, including cooperative matrix and integer-dot variants selected from device capabilities.

## Validation

The included TinyLlamas fixture demonstrates an actual GGUF completion. The Linux validation run produced streamed text through both CPU execution and the Vulkan software device, with Vulkan logs confirming layer offload and a Vulkan KV buffer. This verifies the end-to-end path; it is not a substitute for native AMD RDNA hardware validation or Windows CI.
