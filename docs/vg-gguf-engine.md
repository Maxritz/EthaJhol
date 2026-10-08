# vg-gguf-engine

`vg-gguf-engine` is a portable GGUF inference engine with a compact C99 systems layer and a complete C++17 model-generation layer. It is designed for Windows 11 and Linux, uses Vulkan when available, and keeps large-model operation explicit through mmap/lazy loading, bounded tensor residency, paged KV state, radix prefix reuse, and plugin interfaces.

The project combines a native GGUF and storage core with a bundled full graph runtime. The native layer provides metadata inspection, typed tensor geometry validation, disk-backed tensor leases, host LRU caching, prefetch and streaming hooks, paged KV allocation, immutable sequence forks, radix prefix snapshots, tracing, runtime plugins, and a headless Vulkan quantized matvec test. The full layer exposes a stable C streaming API and a CLI that performs real model tokenization, prompt evaluation, decoding, sampling, and token-piece streaming through the bundled llama.cpp graph runtime.

## Capabilities

| Capability | Status |
| --- | --- |
| Real GGUF prompt completion | Available through `vg-run`. |
| CPU inference | Available with `-ngl 0`, including quantized GGML kernels supported by the bundled runtime. |
| Vulkan inference | Available with `-ngl 99` when a Vulkan device is present; the software Vulkan device is sufficient for smoke testing. |
| Large-model loading | mmap and lazy model loading are enabled by default in the full runner; `--eager` and `--no-mmap` are available for controlled comparisons. |
| Long context | Configurable context size, native paged KV/radix contracts, and runtime KV management. |
| Sampling | Temperature, top-k, top-p, repeat penalty, deterministic seed support in the C API, and streaming callbacks. |
| Dense, MoE, recurrent, and hybrid model families | Covered by the pinned bundled graph runtime. Unsupported files fail with diagnostics rather than silently falling back to incorrect math. |
| Extension model | Versioned C ABI for model, quantizer, storage, backend, and sampler plugins. |
| Operating systems | Linux is validated in the sandbox; C99/C++17, CMake, Vulkan, mmap, Win32 mapping, and Win32 dynamic-loading branches are included for Windows 11. |

## Build the complete engine

The default build includes the bundled llama.cpp runtime and Vulkan backend:

```sh
cmake -S . -B build -DVG_ENABLE_VULKAN=ON -DVG_BUILD_FULL_ENGINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

For a small embeddable systems-only build, disable the full graph runtime:

```sh
cmake -S . -B build-core -DVG_BUILD_FULL_ENGINE=OFF -DVG_ENABLE_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --config Release
ctest --test-dir build-core --output-on-failure
```

The full backend can also use an external llama.cpp checkout with `-DVG_LLAMA_CPP_PATH=/path/to/llama.cpp`. This is useful when tracking a newer upstream model implementation without changing the native C99 layer.

## Run a real model

The full runner uses mmap plus lazy model loading by default. The following command streams generated pieces to stdout:

```sh
LD_LIBRARY_PATH=build/llama.cpp/bin:build/Release ./build/vg-run \
  -m model.gguf -p "Write a short story about a Vulkan demo" \
  -n 128 -c 8192 -ngl 99 --temp 0.8 --top-k 40 --top-p 0.95
```

Use `-ngl 0` for CPU-only execution. Use `--eager` to keep the model tensors resident according to the runtime’s normal loading policy, `--no-mmap` to disable mapping, `--threads N` to set CPU generation threads, and `-c N` to select the context window. The C API in `include/vg/full_engine.h` lets a host application receive each token piece through `VG_TokenCallback`.

On Windows 11, build with Visual Studio or clang-cl through the same CMake project and add the generated llama runtime DLL directory to `PATH` before launching `vg-run.exe`.

## Native core tools

`vg-engine inspect model.gguf` prints GGUF architecture, tensor geometry, and storage information. `vg-engine stream model.gguf TENSOR` exercises a bounded lazy tensor lease. `vg-engine vk-selftest` validates the native Vulkan dispatch. `vg-tests` covers paged KV allocation, sequence fork behavior, and radix lookup. `vg-quant-tests` covers scalar conversion and legacy packed quantization. `vg-bench` measures page growth and radix lookup behavior.

## Architecture

The storage path is tiered: metadata remains resident, tensor payloads are addressed by file range, host cache entries are bounded and leased, and device backends may stage only the tiles required by the active graph. The long-context path treats KV pages as immutable after publication, allowing forked sequences and radix prefixes to share pages safely. Prefix identity includes model and adapter state so cached pages cannot cross incompatible execution configurations.

The demoscene optimization policy is deliberately practical. Quantized weights remain packed until the fused backend consumes them; hot loops are laid out for SIMD and branch-light execution; read-ahead is explicit rather than accidental; intermediate buffers are minimized; and trace hooks expose storage, page, cache, and dispatch costs before tuning. AMD RDNA-specific shader variants belong in backend plugins so the portable core does not acquire device-specific assumptions.

## License and third-party runtime

The native engine files are MIT licensed. The bundled `third_party/llama.cpp` tree retains its upstream license and notices. Consult `third_party/llama.cpp/LICENSE` and its accompanying notices when redistributing the complete bundle.

## References

[1]: https://github.com/FlashML-org/FreeToken "FreeToken repository"

[2]: https://github.com/JustVugg/colibri "colibri repository"

[3]: https://github.com/patcarter883/minisglang-rdna4 "mini-sglang RDNA4 repository"
