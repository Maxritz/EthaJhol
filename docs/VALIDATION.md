# Validation Record

The feature-complete build was validated on Linux with GCC, CMake, the system Vulkan loader, and the software Vulkan device available in the sandbox. The repository is clean at commit `2c4ffc1`.

| Check | Result |
| --- | --- |
| Bundled Vulkan full-engine build | Passed. |
| CPU-only full-engine build | Passed. |
| Core-only no-Vulkan build | Passed. |
| C99 core tests | Passed. |
| Quantization tests | Passed. |
| Real GGUF full-generation CTest smoke | Passed. |
| Real CPU generation | Passed on `stories15M-q4_0.gguf`. |
| Real Vulkan generation | Passed on `stories15M-q4_0.gguf` with `-ngl 99`. |
| Native GGUF inspector | Parsed the real model index with 57 tensors. |
| Runtime plugin loading | Passed in the alpha validation and remains part of the build. |

The real CPU smoke command was:

```sh
./build-bundled/vg-run -m testdata/stories15M-q4_0.gguf -p 'Once upon a time' -n 12 -c 256 -ngl 0 --temp 0.0
```

It streamed: `, there was a little boy named Tim. He had a`

The real Vulkan smoke command was:

```sh
./build-bundled/vg-run -m testdata/stories15M-q4_0.gguf -p 'Once upon a time' -n 12 -c 256 -ngl 99 --temp 0.0
```

It streamed: `, there was a little piggy named Peep.` The process completed successfully and the full model bridge initialized Vulkan offload. The software device is a functional smoke target; native AMD RDNA2/RDNA3/RDNA4 hardware and Windows 11 CI still require testing outside this Linux sandbox.

The full runner defaults to mmap plus lazy loading. The native C99 tensor-source layer independently exposes bounded storage leases, range reads, prefetch hooks, and cache statistics; the bundled full runtime supplies the complete model-family graph and its own mmap/lazy implementation behind the C ABI.
