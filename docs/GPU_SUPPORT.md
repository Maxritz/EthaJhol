# GPU execution: coverage, residency and dispatch

This document states exactly what the Vulkan backend executes today, what falls back to the
CPU, and how the resident-weight path is structured. It is the contract the model probe
(`vg_model_probe`) enforces: a model is only accepted if every operator it needs exists.

## Op coverage

| Operator | GPU | Notes |
| --- | --- | --- |
| Matrix-vector (`W * x`) for **F32** | Yes | `shaders/matvec_f32.comp` |
| Matrix-vector for **F16** | Yes | `shaders/matvec_f16.comp` |
| Matrix-vector for **Q4_0** | Yes | `shaders/matvec_q4_0.comp` (18-byte GGML blocks, packed nibbles consumed on device) |
| Matrix-vector for **Q8_0** | Yes | `shaders/matvec_q8_0.comp` (34-byte GGML blocks) |
| Matrix-vector for **Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_K / Q4_1 / Q5_0 / Q5_1 / IQ4_NL** | No | CPU fused kernels only |
| RMSNorm / Softmax / RoPE / SwiGLU | API available (`vg_vk_rmsnorm`, `vg_vk_softmax`, `vg_vk_rope`, `vg_vk_swiglu`) | The graph keeps these on the CPU: they are small vectors, so staging them across PCIe costs more than the arithmetic. |
| Attention scores / KV cache | CPU | Paged host KV cache |

`vg_vk_supports_type()` and `vg_quant_info(type)->supported_vulkan` are the single source of
truth for this table; the graph asks them before dispatching. A type that is reported as
unsupported is never silently executed with wrong math — it runs on the CPU.

For a **Q4_K_M / Q6_K** model, therefore, the GPU path is currently unused and everything runs
on the CPU. That is a deliberate, documented limit rather than a silent fallback: adding a
shader for a new type means adding it to `shaders/`, `pipeline_for()`, `vg_vk_supports_type()`
and `g_info[].supported_vulkan` in `src/quant.c`.

## Resident weights

`VG_VKWeight` is the device-side representation of one weight tensor: a device buffer holding
the packed GGML bytes plus its geometry. `vg_vk_weight_upload()` is called once per weight and
`vg_vk_matvec()` reuses that buffer for every token.

`src/model_graph.c` owns an LRU cache of these (`vk_slots`, 32 entries for dense models, 16 for
MoE). A weight is keyed by its tensor data pointer, which is stable because tensors come from
the read-only mmap. When the cache is full the least recently used entry is released instead of
uploading a duplicate. For MoE models this is also the **expert cache on the device**: routed
experts stay resident across tokens and are evicted by LRU when a different expert set is used.

The host-side expert cache (`VG_ExpertCache`) is a separate, always-on cache keyed by
`(layer, expert, gate|up|down)`. It avoids re-resolving the stacked expert slice
(`blk.N.ffn_*_exps.weight` + `expert * stride`) on every token and keeps statistics available
through `vg_model_graph_expert_stats()`.

## Dispatch

Each dispatch uses:

* one **persistent** command buffer and one fence per `VG_VK` (allocated at open, reused
  forever) instead of allocating/freeing a command buffer per call;
* **reused staging buffers** for the input and output vectors, grown only when a larger vector
  appears;
* one `vkQueueSubmit` followed by `vkWaitForFences` — never `vkQueueWaitIdle`, which idles the
  whole device and serializes every queue;
* one pre-allocated descriptor set per pipeline, updated with `vkUpdateDescriptorSets` per
  dispatch (no descriptor pool churn).

Per-op cost is therefore one submit + one fence wait, not a full device idle plus allocation.

## Verifying a device

```sh
./build/vg-engine --vulkan-info        # adapter name, API version, heap size
./build/vg-engine --vulkan-self-test   # Q4_0 matvec against a hand-computed result
./build/vg-engine probe model.gguf     # which ops/quants this model would need
./build/vg-run -m model.gguf -p "..." --ngl 1   # enable the GPU path
```

`--vulkan-self-test` builds two Q4_0 rows whose weights are exactly `+1.0`, runs the real
shader, and expects `32.0` per row for a 32-element all-ones input.

## Continuous integration status

The shaders are compiled to SPIR-V at build time by `glslangValidator`, and the backend is
compiled only when a Vulkan SDK is found (`find_package(Vulkan)`). On a machine without the SDK
CMake falls back to `src/vulkan_stub.c`, so the CPU build is unaffected.

What has been verified for this change, in the CPU-only container used to build it:

| Check | Result |
| --- | --- |
| All 8 shaders → SPIR-V (`glslangValidator` 16.6.0) | pass, non-empty modules (1.8–6.9 KiB) |
| `vulkan_backend.cpp` against Vulkan-Headers 1.3.290, `-Wall -Wextra -Wconversion -Wshadow -Wpedantic` | pass, no warnings |
| CMake Vulkan build (`-DVulkan_INCLUDE_DIR=… -DVulkan_LIBRARY=…`), shaders compiled by the build | pass |
| `ctest` on that build (CPU fallback path active) | 6/6 pass |
| `vg-engine --vulkan-info` with a loader but **no ICD** | reports `Vulkan unavailable`, exit non-zero, no crash |
| `vg-run --ngl 99` with no ICD | falls back to CPU, output identical to `--ngl 0` |
| Device execution (`vk_matvec` on real hardware) | **not verified** — no ICD in this environment |

Runtime GPU execution has therefore **not** been validated on a machine with a device. Treat the
coverage table and dispatch description above as the designed behaviour, and run
`--vulkan-self-test` on the target GPU before trusting a run.

## Offload granularity

`--ngl N` is a single switch: any value greater than zero asks for the GPU path for every weight
whose type has a shader, and zero keeps everything on the CPU. Partial per-layer offload is not
implemented, so the name is kept for interface compatibility with llama.cpp-style tooling.
