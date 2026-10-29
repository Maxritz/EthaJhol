# GPU execution: coverage, residency and dispatch

This document states exactly what the Vulkan backend executes today, what falls back to the
CPU, and how the device-resident paths are structured. It is the contract the model probe
(`vg_model_probe`) enforces: a model is only accepted if every operator it needs exists.

## Op coverage

| Operator | GPU | Notes |
| --- | --- | --- |
| Matrix-vector (`W * x`) for **F32** | Yes | `shaders/matvec_f32.comp` |
| Matrix-vector for **F16** | Yes | `shaders/matvec_f16.comp` |
| Matrix-vector for **Q4_0** | Yes | `shaders/matvec_q4_0.comp` (18-byte GGML blocks) |
| Matrix-vector for **Q8_0** | Yes | `shaders/matvec_q8_0.comp`, dp4a form (`GL_EXT_integer_dot_product`), 8 rows per workgroup |
| Matrix-vector for **I8** + scales | Yes | `shaders/matvec_i8.comp` |
| Matrix-vector for **Q4_K** | Yes | `shaders/matvec_q4_k.comp`, reads the native 144-byte blocks (no repack) |
| Matrix-vector for **Q6_K** | Yes | `shaders/matvec_q6_k.comp`, reads the native 210-byte blocks (no repack) |
| Activation quantisation to Q8 (per block) | Yes | `shaders/activation_quant_q8.comp` |
| RMSNorm | Yes | `shaders/rmsnorm.comp`, one workgroup per row, arbitrary row length |
| RoPE | Yes | `shaders/rope.comp` (scalar form) and `shaders/rope_adj.comp` (precomputed frequencies) |
| Attention scores | Yes | `shaders/attn_scores.comp` |
| Residual add | Yes | `shaders/resid_add.comp` |
| Softmax / SwiGLU | Yes | `shaders/softmax.comp`, `shaders/swiglu.comp` |
| Fused FFN (gate+up+swiglu+down, one command buffer) | Yes | `vg_vk_ffn_fused`, `vg_vk_ffn_fused_x2` (two MoE experts) |
| Fused layer tail / whole dense layer / MoE head | Yes | `vg_vk_layer_tail`, `vg_vk_dense_layer`, `vg_vk_dense_forward`, `vg_vk_dense_forward_final`, `vg_vk_moe_head` |
| Everything else (**Q2_K / Q3_K / Q5_K / Q8_K / Q4_1 / Q5_0 / Q5_1 / IQ\*, ternary**) | No | CPU fused kernels only |

`vg_quant_info(type)->supported_vulkan` is the single source of truth for the matvec table; the
graph asks before dispatching. A type that is reported as unsupported is never silently executed
with wrong math — it runs on the CPU. Adding a shader for a new type means adding it to
`shaders/`, to the backend's pipeline table, and to `g_info[].supported_vulkan` in `src/quant.c`.

## Residency model

There is no per-op upload. Three things stay resident for the process lifetime:

* **Persistent cache** — `vg_vk_cache_acquire(vk, key, tag, data, bytes, &buf)` uploads once,
  keyed by `(tensor pointer, tag)`. Tags separate roles that share a tensor pointer:
  `0x800` = attention output projection, `0x9000` = a norm weight, `0x10000` = router weights,
  `0x30000` = lm_head.
* **Host-visible scratch** — `vg_vk_scratch_acquire(vk, slot, bytes, &buf)`, grow-only, reused by
  every dispatch that needs a staging buffer.
* **Device-local scratch** — `vg_vk_dev_scratch_acquire(vk, slot, bytes, &buf)` for state the GPU
  reads and writes directly, such as the per-layer KV cache and the hidden state of the fused
  dense path.

The host-side expert cache is separate: `VG_ExpertCache` is keyed by
`(layer, expert, gate|up|down)` and resolves the slice
`blk.N.ffn_*_exps.weight + expert * stride` once per expert instead of once per token.
`vg_model_graph_expert_stats()` reports hits/misses/evictions and
`vg_model_graph_set_expert_cache()` resizes it.

## Dispatch

* Weight matvecs and the fused paths record **one command buffer and submit once**
  (`vkQueueSubmit` + a single fence wait). Per-token the dense path is one submit for all layers,
  not one per layer and certainly not one per op.
* `vkQueueWaitIdle` is never used: it idles the whole device and serialises every queue.
* Independent matvecs are batched into one submission through `VG_VKMatvecReq`
  (`vg_vk_matvec_q8_0_batch` and friends) with per-dispatch barriers between dependent ops.
* Descriptor sets are allocated per pipeline and rebound per dispatch; there is no pool churn on
  the hot path.

`VG_LAYER_TAIL=0` disables the fused layer tail, `VG_GPU_ATTN=0` disables the fused dense layer,
and `VG_MMTRACE=1` prints the per-op dispatch trace. `VG_DUMP=1` prints layer-0 activations.
The fused paths are automatically off when a LoRA adapter is attached, because the adapter delta
has to be applied on the host after each targeted matvec.

## Verifying a device

```sh
./build/vg-engine --vulkan-info        # adapter name, API version, heap size
./build/vg-engine --vulkan-self-test   # I8 matvec against a hand-computed result
./build/vg-engine probe model.gguf     # which ops/quants this model would need
./build/vg-run -m model.gguf -p "..." --ngl 1   # enable the GPU path
```

`--vulkan-self-test` uploads a 1×4 int8 row and its scale, runs the real shader, and expects
`+10.0` at scale 1.0 and `-5.0` at scale 0.5 — it exercises upload, descriptor binding and
dispatch, not just the shader source.

## Build and verification status

The shaders are compiled to SPIR-V at build time by `glslangValidator`, and the backend is only
compiled when a Vulkan SDK is found (`find_package(Vulkan)`). Without the SDK, CMake builds
`src/vulkan_stub.c`, which covers the whole backend API and reports `VG_E_UNSUPPORTED` from every
entry point, so the CPU build links and runs unchanged.

| Check | Result |
| --- | --- |
| `vulkan_backend.cpp` against Vulkan-Headers 1.3.290, `-Wall -Wextra -Wconversion -Wshadow -Wpedantic` | compiles, no warnings |
| `model_graph.c`, `native_engine.c`, `main.c`, `tensor_source.c` with `-DVG_HAS_VULKAN` | compile, no warnings |
| CPU-only build (`-DVG_ENABLE_VULKAN=OFF`) through `vulkan_stub.c` | builds, 8/8 suites pass |
| Device execution (`vk_matvec`, fused paths on real hardware) | **not verified** — no Vulkan ICD in the build environment |

Runtime GPU execution has therefore **not** been validated on a machine with a device. Treat the
coverage table and dispatch description above as the designed behaviour, and run
`--vulkan-self-test` on the target GPU before trusting a run.

## Offload granularity

`--ngl N` is a single switch: any value greater than zero asks for the GPU path, and zero keeps
everything on the CPU. Partial per-layer offload is not implemented, so the name is kept for
interface compatibility with llama.cpp-style tooling.
