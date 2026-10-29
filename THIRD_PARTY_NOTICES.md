# Third-party notices

This repository contains no vendored third-party source tree. An earlier revision bundled a
shallow copy of llama.cpp under `third_party/llama.cpp`; it was removed (commit "remove vendored
third_party/llama.cpp and stale engine snapshot tarball") because the build never referenced it,
and the engine has since been a self-contained native implementation.

What this project does depend on at build/run time:

* **CMake** and **Ninja** as build tools, and a **C99/C++17** compiler.
* **Vulkan** headers, loader and SDK *optionally*: when `find_package(Vulkan)` succeeds the
  Vulkan backend and its GLSL shaders are compiled, otherwise `src/vulkan_stub.c` is used and
  the build is CPU-only.
* **glslangValidator** (optional, only for the Vulkan build) to compile `shaders/*.comp` to
  SPIR-V.

## Reference material, not code

The GGUF container format, the quantized block layouts, the SentencePiece/byte-level BPE
tokenizer contracts and the BitNet/Ternary-Bonsai quant types (TQ1_0, TQ2_0, Q1_0, Q2_0) are
implemented from their published specifications, using llama.cpp as the reference for the
on-disk layouts and for architecture naming. No llama.cpp source is copied or linked: the
decoders in `src/quant.c` and the graph in `src/model_graph.c` are independent implementations,
and the ternary decoders are pinned to upstream behaviour by golden vectors in
`tests/test_quant.c`.

The engine files are distributed under the MIT License in `LICENSE`.
