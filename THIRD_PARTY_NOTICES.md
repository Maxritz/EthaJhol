# Third-Party Notices

The complete build bundles a shallow copy of the llama.cpp project under `third_party/llama.cpp`. The bundled tree is used as the full model graph, tokenizer, sampler, GGUF compatibility, CPU kernel, and Vulkan backend implementation behind `vg_full` and `vg-run`.

The upstream llama.cpp tree is distributed under the MIT License. Its complete license text is retained at `third_party/llama.cpp/LICENSE`. Additional dependency notices retained by the upstream tree are located under `third_party/llama.cpp/licenses` and `third_party/llama.cpp/gguf-py/LICENSE`. Redistributors should preserve those files together with the native engine `LICENSE`.

The native `vg-gguf-engine` files outside `third_party/llama.cpp` are distributed under the MIT License in the repository root.
