# Status — vg-gguf-engine (updated 2026-09-27, session 2)

## TL;DR
- **Correct, coherent inference achieved and verified.** The Llama-3.2-1B now answers "The capital of France is Paris." matching llama.cpp exactly (top8 791=24.71 vs 24.62).
- **Root cause of all gibberish found: RoPE pairing convention.** ggml `GGML_ROPE_TYPE_NORMAL` (LLAMA arch) rotates ADJACENT pairs (2k,2k+1); the engine used NEOX-style (k, k+d/2). Position 0 is a no-op under both — which masked it in all short/position-0 tests.
- **COMPASS Vulkan methods implemented** (from G:\AMD-Ai\AMD-AI-COMPASS docs): persistent device-local weight cache, repacked aligned Q8_0 + `OpSDot` (dp4a) kernel, cooperative workgroup-per-row matvec, batched dispatch (one CB, fence, barriers, descriptor ring), reusable scratch, `_fseeki64` for >4GiB.
- **GPU results**: 1B TG **6.9→6.0 tok/s** (CPU 2.3, old GPU 0.39); 7B TG **0.75 tok/s** (CPU 0.18). Correct outputs both.
- **NeoHorse is not a normal GGUF** (JEV-style custom format) — its NaN is expected; deprioritized.

## Bugs fixed today (all verified)
1. Tokenizer crash: fallback re-added tokens with local cap → heap corruption. Removed; `added_cap` in struct.
2. `half_to_float` subnormal fp16 scales (Q8_0) were up to 1000× wrong → fixed in cpu_ops.c + quant.c (+ shader decode_f16).
3. **RoPE pairing (NORMAL=adjacent, not NEOX=half)** — the correctness root cause; fixed in `vg_cpu_rope` + both rope_freqs paths.
4. rope_freqs: guard compared `n_rope_freqs >= n_rot` (32 vs 64 → never used); formula must be `pos*base^(-2i/n_rot)/ff[i]`.
5. >4 GiB GGUFs opened with 32-bit `fseek(SEEK_END)` (MinGW) → `_fseeki64`.
6. Prompt file read with 32-bit `ftell` → grow-while-reading.
7. Llama-3 chat framing applied to non-Llama-3 vocabs → gated on `<|start_header_id|>` presence.
8. GPU: scratch aliasing (x/y/xd/xq shared one buffer) → slot-indexed scratch.

## Verified numbers (this box, RX 9070 XT)
| Model | Quant | CPU TG | GPU TG | Notes |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | Q8_0 | 2.30 t/s | 6.0–6.9 t/s | "The capital of France is Paris." (== llama.cpp) |
| GIGABATEMAN-7B | Q8_0 | 0.18 t/s | 0.75 t/s | "Paris, a city known for its rich…" |
| rocmforge-7b | Q8_0 | 0.26 t/s | — | coherent |

Per-matvec GPU breakdown: prep 39µs, **submit 184µs, read 88µs** → host round-trip per op dominates; GPU idle ~90%.

## Ground truth toolchain (built)
- llama.cpp fork from `H:\llamadx\llama.cpp\.llama-dx-reference` built at `C:\Users\rr\AppData\Local\Temp\opencode\llamaref2\bin` (llama-completion, llama-tokenize, llama-eval-callback) + `dump_logits.exe` (custom; f32 KV/FA-off).
- Numpy oracle `ref_llama.py` (f32 exact, rope_freqs; now adjacent-pair rope → matches llama.cpp).

## Next levers (in impact order)
1. **One command buffer per token** (COMPASS 3.1/8.4): move RMSNorm/RoPE/attention/SwiGLU onto GPU kernels (vg_vk_rmsnorm/rope/softmax/swiglu already exist; attention kernel needed) so an entire decode step is ONE submit. Removes the 184µs+88µs × N per-op round-trips.
2. Batch q/k/v + gate/up (batch API already in place) for a quick partial win.
3. Q4_K/Q6_K dp4a kernels (currently GPU = Q8_0/F32/F16 only; Q4_K_M CPU path works).
4. MoE FFN path (L3.2-8x3B still garbage) + hot-expert VRAM paging per user spec.
5. Dense split RAM/VRAM (`--ngl`), threadpool for CPU path.

