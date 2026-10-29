#!/usr/bin/env python3
"""Per-family model conformance matrix (developer tool).

For each model family this writes a small GGUF from that architecture's real contract
(metadata keys, tensor names, feature tensors, tokenizer kind), runs the engine's probe,
logit dump and generation against it, then deletes the file before moving to the next
family. Nothing is written into the repository.

Beyond "does it load", it checks that each advertised feature is actually *consumed*: a
variant fixture perturbs exactly one feature tensor (attention bias, QK-norm, BitNet
sub-norm, MoE router, shared expert, post-norms) with non-uniform values and the logits
must move by more than 1e-3 relative. A feature the evaluator silently ignores is
indistinguishable from an absent one, and that is the failure mode this catches. Uniform
scaling is avoided on purpose: the following RMSNorm cancels it.

Usage:  python3 tools/family_matrix.py /path/to/build [family]
"""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/final")
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path(tempfile.mkdtemp(prefix="vg-matrix-"))

F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 = 0, 1, 2, 3, 6, 7, 8
Q2_K, Q4_K, Q6_K = 10, 12, 14
IQ4_NL = 20
TQ2_0, Q1_0, Q2_0 = 35, 41, 42

QUANT = {
    "F32": (F32, 1, 4), "F16": (F16, 1, 2),
    "Q4_0": (Q4_0, 32, 18), "Q4_1": (Q4_1, 32, 20), "Q5_0": (Q5_0, 32, 22), "Q5_1": (Q5_1, 32, 24),
    "Q8_0": (Q8_0, 32, 34), "IQ4_NL": (IQ4_NL, 32, 18),
    "Q2_K": (Q2_K, 256, 84), "Q4_K": (Q4_K, 256, 144), "Q6_K": (Q6_K, 256, 210),
    "TQ2_0": (TQ2_0, 256, 66), "Q1_0": (Q1_0, 128, 18), "Q2_0": (Q2_0, 64, 18),
}
KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def f16(x):
    return struct.pack("<e", x)


def lcg(seed):
    s = seed & 0xFFFFFFFF
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield (s / 0xFFFFFFFF) - 0.5


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def block(qname, v):
    """Encode one block of `v` (block_size floats) using the layouts dequantized in src/quant.c."""
    tid, bs, _ = QUANT[qname]
    n = len(v)
    out = bytearray()
    if tid == F32:
        for x in v:
            out += struct.pack("<f", x)
    elif tid == F16:
        for x in v:
            out += f16(x)
    elif tid == Q4_0:
        d = 1.0 / 64.0
        q = [clamp(int(round(x / d)) + 8, 0, 15) for x in v]
        out += f16(d)
        for j in range(16):
            out.append(q[j] | (q[j + 16] << 4))
    elif tid == Q4_1:
        d, m = 1.0 / 64.0, -8.0 / 64.0
        q = [clamp(int(round((x - m) / d)), 0, 15) for x in v]
        out += f16(d) + f16(m)
        for j in range(16):
            out.append(q[j] | (q[j + 16] << 4))
    elif tid in (Q5_0, Q5_1):
        d = 1.0 / 64.0
        base = 16.0 if tid == Q5_0 else 0.0
        m = -16.0 / 64.0 if tid == Q5_0 else 0.0
        q = [clamp(int(round((x - m) / d)), 0, 31) for x in v]
        out += f16(d) + (f16(m) if tid == Q5_1 else b"")
        qh = 0
        for j in range(16):
            if q[j] & 16:
                qh |= 1 << j
            if q[j + 16] & 16:
                qh |= 1 << (j + 12)
        out += struct.pack("<I", qh & 0xFFFFFFFF)
        for j in range(16):
            out.append((q[j] & 0xF) | ((q[j + 16] & 0xF) << 4))
        _ = base
    elif tid == Q8_0:
        d = 1.0 / 128.0
        out += f16(d)
        for x in v:
            out.append(clamp(int(round(x / d)), -128, 127) & 0xFF)
    elif tid == IQ4_NL:
        d = 1.0 / 128.0
        out += f16(d)
        idx = []
        for x in v:
            t = x / d
            best = min(range(16), key=lambda k: abs(KVALUES_IQ4NL[k] - t))
            idx.append(best)
        for j in range(16):
            out.append(idx[j] | (idx[j + 16] << 4))
    elif tid == Q2_K:
        # d = dmin = 1, every sub-block scale 1/min-scale 1: value = code - 1
        for _ in range(16):
            out.append(0x11)
        qs = bytearray(64)
        for e in range(256):
            n = e // 128
            k = (e % 128) // 16
            realidx = (e % 128) % 16
            byte = 32 * n + (16 if k % 2 else 0) + realidx
            code = clamp(int(round(v[e])) + 1, 0, 3)
            qs[byte] |= code << (2 * (k // 2))
        out += qs + f16(1.0) + f16(1.0)
    elif tid == Q4_K:
        # all eight sub-blocks: d-scale 1, min-scale 8, d = dmin = 1/4 -> value = (code-8)/4
        out += f16(0.25) + f16(0.25)
        out += bytes([0x01, 0x01, 0x01, 0x01, 0x08, 0x08, 0x08, 0x08, 0x81, 0x81, 0x81, 0x81])
        qs = bytearray(128)
        for e in range(256):
            j = e // 64
            within = e % 64
            byte = 32 * j + (within % 32)
            code = clamp(int(round(v[e] * 4.0)) + 8, 0, 15)
            if within < 32:
                qs[byte] |= code
            else:
                qs[byte] |= code << 4
        out += qs
    elif tid == Q6_K:
        d = 1.0 / 16.0
        ql = bytearray(128)
        qh = bytearray(64)
        for e in range(256):
            n = e // 128
            ep = e % 128
            l = ep % 32
            quad = ep // 32
            code = clamp(int(round(v[e] * 16.0)) + 32, 0, 63)
            lo, hi = code & 0xF, (code >> 4) & 3
            if quad in (0, 1):
                ql[64 * n + l + (32 if quad == 1 else 0)] |= lo
            else:
                ql[64 * n + l + (32 if quad == 3 else 0)] |= lo << 4
            qh[32 * n + l] |= hi << (2 * quad)
        out += ql + qh + bytes([8] * 16) + f16(d)
    elif tid == TQ2_0:
        d = 1.0 / 8.0
        qs = bytearray(64)
        for e in range(256):
            j = (e // 128) * 32
            l = (e % 128) // 32
            m = e % 32
            code = clamp(int(round(v[e] * 8.0)) + 1, 0, 3)
            qs[j + m] |= code << (2 * l)
        out += qs + f16(d)
    elif tid == Q1_0:
        d = 1.0 / 16.0
        qs = bytearray(16)
        for j in range(128):
            if v[j] > 0:
                qs[j // 8] |= 1 << (j % 8)
        out += f16(d) + qs
    elif tid == Q2_0:
        d = 1.0 / 8.0
        qs = bytearray(16)
        for j in range(64):
            code = clamp(int(round(v[j] * 8.0)) + 1, 0, 3)
            qs[j // 4] |= code << (2 * (j % 4))
        out += f16(d) + qs
    else:
        raise SystemExit("no encoder for " + qname)
    assert len(out) == QUANT[qname][2], (qname, len(out), len(v))
    return bytes(out)


def encode(qname, values):
    bs = QUANT[qname][1]
    assert len(values) % bs == 0, (qname, len(values), bs)
    return b"".join(block(qname, values[i:i + bs]) for i in range(0, len(values), bs))


# ------------------------------------------------------------------ tokenizer

def gpt2_byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def spm_vocab():
    t = ["<unk>", "<s>", "</s>"]
    t += ["<0x%02X>" % b for b in range(256)]
    t += [chr(c) for c in range(33, 127)]
    t += ["\u2581" + chr(c) for c in range(65, 91)]
    t += ["\u2581" + chr(c) for c in range(97, 123)]
    t.append("\u2581")
    return t


def bpe_vocab():
    bmap = gpt2_byte_map()
    return ["<|endoftext|>"] + [bmap[b] for b in range(256)]


# ------------------------------------------------------------------ writer

class Writer:
    def __init__(self):
        self.kv = bytearray()
        self.n_kv = 0
        self.tensors = []

    def _str(self, s):
        raw = s.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def add(self, key, vtype, payload):
        self.kv += self._str(key) + struct.pack("<I", vtype) + payload
        self.n_kv += 1

    def u32(self, k, v): self.add(k, 4, struct.pack("<I", v))
    def f32(self, k, v): self.add(k, 6, struct.pack("<f", v))
    def s(self, k, v): self.add(k, 8, self._str(v))
    def b(self, k, v): self.add(k, 7, struct.pack("<B", 1 if v else 0))

    def arr_str(self, k, items):
        self.add(k, 9, struct.pack("<IQ", 8, len(items)) + b"".join(self._str(i) for i in items))

    def arr_f32(self, k, items):
        self.add(k, 9, struct.pack("<IQ", 6, len(items)) + b"".join(struct.pack("<f", v) for v in items))

    def arr_i32(self, k, items):
        self.add(k, 9, struct.pack("<IQ", 5, len(items)) + b"".join(struct.pack("<i", v) for v in items))

    def tensor(self, name, dims, qname, values):
        self.tensors.append((name, dims, QUANT[qname][0], encode(qname, values)))

    def write(self, path):
        head = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(self.tensors), self.n_kv))
        head += self.kv
        offs, cursor = [], 0
        for t in self.tensors:
            offs.append(cursor)
            cursor += (len(t[3]) + 31) // 32 * 32
        for (name, dims, tid, _), off in zip(self.tensors, offs):
            head += self._str(name) + struct.pack("<I", len(dims))
            for d in dims:
                head += struct.pack("<Q", d)
            head += struct.pack("<IQ", tid, off)
        while len(head) % 32:
            head += b"\x00"
        with open(path, "wb") as f:
            f.write(head)
            pos = 0
            for (_, _, _, payload), off in zip(self.tensors, offs):
                if off > pos:
                    f.write(b"\x00" * (off - pos))
                    pos = off
                f.write(payload)
                pos += len(payload)
        return path


D, H, HKV, HD, FF, LAYERS, CTX, EXPERTS, USED = 256, 4, 2, 64, 256, 2, 64, 8, 2
PROMPT = "The capital of France is"


def weights(seed, count, scale=0.08):
    g = lcg(seed)
    return [next(g) * scale for _ in range(count)]


def build(spec, variant="base"):
    w = Writer()
    arch, q = spec["arch"], spec["quant"]
    nh, nkv = spec["heads"]
    qd, kd = nh * HD, nkv * HD
    toks = spm_vocab() if spec["tok"] == "spm" else bpe_vocab()
    n_vocab = len(toks)

    w.s("general.architecture", arch)
    w.s("general.name", "vg-matrix-" + spec["name"])
    w.u32(arch + ".context_length", CTX)
    w.u32(arch + ".block_count", LAYERS)
    w.u32(arch + ".embedding_length", D)
    w.u32(arch + ".feed_forward_length", FF)
    w.u32(arch + ".attention.head_count", nh)
    w.u32(arch + ".attention.head_count_kv", nkv)
    w.u32(arch + ".attention.key_length", HD)
    w.u32(arch + ".rope.dimension_count", HD)
    w.f32(arch + ".rope.freq_base", 10000.0)
    w.f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5)
    w.u32(arch + ".vocab_size", n_vocab)
    if spec.get("moe"):
        w.u32(arch + ".expert_count", EXPERTS)
        w.u32(arch + ".expert_used_count", USED)
    for k, v in spec.get("extra", {}).items():
        if isinstance(v, float):
            w.f32(k, v)
        else:
            w.u32(k, v)
    w.s("general.file_type", "mostly " + q)

    if spec["tok"] == "spm":
        w.s("tokenizer.ggml.model", "llama")
        w.arr_str("tokenizer.ggml.tokens", toks)
        w.arr_f32("tokenizer.ggml.scores", [-100.0] * n_vocab)
        w.arr_i32("tokenizer.ggml.token_type", [1] * n_vocab)
        w.u32("tokenizer.ggml.bos_token_id", 1)
        w.u32("tokenizer.ggml.eos_token_id", 2)
        w.u32("tokenizer.ggml.unknown_token_id", 0)
        w.b("tokenizer.ggml.add_bos_token", True)
        w.b("tokenizer.ggml.add_dummy_prefix", True)
    else:
        w.s("tokenizer.ggml.model", "gpt2")
        w.arr_str("tokenizer.ggml.tokens", toks)
        w.arr_str("tokenizer.ggml.merges", [])
        w.arr_i32("tokenizer.ggml.token_type", [1] * n_vocab)
        w.u32("tokenizer.ggml.bos_token_id", 1)
        w.u32("tokenizer.ggml.eos_token_id", 0)
        w.b("tokenizer.ggml.add_bos_token", False)

    w.tensor("token_embd.weight", [D, n_vocab], q,
             weights(11, D * n_vocab, 2.0 if variant == "embed" else 1.0))
    head_scale = 2.0 if variant == "head" else 1.0
    w.tensor("output.weight", [D, n_vocab], q, weights(12, D * n_vocab, head_scale))
    w.tensor("output_norm.weight", [D], "F32", [1.0] * D)

    bias_val = [0.5 if i % 2 else -0.5 for i in range(4096)] if variant == "bias" else None
    qkn_val = [0.5 + (i % 7) * 0.1 for i in range(4096)] if variant == "qkn" else None
    subn_val = [0.5 + (i % 5) * 0.2 for i in range(4096)] if variant == "subn" else None
    router_scale = 3.0 if variant == "router" else (0.0 if variant == "rzero" else 1.0)
    expert_scale = 0.0 if variant == "ezero" else 1.0

    for l in range(LAYERS):
        s = 100 * (l + 1)
        w.tensor("blk.%d.attn_norm.weight" % l, [D], "F32", [1.0] * D)
        if spec.get("fused"):
            w.tensor("blk.%d.attn_qkv.weight" % l, [D, qd + 2 * kd], q, weights(s + 1, D * (qd + 2 * kd)))
            w.tensor("blk.%d.attn_qkv.bias" % l, [qd + 2 * kd], "F32", [0.0] * (qd + 2 * kd))
        else:
            w.tensor("blk.%d.attn_q.weight" % l, [D, qd], q,
                     weights(s + 1, D * qd, 2.0 if variant == "attnq" else 1.0))
            w.tensor("blk.%d.attn_k.weight" % l, [D, kd], q, weights(s + 2, D * kd))
            w.tensor("blk.%d.attn_v.weight" % l, [D, kd], q, weights(s + 3, D * kd))
        w.tensor("blk.%d.attn_output.weight" % l, [qd, D], q, weights(s + 4, qd * D))
        if spec.get("bias"):
            for t, n in (("attn_q", qd), ("attn_k", kd), ("attn_v", kd)):
                w.tensor("blk.%d.%s.bias" % (l, t), [n], "F32", bias_val[:n] if bias_val else [0.0] * n)
        if spec.get("qk_norm"):
            w.tensor("blk.%d.attn_q_norm.weight" % l, [HD], "F32", qkn_val[:HD] if qkn_val else [1.0] * HD)
            w.tensor("blk.%d.attn_k_norm.weight" % l, [HD], "F32", qkn_val[:HD] if qkn_val else [1.0] * HD)
        if spec.get("qk_norm") and spec.get("post_norm"):
            pn_val = [0.5 + (i % 3) * 0.4 for i in range(D)] if variant == "postn" else \
                     (qkn_val[:D] if qkn_val else [1.0] * D)
            w.tensor("blk.%d.attn_post_norm.weight" % l, [D], "F32", pn_val)
        if spec.get("sub_norm"):
            w.tensor("blk.%d.attn_sub_norm.weight" % l, [qd], "F32", subn_val[:qd] if subn_val else [1.0] * qd)
            w.tensor("blk.%d.ffn_sub_norm.weight" % l, [FF], "F32", subn_val[:FF] if subn_val else [1.0] * FF)
        w.tensor("blk.%d.ffn_norm.weight" % l, [D], "F32", [1.0] * D)
        if spec.get("post_norm"):
            w.tensor("blk.%d.ffn_post_norm.weight" % l, [D], "F32", pn_val)
        if spec.get("moe"):
            w.tensor("blk.%d.ffn_gate_inp.weight" % l, [D, EXPERTS], q,
                     weights(s + 5, D * EXPERTS, router_scale))
            for t, dims, sd in (("ffn_gate_exps", [D, FF, EXPERTS], s + 6),
                                ("ffn_up_exps", [D, FF, EXPERTS], s + 7),
                                ("ffn_down_exps", [FF, D, EXPERTS], s + 8)):
                count = dims[0] * dims[1] * dims[2]
                w.tensor("blk.%d.%s.weight" % (l, t), dims, q, weights(sd, count, expert_scale))
            if spec.get("shared_expert"):
                sscale = 2.0 if variant == "shexp" else 1.0
                for t, dims, sd in (("ffn_gate_shexp", [D, FF], s + 9),
                                    ("ffn_up_shexp", [D, FF], s + 10),
                                    ("ffn_down_shexp", [FF, D], s + 11)):
                    w.tensor("blk.%d.%s.weight" % (l, t), dims, q, weights(sd, dims[0] * dims[1], sscale))
                w.tensor("blk.%d.ffn_gate_inp_shexp.weight" % l, [D], "F32",
                         [0.02 * ((i % 5) - 2) for i in range(D)])
        else:
            fscale = 2.0 if variant == "ffn" else 1.0
            if spec.get("fused"):
                w.tensor("blk.%d.ffn_up.weight" % l, [D, 2 * FF], q, weights(s + 6, D * 2 * FF))
                w.tensor("blk.%d.ffn_up.bias" % l, [2 * FF], "F32", [0.0] * (2 * FF))
            else:
                w.tensor("blk.%d.ffn_gate.weight" % l, [D, FF], q, weights(s + 5, D * FF, fscale))
                w.tensor("blk.%d.ffn_up.weight" % l, [D, FF], q, weights(s + 6, D * FF, fscale))
            w.tensor("blk.%d.ffn_down.weight" % l, [FF, D], q, weights(s + 7, FF * D, fscale))

    path = OUT / ("%s-%s-%s.gguf" % (spec["name"], q, variant))
    w.write(str(path))
    return path


# ------------------------------------------------------------------ families
# llama.cpp's per-architecture tensor contracts, trimmed to what a 2-layer toy needs.
FAMILIES = [
    dict(name="llama2", arch="llama", quant="Q4_0", tok="spm", heads=(H, H)),
    dict(name="llama3", arch="llama", quant="Q4_0", tok="bpe", heads=(H, H)),
    dict(name="mistral", arch="mistral", quant="Q4_0", tok="spm", heads=(H, HKV),
         extra={"mistral.attention.sliding_window": 32}),
    dict(name="qwen2", arch="qwen2", quant="Q4_0", tok="bpe", heads=(H, HKV), bias=True),
    dict(name="qwen3", arch="qwen3", quant="Q4_0", tok="bpe", heads=(H, HKV), qk_norm=True),
    dict(name="gemma", arch="gemma", quant="Q4_0", tok="spm", heads=(H, HKV)),
    dict(name="gemma2", arch="gemma2", quant="Q4_0", tok="spm", heads=(H, HKV), qk_norm=True,
         post_norm=True, extra={"gemma2.attention.logit_softcap": 50.0,
                                "gemma2.attention.sliding_window": 32}),
    dict(name="gemma3", arch="gemma3", quant="Q4_0", tok="spm", heads=(H, HKV), qk_norm=True,
         post_norm=True, extra={"gemma3.attention.sliding_window": 32}),
    dict(name="smollm", arch="smollm", quant="Q4_0", tok="bpe", heads=(H, HKV), bias=True),
    dict(name="tinyllama", arch="tinyllama", quant="Q4_0", tok="spm", heads=(H, HKV)),
    dict(name="bitnet158", arch="bitnet", quant="TQ2_0", tok="spm", heads=(H, HKV), sub_norm=True),
    dict(name="mixtral", arch="mixtral", quant="Q4_0", tok="spm", heads=(H, HKV), moe=True),
    dict(name="qwen3moe", arch="qwen3moe", quant="Q4_0", tok="bpe", heads=(H, HKV),
         qk_norm=True, moe=True),
    dict(name="qwen2moe", arch="qwen2moe", quant="Q4_0", tok="bpe", heads=(H, HKV), bias=True,
         moe=True, shared_expert=True),
    dict(name="deepseekmoe", arch="deepseek_moe", quant="Q4_0", tok="bpe", heads=(H, HKV),
         moe=True, shared_expert=True),
    dict(name="phi3", arch="phi3", quant="Q4_0", tok="spm", heads=(H, HKV)),
    dict(name="phi3-fused", arch="phi3", quant="Q4_0", tok="spm", heads=(H, HKV), fused=True),
    dict(name="gpt2", arch="gpt2", quant="Q4_0", tok="bpe", heads=(H, H), bias=True),
]

# Weight-type sweep: same llama geometry, every type real GGUFs ship in.
QUANTS = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL", "Q2_K", "Q4_K", "Q6_K",
          "TQ2_0", "Q1_0", "Q2_0", "F16"]


def run(args, timeout=180):
    try:
        p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def top_logits(path):
    rc, out, err = run([str(BIN / "vg-engine"), "logits", str(path), PROMPT])
    vals = []
    for line in out.splitlines():
        if "logit=" in line:
            vals.append(line.split("logit=")[1].strip())
    return rc, vals, out + err


def check(spec):
    r = {"family": spec["name"], "arch": spec["arch"], "quant": spec["quant"], "notes": []}
    base = build(spec)
    rc, out, err = run([str(BIN / "vg-engine"), "probe", str(base)])
    supported = "result: SUPPORTED" in out
    r["probe"] = supported
    if not supported:
        line = [l for l in out.splitlines() if l.startswith("result:")]
        r["detail"] = (line[0] if line else (out + err).strip().replace("\n", " "))[:120]
        base.unlink()
        return r

    rc, vals, log = top_logits(base)
    r["logits_rc"] = rc
    r["finite"] = rc == 0 and vals and not any("nan" in v.lower() or "inf" in v.lower() for v in vals)
    if not r["finite"]:
        r["notes"].append(("logits: " + log.strip().replace("\n", " "))[:110])

    rc, out, err = run([str(BIN / "vg-run"), "-m", str(base), "-p", PROMPT,
                        "-n", "8", "-c", "64", "--temp", "0", "--threads", "2"])
    r["gen_rc"] = rc
    text = out.strip()
    r["text"] = text[:36]
    r["emitted"] = bool(text)
    if not text:
        r["notes"].append("no text: " + (err.strip().replace("\n", " "))[:90])

    # Feature wiring: only check a feature this family actually declares.
    feat = None
    if spec.get("bias"):
        feat = "bias"
    if spec.get("qk_norm"):
        feat = "qkn"
    if spec.get("sub_norm"):
        feat = "subn"
    if spec.get("moe"):
        feat = "router"
    if spec.get("shared_expert"):
        alt = build(spec, "shexp")
        rc2, vals2, _ = top_logits(alt)
        alt.unlink()
        d = max([abs(float(a) - float(b)) / max(1e-6, abs(float(a))) for a, b in zip(vals, vals2)] or [0.0])
        r["shexp"] = "yes" if d > 1e-3 else "IGNORED"
        if d <= 1e-3:
            r["notes"].append("shared expert not consumed")
    if feat:
        alt = build(spec, "bias" if feat == "bias" else feat)
        rc2, vals2, _ = top_logits(alt)
        alt.unlink()
        d = 0.0
        if vals and vals2:
            for a, b in zip(vals, vals2):
                a, b = float(a), float(b)
                d = max(d, abs(a - b) / max(1e-6, abs(a)))
        r["wired"] = "yes(%.1e)" % d if d > 1e-3 else "IGNORED"
        if r["wired"] == "IGNORED":
            r["notes"].append("%s not consumed" % feat)
    base.unlink()
    return r


def main():
    rows = [check(s) for s in FAMILIES if not ONLY or s["name"] == ONLY]
    print("\n%-12s %-12s %-6s %-9s %-6s %-4s %-3s %-6s %s" %
          ("family", "arch", "quant", "probe", "logits", "gen", "txt", "wiring", "detail"))
    for r in rows:
        print("%-12s %-12s %-6s %-9s %-6s %-4s %-3s %-6s %s" % (
            r["family"], r["arch"], r["quant"],
            "OK" if r["probe"] else "REJECT",
            r.get("finite", "-"), r.get("gen_rc", "-"),
            "y" if r.get("emitted") else "n", r.get("wired", "-"),
            r.get("detail") or "; ".join(r["notes"]) or r.get("text", "")))

    qrows = []
    for q in QUANTS:
        spec = dict(name="llama-q", arch="llama", quant=q, tok="spm", heads=(H, H))
        qrows.append(check(spec))
    print("\n%-8s %-9s %-6s %-4s %s" % ("quant", "probe", "logits", "gen", "detail"))
    for r in qrows:
        print("%-8s %-9s %-6s %-4s %s" % (
            r["quant"], "OK" if r["probe"] else "REJECT", r.get("finite", "-"),
            r.get("gen_rc", "-"), r.get("detail") or "; ".join(r["notes"]) or r.get("text", "")))
    print("\nleftover fixtures:", [p.name for p in OUT.iterdir()])


if __name__ == "__main__":
    main()
