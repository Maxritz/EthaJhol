import struct
import sys
from pathlib import Path


def s(value: str) -> bytes:
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def kv(key: str, value: str) -> bytes:
    return s(key) + struct.pack('<I', 8) + s(value)


def tensor(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    return s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', ggml_type, offset)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'tiny.gguf')
    metadata = kv('general.architecture', 'llama') + kv('general.name', 'vg-fixture')
    infos = tensor('token_embd.weight', [4], 24, 0) + tensor('output.scale', [1], 0, 32)
    header = struct.pack('<IIQQ', 0x46554747, 3, 2, 2) + metadata + infos
    data_base = (len(header) + 31) // 32 * 32
    blob = bytearray(data_base + 36)
    blob[:len(header)] = header
    blob[data_base:data_base + 4] = bytes([1, 254, 3, 252])
    struct.pack_into('<f', blob, data_base + 32, 0.25)
    out.write_bytes(blob)
    print(out, len(blob), data_base)


if __name__ == '__main__':
    main()
