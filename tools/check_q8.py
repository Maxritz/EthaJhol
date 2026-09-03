import struct
f = open("/home/rr/models/llama-3.2-1b-q8.gguf", "rb")
f.seek(7831552)  # data_base
d = f.read(68)
# Q8_0 block: fp16 scale (2 bytes) + 32 int8 values = 34 bytes per block
scale0 = struct.unpack("<e", d[0:2])[0]
print(f"block0: fp16_scale={scale0}")
print(f"  first 8 qs values: {list(d[2:10])}")
print(f"  next 8 qs values:  {list(d[10:18])}")
scale1 = struct.unpack("<e", d[34:36])[0]
print(f"block1: fp16_scale={scale1}")
print(f"  first 8 qs values: {list(d[36:44])}")
f.close()
