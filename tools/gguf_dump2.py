import struct, sys
f = open(sys.argv[1], 'rb')
magic = struct.unpack('<I', f.read(4))[0]
version = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<Q', f.read(8))[0]
kv_count = struct.unpack('<Q', f.read(8))[0]
print('header: magic=%#x version=%d tensors=%d kv=%d' % (magic, version, tensor_count, kv_count))
for i in range(kv_count):
    pos = f.tell()
    klen = struct.unpack('<Q', f.read(8))[0]
    key = f.read(klen).decode()
    vtype_pos = f.tell()
    vtype = struct.unpack('<I', f.read(4))[0]
    if key == 'tokenizer.ggml.tokens':
        print('FOUND at offset %#x' % pos)
        print('  key_len=%d key=%r' % (klen, key))
        print('  vtype=%d at offset %#x' % (vtype, vtype_pos))
        if vtype == 9:
            et_pos = f.tell()
            elem_type = struct.unpack('<I', f.read(4))[0]
            n_pos = f.tell()
            n = struct.unpack('<Q', f.read(8))[0]
            print('  elem_type=%d at offset %#x' % (elem_type, et_pos))
            print('  count=%d at offset %#x' % (n, n_pos))
            print('  first element starts at offset %#x' % f.tell())
            # Read first 3 elements
            for j in range(min(n, 3)):
                el_pos = f.tell()
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode(errors='replace')
                print('  elem[%d] at offset %#x: slen=%d val=%r' % (j, el_pos, slen, val))
            # Now check what our C code would read
            # C reads: u32 for elem_type (4 bytes), u64 for count (8 bytes)
            # After that, for each element: read_string which reads u64 length then string
            print()
            print('  === What C read_array_text would do ===')
            print('  C reads u32 at offset %#x -> elem_type' % et_pos)
            print('  C reads u64 at offset %#x -> count' % n_pos)
            print('  Then reads first string at offset %#x' % f.tell())
        break
    else:
        if vtype == 8:
            slen = struct.unpack('<Q', f.read(8))[0]
            f.read(slen)
        elif vtype == 9:
            elem_type = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            for j in range(n):
                if elem_type == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.read(slen)
                elif elem_type in (0,7):
                    f.read(1)
                elif elem_type in (2,3):
                    f.read(2)
                elif elem_type in (4,5,6):
                    f.read(4)
                elif elem_type in (10,11,12):
                    f.read(8)
        elif vtype in (0,7):
            f.read(1)
        elif vtype in (2,3):
            f.read(2)
        elif vtype in (4,5,6):
            f.read(4)
        elif vtype in (10,11,12):
            f.read(8)
f.close()
