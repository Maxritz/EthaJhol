import struct, sys
f = open(sys.argv[1], 'rb')
magic = struct.unpack('<I', f.read(4))[0]
version = struct.unpack('<I', f.read(4))[0]
tensor_count = struct.unpack('<Q', f.read(8))[0]
kv_count = struct.unpack('<Q', f.read(8))[0]
print('magic=%#x version=%d tensors=%d kv=%d' % (magic, version, tensor_count, kv_count))
for i in range(kv_count):
    klen = struct.unpack('<Q', f.read(8))[0]
    key = f.read(klen).decode()
    vtype = struct.unpack('<I', f.read(4))[0]
    if vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        val = f.read(slen).decode(errors='replace')
        print('  [%d] key=%r type=%d val=%r' % (i, key, vtype, val[:80]))
    elif vtype == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        print('  [%d] key=%r type=9 elem_type=%d count=%d' % (i, key, elem_type, n))
        show = min(n, 3)
        for j in range(show):
            if elem_type == 8:
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode(errors='replace')
                print('    [%d] %r' % (j, val))
            elif elem_type in (4,5,6):
                val = struct.unpack('<I', f.read(4))[0]
                print('    [%d] %d' % (j, val))
        remaining = n - show
        for j in range(remaining):
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
        if remaining > 0:
            print('    ... (skipped %d elements)' % remaining)
    else:
        if vtype in (0,7):
            val = struct.unpack('<B', f.read(1))[0]
        elif vtype in (4,5,6):
            val = struct.unpack('<I', f.read(4))[0]
        elif vtype in (2,3):
            val = struct.unpack('<H', f.read(2))[0]
        elif vtype in (10,11,12):
            val = struct.unpack('<Q', f.read(8))[0]
        else:
            val = None
        print('  [%d] key=%r type=%d val=%r' % (i, key, vtype, val))
f.close()
