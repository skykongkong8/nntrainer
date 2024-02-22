import numpy as np

def inspect_file(file_name, _dtype = "float32"):
    with open(file_name, "rb") as f:
        while True:
            if (_dtype == "float32"):
                sz = int.from_bytes(f.read(4), byteorder="little")
            elif (_dtype =="float16"):
                sz = int.from_bytes(f.read(2), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            print(np.fromfile(f, dtype=_dtype, count=sz))

def float16_to_hex(value):
    float16_value = np.float16(value)
    hex_value = hex(np.float16(float16_value).view(np.uint16))
    return hex_value

def inspect_file_in_hex(file_name, _dtype = "float32"):
    with open(file_name, "rb") as f:
        while True:
            if (_dtype == "float32"):
                sz = int.from_bytes(f.read(4), byteorder="little")
            elif (_dtype =="float16"):
                sz = int.from_bytes(f.read(2), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            for i in np.fromfile(f, dtype=_dtype, count=sz):
                print(float16_to_hex(i), end="\t")
            print()

def inspect_embedding_mixed_file(file_name):
    with open(file_name, "rb") as f:
        idx = 1
        _dtype="float16"
        while True:
            if (idx == 2 or idx == 6):
                idx+=1
                _dtype="float32"
                sz = int.from_bytes(f.read(4), byteorder="little")
            else:
                idx+=1
                _dtype = "float16"
                sz = int.from_bytes(f.read(2), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            print(np.fromfile(f, dtype=_dtype, count=sz))


if __name__ == "__main__":  
    # inspect_file_in_hex("ln_axis_3_fp16fp16.nnlayergolden", _dtype = "float16")
    # inspect_file_in_hex("bn_width_training_fp16fp16.nnlayergolden", _dtype = "float16")

    # inspect_file("concat_dim3_fp16fp16.nnlayergolden", _dtype = "float16")
    # print("FP16")
    inspect_file('added_w16a16.nnlayergolden', _dtype="float16")
    print()
    # inspect_file('bn_channels_training.nnlayergolden')
    # inspect_file('dropout_20_training.nnlayergolden')
    # inspect_embedding_mixed_file("embedding_mixed_many.nnlayergolden")

        