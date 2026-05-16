import os, sys
from os import PathLike
import json
import struct
import torch

from typing import Tuple, List

SAFETENSORS_NOT_TENSOR_KEYS = ("__metadata__", )
SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32":torch.float32, 
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    #"U64": torch.uint64,
    #"U32": torch.uint32,
    #"U16": torch.uint16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
def _load_tensor(fp, tensor_name: str, dtype_name: str, data_offsets: Tuple[int, int], shape: List[int], header_size: int) -> torch.Tensor:
    if dtype_name not in SAFETENSORS_DTYPE_MAP:
        raise Exception(f"Tensor {tensor_name}: unknown data type {dtype_name}")
    dtype = SAFETENSORS_DTYPE_MAP[dtype_name]
    buffer_len = data_offsets[1] - data_offsets[0]
    if not shape:
        return torch.Tensor([], dtype=dtype)
    total_elements = 1
    for i in shape:
        total_elements *= i
    if total_elements * dtype.itemsize != buffer_len:
        raise Exception(f"Tensor {tensor_name}: data offsets can't match shape.")
    fp.seek(header_size + data_offsets[0], os.SEEK_SET)
    buffer = fp.read(buffer_len)
    if len(buffer) != buffer_len:
        raise Exception(f"Tensor {tensor_name}: data is too short in file, is it truncated?")
    tensor = torch.frombuffer(bytearray(buffer), dtype=dtype).reshape(shape)
    return tensor
    
def load_safetensors(path: PathLike):
    with open(path, "rb") as f:
        assert f.seekable(), "File object is not seekable"

        buffer = f.read(8)
        if len(buffer) < 8:
            raise Exception(f"File too short to read header length")
        header_len = struct.unpack("<Q", buffer)[0]
        if header_len <= 0:
            raise Exception(f"Invalid header len {header_len}")
        header_json = f.read(header_len)
        if len(header_json) != header_len:
            raise Exception(f"File too short to read header")
        
        #print(header_json)
        header = json.loads(header_json)
        for key, tensor_info in header.items():
            if key in SAFETENSORS_NOT_TENSOR_KEYS:
                continue
            if 'dtype' not in tensor_info \
                or 'data_offsets' not in tensor_info \
                or 'shape' not in tensor_info:
                raise Exception(f"Invalid tensor {key}, not dtype/data_offsets/shape all present: {tensor_info}")
            tensor = _load_tensor(f, key, tensor_info['dtype'], tensor_info['data_offsets'], tensor_info['shape'], 8 + header_len)
            tensor_info['tensor'] = tensor
        return header
    
def tensors_compare():
    import safetensors
    import pathlib
    #import pdb; pdb.set_trace()
    path = pathlib.Path('./models/chatgpt2-small/model.safetensors')
    mydata = load_safetensors(path)
    passed = True
    with safetensors.safe_open(path, framework="pt") as f:
        for k in f.keys():
            my = mydata[k]['tensor']
            other = f.get_tensor(k)
            diff = my - other
            sum_diff = torch.sum(diff ** 2).item()
            avergage_diff =  sum_diff / diff.numel()
            this_passed = (sum_diff < 1e-3) and (avergage_diff < 1e-9)
            print(f"comparing {k}: total diff={sum_diff}, average diff={avergage_diff}, passed={this_passed}")
            passed = passed and this_passed
    print("PASSED" if passed else "FAILED")
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(tensors_compare())