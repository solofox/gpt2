import os, sys
from os import PathLike
import json
import logging
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
    
def load_safetensors(path: PathLike, keep_meta=False):
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
        
        header = json.loads(header_json)
        tensors = {}
        for key, tensor_info in header.items():
            if key in SAFETENSORS_NOT_TENSOR_KEYS:
                continue
            if 'dtype' not in tensor_info \
                or 'data_offsets' not in tensor_info \
                or 'shape' not in tensor_info:
                raise Exception(f"Invalid tensor {key}, not dtype/data_offsets/shape all present: {tensor_info}")
            tensor = _load_tensor(f, key, tensor_info['dtype'], tensor_info['data_offsets'], tensor_info['shape'], 8 + header_len)
            if keep_meta:
                tensor_info['tensor'] = tensor
                tensors[key] = tensor_info
            else:
                tensors[key] = tensor
        return tensors

def save_tensors(path: PathLike, metadata=None, **tensors_dict):
    from safetensors.torch import save_file  
    return save_file(tensors_dict, path, metadata)

def setup_logging(level="INFO"):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logger = logging.getLogger()
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.setLevel(numeric_level)