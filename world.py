import os
import torch
import torch.distributed as dist
import logging
from pathlib import Path
from typing import Literal, Final

WORLD_SIZE: Final[int] = int(os.getenv("LOCAL_WORLD_SIZE")) if os.getenv("LOCAL_WORLD_SIZE") else 1
RANK: Final[int] = int(os.getenv("LOCAL_RANK")) if os.getenv("LOCAL_RANK") else 0
assert WORLD_SIZE >= 1
assert RANK >= 0 and RANK < WORLD_SIZE

def select_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "cuda":
        num_gpus = torch.cuda.device_count()
        assert RANK < num_gpus, f"Not GPU available for rank {RANK}"
        return torch.device(f"cuda:{RANK}")
    else:
        device_names = device_name.split(',')
        return torch.device(device_names[RANK])

def initialize(device, backend: Literal["gloo", "nccl", "auto"] = "auto"):
    if not dist.is_available():
        assert WORLD_SIZE == 1, "torch.distributed is not available while do tensor parallel"
        return
    if backend == "auto":
        if device.type == 'cuda':
            backend = 'nccl'
        else:
            backend = 'gloo'
    logging.info(f"init distributed using backend={backend}, rank={RANK}, world_size={WORLD_SIZE}")
    dist.init_process_group(backend=backend)

def tensor_split(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Split the tensor along the specified dim.
    '''
    if WORLD_SIZE == 1:
        return tensor
    assert tensor.size(dim) % WORLD_SIZE == 0
    my_part = tensor.split(tensor.size(dim) // WORLD_SIZE, dim=dim)[RANK]
    return my_part

def all_gather(tensor: torch.Tensor, dim=-1) -> torch.Tensor:
    if WORLD_SIZE == 1:
        return tensor
    tensor_list = [ torch.empty_like(tensor) for _ in range(WORLD_SIZE) ]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=dim)

def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    '''
    Reduce the tensor inplace.
    '''
    if WORLD_SIZE == 1:
        return
    dist.all_reduce(tensor, op)

def broadcast(tensor: torch.Tensor, src=0):
    if WORLD_SIZE == 1:
        return 
    dist.broadcast(tensor, src=src)
