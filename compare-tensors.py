import os, sys
import click
import math
import torch
import safetensors
import utils
import pathlib
import contextlib 

@contextlib.contextmanager
def pretty_print_tensor(precision=4, linewidth=2048):
    """
    临时修改打印配置，完整且美观地展示 PyTorch 张量数据
    precision: 保留的小数位数
    linewidth: 每行显示的字符宽度
    """
    # 获取 PyTorch 当前的打印配置
    try:
        # threshold=float('inf') 表示不截断，显示所有元素
        # profile="full" 也是一个很好的快捷方式，等同于把阈值设为无穷大
        torch.set_printoptions(threshold=float('inf'), linewidth=linewidth, precision=precision)
        yield
    finally:
        # 退出代码块后，自动恢复原来的打印配置
        torch.set_printoptions(profile='default')

def load_using_safetensors(path):
    result = {}
    with safetensors.safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            result[k] = {
                'tensor': tensor
            }
    return result

def do_compare(title: str, tensor1_dict: dict, tensor2_dict: dict, show_data: bool = False):
    print(f"Comparing tensors {title}")
    bad_cases = 0
    common_keys = set(tensor1_dict.keys()).intersection(set(tensor2_dict.keys()))
    tensor1_ukeys = set(tensor1_dict.keys()).difference(common_keys)
    tensor2_ukeys = set(tensor2_dict.keys()).difference(common_keys)
    for key in common_keys:
        tensor1 = tensor1_dict[key]['tensor']
        tensor2 = tensor2_dict[key]['tensor']
        is_good = True
        if show_data:
            with pretty_print_tensor():
                print("tensor1: ", tensor1)
                print("tensor2: ", tensor2)

        def squeeze_high(shape: list):
            while len(shape) > 1 and shape[0] == 1:
                shape.pop(0)
        shape1 = squeeze_high(list(tensor1.shape))
        shape2 = squeeze_high(list(tensor2.shape))
        if shape1 != shape2:
            print(f"  Tensor {key}: inconsistent shape, {tensor1.shape} vs {tensor2.shape}")
            is_good = False
        elif tensor1.dtype != tensor2.dtype:
            print(f"  Tensor {key}: inconsistent type, {tensor1.dtype} vs {tensor2.dtype}")
            is_good = False
        else:
            is_good = torch.allclose(tensor1, tensor2, rtol=1e-4, atol=1e-5)
            diff = tensor1 - tensor2
            distance = torch.sqrt(torch.sum(diff ** 2)).item()
            normalized_distance =  distance / diff.numel()
            print(f"  Tensor {key}: distance={distance}, normalized distance={normalized_distance}, tensor.allclose={is_good}. shape={tensor1.shape}, dtype={tensor1.dtype}")
        if not is_good:
            bad_cases += 1
    if tensor1_ukeys:
        print(f"  Tensors in tensors set 1 ONLY: {tensor1_ukeys}")
        bad_cases += len(tensor1_ukeys)
    if tensor2_ukeys:
        print(f"  Tensors in tensors set 2 ONLY: {tensor2_ukeys}")
        bad_cases += len(tensor2_ukeys)
    return bad_cases

def do_compare_cache(title: str, tensor1_dict: dict, tensor2_dict: dict, is_multihead: bool = False, is_seq2seq: bool = False):
    print(f"Comparing kvcache tensors {title}, is_multihead={is_multihead}, is_seq2seq={is_seq2seq}")
    
    tensor1 = tensor1_dict['x']['tensor']
    tensor2 = tensor2_dict['x']['tensor']
    seq_len1 = tensor1.shape[-2]        
    seq_len2 = tensor2.shape[-2]
    shorter_len = min(seq_len1, seq_len2)

    def is_upper_all_zeros(t):
        lower_mask = torch.ones((t.shape[-2:]), dtype=torch.bool).tril()
        nt = t.masked_fill(lower_mask, 0)
        return nt.allclose( torch.tensor(0, dtype=t.dtype) )

    if is_seq2seq:
        dim = -1
        if not is_upper_all_zeros(tensor1):
            print(f"tensor1's upper is not all close to zeros.")
            return 1
        if not is_upper_all_zeros(tensor2):
            print(f"tensor2's upper is not all close to zeros.")
            return 1
        # dim[-1]和dim[-2]都是seq_len，先去掉最后一个维度的突出部分
        tensor1p1, tensor1p2 = tensor1.split((shorter_len, tensor1.shape[dim] - shorter_len), dim=dim)
        tensor2p1, tensor2p2 = tensor2.split((shorter_len, tensor2.shape[dim] - shorter_len), dim=dim)
        tensor1 = tensor1p1
        tensor2 = tensor2p1

    dim = -2

    otherdims_shape1 = list(tensor1.shape)
    otherdims_shape2 = list(tensor2.shape)
    otherdims_shape1[dim] = 0
    otherdims_shape2[dim] = 0
    if otherdims_shape1 != otherdims_shape2:
        print(f"tensor1's shape {tensor1.shape}, tensor2's shape {tensor2.shape}, they are not matched at dim={dim}")
        return 1
    
    #tensor1 = tensor1[:, :, -1:, :]
    tensor1p1, tensor1p2 = tensor1.split((shorter_len, tensor1.shape[dim] - shorter_len), dim=dim)
    tensor2p1, tensor2p2 = tensor2.split((shorter_len, tensor2.shape[dim] - shorter_len), dim=dim)

    print(tensor1p1.view(-1))
    print(tensor2p1.view(-1))

    is_good = torch.allclose(tensor1p1, tensor2p1, rtol=1e-4, atol=1e-5)
    diff = tensor1p1 - tensor2p1
    distance = torch.sqrt(torch.sum(diff ** 2)).item()
    normalized_distance =  distance / diff.numel()
    print(f"  Tensor common: distance={distance}, normalized distance={normalized_distance}, tensor.allclose={is_good}. shape={tensor1p1.shape}, dtype={tensor1p1.dtype}")
    if tensor1p2.numel() > 0:
        print(f"  Remains1: {tensor1p2.shape} at dim={dim}")
    if tensor1p2.numel() > 0:
        print(f"  Remains2: {tensor2p2.shape} at dim={dim}")
    return 0 if is_good else 1

@click.group
def cli():
    pass

@cli.command(name="impl")
@click.argument("path", type=str)
def impl_compare(path: str):
    path = pathlib.Path(path)
    tensor1_dict = utils.load_safetensors(path)
    tensor1_dict.pop('__metadata__', None)
    tensor2_dict = load_using_safetensors(path)
    bad_cases = do_compare("by different implementation, first=my, second=safetensors", tensor1_dict, tensor2_dict)
    print("PASSED" if bad_cases == 0 else "FAILED")
    return bad_cases

@cli.command(name="tensor")
@click.option("--impl", type=click.Choice(["safetensors", "my", "all"], case_sensitive=True), required=True, default="my")
@click.option("--data", "show_data", is_flag=True, default=False, help="是否展示数据")
@click.argument("tensor1_path", type=str)
@click.argument("tensor2_path", type=str)
def tensor_compare(impl: str, tensor1_path: str, tensor2_path: str, show_data: bool):
    bad_cases_first = 0
    bad_cases_second = 0

    tensor1_path = pathlib.Path(tensor1_path)
    tensor2_path = pathlib.Path(tensor2_path)

    if impl in ('all', "safetensors"):
        tensor1_dict = load_using_safetensors(tensor1_path)
        tensor2_dict = load_using_safetensors(tensor2_path)
        bad_cases_first = do_compare(f"loaded by safetensors, first={tensor1_path}, second={tensor2_path}", tensor1_dict, tensor2_dict, show_data=show_data)
        print("PASSED" if bad_cases_first == 0 else "FAILED")
    
    if impl in ('all', "my"):
        tensor1_dict = utils.load_safetensors(tensor1_path)
        tensor1_dict.pop('__metadata__', None)
        tensor2_dict = utils.load_safetensors(tensor2_path)
        tensor2_dict.pop('__metadata__', None)
        bad_cases_second = do_compare(f"loaded by my, first={tensor1_path}, second={tensor2_path}", tensor1_dict, tensor2_dict, show_data=show_data)
        print("PASSED" if bad_cases_second == 0 else "FAILED")
    
    return bad_cases_first + bad_cases_second

@cli.command(name="cache")
@click.option("--impl", type=click.Choice(["safetensors", "my", "all"], case_sensitive=True), required=True, default="my")
@click.argument("tensor1_path", type=str)
@click.argument("tensor2_path", type=str)
def cache_compare(impl: str, tensor1_path: str, tensor2_path: str):
    bad_cases_first = 0
    bad_cases_second = 0

    basename = os.path.basename(tensor1_path)
    tensor1_path = pathlib.Path(tensor1_path)
    tensor2_path = pathlib.Path(tensor2_path)

    MULTIHEAD_FILES = ['.q.safetensors', '.k.safetensors', '.v.safetensors', '.softmax.safetensors', '.ah.safetensors']
    is_multihead = any([ basename.endswith(postfix) for postfix in MULTIHEAD_FILES ])
    is_seq2seq = basename.endswith(".self.safetensors")
    
    if impl in ('all', "safetensors"):
        tensor1_dict = load_using_safetensors(tensor1_path)
        tensor2_dict = load_using_safetensors(tensor2_path)
        bad_cases_first = do_compare_cache(f"loaded by safetensors, first={tensor1_path}, second={tensor2_path}", tensor1_dict, tensor2_dict, is_multihead=is_multihead, is_seq2seq=is_seq2seq)
        print("PASSED" if bad_cases_first == 0 else "FAILED")
    
    if impl in ('all', "my"):
        tensor1_dict = utils.load_safetensors(tensor1_path)
        tensor1_dict.pop('__metadata__', None)
        tensor2_dict = utils.load_safetensors(tensor2_path)
        tensor2_dict.pop('__metadata__', None)
        bad_cases_second = do_compare_cache(f"loaded by my, first={tensor1_path}, second={tensor2_path}", tensor1_dict, tensor2_dict, is_multihead=is_multihead, is_seq2seq=is_seq2seq)
        print("PASSED" if bad_cases_second == 0 else "FAILED")
    
    return bad_cases_first + bad_cases_second

if __name__ == "__main__":
    sys.exit(cli())
