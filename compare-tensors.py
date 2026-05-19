import os, sys
import click
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
@click.option("--impl", type=click.Choice(["safetensors", "my", "all"], case_sensitive=True), required=True, default="all")
@click.option("--data", "show_data", is_flag=True, default=False, help="是否展示数据")
@click.argument("tensor1_path", type=str)
@click.argument("tensor2_path", type=str)
def tensor_compare(impl: str, tensor1_path: str, tensor2_path: str, show_data: bool):
    bad_cases_first = 0
    bad_cases_second = 0

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

if __name__ == "__main__":
    sys.exit(cli())
