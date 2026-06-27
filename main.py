import sys
import os
import torch
import torch.distributed as dist
import click
import pathlib
import logging
import time
import utils

import llm_types
import kv_cache
import samplers
import llm_loader
import world

in_debug_mode = False

def generate(model_instance: llm_types.Model, tokenizer: llm_types.Tokenizer, sampler: llm_types.Sampler, prompt: str, output_tokens: int, truncate_to: int=-1, disable_kvcache: bool=False):
    input_ids_cpu = tokenizer.encode(prompt)
    
    if truncate_to >= 0 and len(input_ids_cpu) > truncate_to:
        prompt_tokens = len(input_ids_cpu)
        input_ids_cpu = input_ids_cpu[:truncate_to]
        prompt = tokenizer.batch_decode(input_ids_cpu)
        logging.debug(f"Truncate prompt from {prompt_tokens} tokens => {len(input_ids_cpu)} tokens")
    else:
        logging.debug(f"Prompt has {len(input_ids_cpu)} token")

    if len(input_ids_cpu) >= model_instance.context_window:
        logging.error(f"Input is too long, it has {len(input_ids_cpu)} tokens but max context window is {model_instance.context_window}")
        return 0, "length"
    
    max_output_tokens = model_instance.context_window - len(input_ids_cpu)
    if output_tokens <= 0:
        output_tokens = max_output_tokens
    elif output_tokens > max_output_tokens:
        output_tokens = max_output_tokens
    input_ids = torch.tensor([input_ids_cpu], dtype=torch.int).to(model_instance.device)

    if not disable_kvcache:
        kvcache_entry = kv_cache.allocate(model_instance, input_ids, output_tokens)
    else:
        kvcache_entry = None

    generated_tokens = 0
    stop_reason = None
    while True:
        logits = model_instance.forward(input_ids, kvcache_entry=kvcache_entry, use_cache=not disable_kvcache)
        if world.RANK == 0:
            next_token_ids = sampler.sample(logits).to(torch.int)
        else:
            next_token_ids = torch.zeros( (1, 1), dtype=torch.int, device=logits.device)
        world.broadcast(next_token_ids, src=0)
        next_token_ids_cpu = next_token_ids.to("cpu")
        if next_token_ids_cpu[0].item() == model_instance.eos_token_id:
            stop_reason = "finish"
            break
        generated_tokens += 1
        newtext = tokenizer.batch_decode(next_token_ids_cpu.tolist())
        yield newtext[0]
        if generated_tokens >= output_tokens:
            stop_reason = "length"
            break
        if disable_kvcache:
            input_ids = torch.concat([input_ids, next_token_ids], dim=1)
        else:
            input_ids = next_token_ids

    return generated_tokens, stop_reason

@click.command
@click.argument("model_path")
@click.argument("prompt")
@click.option("--device", "-d", help="computation device, auto/cpu/cuda, default is auto (prefer cuda > cpu)", default="auto")
@click.option("--backend", help="distributed backend, auto/gloo/nccl, default is auto", default="auto")
@click.option("--output_tokens", "-m", type=int, help="max output length, default is 0 (until maximum context window size)", default=0)
@click.option("--temperature", "-t", type=float, help=f"temperature, default is {samplers.DEFAULT_TEMPERATURE}", default=samplers.DEFAULT_TEMPERATURE)
@click.option("--topk", "-k", type=int, help="topk, default is ∞ (all tokens)", default=0)
@click.option("--topp", "-p", type=float, help="topp, default is 1.0 (all tokens)", default=1.0)
@click.option("--truncate-to", type=int, help="truncate the prompt to this length, default is -1 (no truncation)", default=-1)
@click.option("--disable-kvcache", is_flag=True, help="disable kvcache", default=False)
@click.option("--debug", is_flag=True, default=False)
def run(model_path: str, prompt: str, output_tokens: int, temperature: float, topk: int, topp: float, device: str, backend: str, truncate_to: int, disable_kvcache: bool, debug: bool):
    global in_debug_mode
    in_debug_mode = debug
    utils.setup_logging("INFO" if not debug else "DEBUG")

    if prompt == "-":
        prompt = sys.stdin.read()
    elif prompt.startswith("@@"):
        prompt = prompt[1:]
    elif prompt.startswith("@"):
        filename = prompt[1:]
        with open(filename, "rt") as f:
            prompt = f.read()

    device_name = device
    device = world.select_device(device_name)
    logging.info(f"Using device: {device}")
    world.initialize(device, backend)

    path = pathlib.Path(model_path)
    model_instance, tokenizer = llm_loader.load(path, device)
    sampler = samplers.NucleusSampler(temperature, topk, topp)

    generated_tokens, stop_reason = 0, "unknown"
    begin_time = time.time()
    output_stream = generate(model_instance, tokenizer, sampler, prompt, output_tokens=output_tokens, truncate_to=truncate_to, disable_kvcache=disable_kvcache)
    try:
        while True:
            newtext = next(output_stream)
            if world.RANK == 0:
                print(newtext, end='', flush=True)
    except StopIteration as e:
        if e.value:
            generated_tokens, stop_reason = e.value        
    print()
    end_time = time.time()
    if world.RANK == 0:
        logging.info(f"generated_tokens={generated_tokens}, stop_reason={stop_reason}, used time={end_time - begin_time}, TPS={generated_tokens / (end_time - begin_time)}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run()


