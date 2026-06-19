import sys
import click
import torch
import threading
import pathlib
import logging
import time
from transformers import TextIteratorStreamer
import utils

import samplers
import llm_loader

in_debug_mode = False
    
def generate(model_instance, tokenizer, sampler, prompt, output_tokens, streaming=True, truncate_to=-1):
    input_ids_cpu = tokenizer.encode(prompt)
    
    if truncate_to >= 0 and len(input_ids_cpu) > truncate_to:
        prompt_tokens = len(input_ids_cpu)
        input_ids_cpu = input_ids_cpu[:truncate_to]
        prompt = tokenizer.batch_decode(input_ids_cpu)
        logging.debug(f"Truncate prompt from {prompt_tokens} tokens => {len(input_ids_cpu)} tokens")
    else:
        logging.debug(f"Prompt has {len(input_ids_cpu)} token")

    if len(input_ids_cpu) > model_instance.context_window:
        logging.error(f"Input is too long, it has {len(input_ids_cpu)} tokens but max context window is {model_instance.context_window}")
        return 0
    
    max_output_tokens = model_instance.context_window - len(input_ids_cpu)
    if output_tokens <= 0:
        output_tokens = max_output_tokens
    elif output_tokens > max_output_tokens:
        output_tokens = max_output_tokens
    input_ids = torch.tensor([input_ids_cpu], dtype=torch.int).to(model_instance.device)
    
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_special_tokens=True, 
        timeout=None  # blocking
    )

    token_generation_time = []
    generated_tokens = 0
    stopped = False
    def _generate(input_ids, output_tokens):
        nonlocal token_generation_time
        nonlocal generated_tokens
        while True:
            begin_time = time.time()
            logits = model_instance.forward(input_ids, batch_id=1)
            next_token_ids = sampler.sample(logits)
            #next_token_ids = torch.tensor([[765]])
            end_time = time.time()
            token_generation_time.append(end_time - begin_time)

            next_token_ids_cpu = next_token_ids.to("cpu")
            streamer.put(next_token_ids_cpu[0])
            if next_token_ids_cpu[0].item() == model_instance.eos_token_id:
                break
            generated_tokens += 1
            output_tokens -= 1
            if output_tokens <= 0:
                break
            input_ids = torch.concat([input_ids, next_token_ids], dim=1)
            if stopped:
                break
        streamer.end()

    gworker = threading.Thread(target=_generate, args=(input_ids, output_tokens), daemon=True)
    gworker.start()
    try:
        for newtext in streamer:
            yield newtext
    except KeyboardInterrupt as e:
        logging.debug("interrupted by Ctrl-C")
    finally:
        stopped = True
        gworker.join()
    logging.debug("Per-Token generation time: %s", token_generation_time)
    return generated_tokens

@click.command
@click.argument("model_path")
@click.argument("prompt")
@click.option("--device", "-d", help="computation device, default is auto (detect cuda -> cpu)", default="auto")
@click.option("--output_tokens", "-m", type=int, help="max output length, default is 0 (until maximum context window size)", default=0)
@click.option("--temperature", "-t", type=float, help=f"temperature, default is {samplers.DEFAULT_TEMPERATURE}", default=samplers.DEFAULT_TEMPERATURE)
@click.option("--topk", "-k", type=int, help="topk, default is ∞ (all tokens)", default=0)
@click.option("--topp", "-p", type=float, help="topp, default is 1.0 (all tokens)", default=1.0)
@click.option("--truncate-to", type=int, help="truncate the prompt to this length, default is -1 (no truncation)", default=-1)
@click.option("--debug", is_flag=True, default=False)
def run(model_path: str, prompt: str, output_tokens: int, temperature: float, topk: int, topp: float, device: str, truncate_to: int, debug: bool):
    global is_debug_mode
    is_debug_mode = debug
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
    device = llm_loader.select_device(device_name)
    logging.info(f"Using device: {device}")

    path = pathlib.Path(model_path)
    model_instance, tokenizer = llm_loader.load(path, device)
    sampler = samplers.NucleusSampler(temperature, topk, topp)

    generated_tokens = 0
    #print(prompt, end='', flush=True)
    begin_time = time.time()
    output_stream = generate(model_instance, tokenizer, sampler, prompt, output_tokens=output_tokens, truncate_to=truncate_to)
    try:
        while True:
            newtext = next(output_stream)
            print(newtext, end='', flush=True)
    except StopIteration as e:
        generated_tokens = e.value or 0
    print()
    end_time = time.time()
    logging.info(f"generated_tokens={generated_tokens}, used time={end_time - begin_time}, TPS={generated_tokens / (end_time - begin_time)}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    run()


