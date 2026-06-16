import click
import torch
import threading
import pathlib
from transformers import TextIteratorStreamer

import samplers
import llm_loader
    
def generate(model_instance, tokenizer, sampler, prompt, output_tokens, streaming=True):
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > model_instance.context_window:
        print(f"Input is too long, max context window is {model_instance.context_window}")
        return
    max_output_tokens = model_instance.context_window - len(input_ids)
    if output_tokens <= 0:
        output_tokens = max_output_tokens
    elif output_tokens > max_output_tokens:
        output_tokens = max_output_tokens
    input_ids = torch.tensor([input_ids], dtype=torch.int)

    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_special_tokens=True, 
        timeout=None  # blocking
    )
    generation_stopped = False
    def _generate(input_ids, output_tokens):
        while True:
            logits = model_instance.forward(input_ids, batch_id=1)
            next_tokens_id = sampler.sample(logits)
            streamer.put(next_tokens_id[0])
            if next_tokens_id[0].item() == model_instance.eos_token_id:
                break
            output_tokens -= 1
            if output_tokens <= 0:
                break
            input_ids = torch.concat([input_ids, next_tokens_id], dim=1)
            if generation_stopped:
                break
        streamer.end()

    gworker = threading.Thread(target=_generate, args=(input_ids, output_tokens), daemon=True)
    gworker.start()
    try:
        for newtext in streamer:
            yield newtext
    finally:
        generation_stopped = True
        gworker.join()

@click.command
@click.argument("model_path")
@click.argument("prompt")
@click.option("--output_tokens", "-m", help="max outout length, default is 0 (until maximum context window size)", default=0)
@click.option("--temperature", "-t", help=f"temperature, default is {samplers.DEFAULT_TEMPERATURE}", default=samplers.DEFAULT_TEMPERATURE)
@click.option("--topk", "-k", help="topk, default is ∞ (all tokens)", default=0)
@click.option("--topp", "-p", help="topp, default is 1.0 (all tokens)", default=1.0)
def run(model_path: str, prompt: str, output_tokens: int, temperature: float, topk: int, topp: float):
    path = pathlib.Path(model_path)
    model_instance, tokenizer = llm_loader.load(path)
    sampler = samplers.NucleusSampler(temperature, topk, topp)

    print(prompt, end='', flush=True)
    output_stream = generate(model_instance, tokenizer, sampler, prompt, output_tokens=output_tokens)
    for newtext in output_stream:
        print(newtext, end='', flush=True)
    print()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run()


