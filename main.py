import click
import torch
import pathlib

import llm_loader

class GenericSampler():
    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    def sample(self, logits, temperature=0.7):
#        return torch.argmax(logits, dim=1, keepdim=True)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
def ask_llm(model_instance, tokenizer, sampler, prompt, ouput_tokens: int = 0, temperature = 0.7):
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > model_instance.context_window:
        print(f"Input is too long, max context window is {model_instance.context_window}")
        return
    max_output_tokens = model_instance.context_window - len(input_ids)
    if ouput_tokens <= 0:
        ouput_tokens = max_output_tokens
    elif ouput_tokens > max_output_tokens:
        ouput_tokens = max_output_tokens
    response = ''
    input_ids = torch.tensor([input_ids], dtype=torch.int)
    print(prompt, end='', flush=True)
    while True:
        logits = model_instance.forward(input_ids)
        next_tokens_id = sampler.sample(logits, temperature=temperature)
        next_tokens = tokenizer.batch_decode(next_tokens_id)
        if next_tokens_id[0].item() == model_instance.eos_token_id:
            print("eos")
            break
        print(next_tokens[0], end='', flush=True)
        response += next_tokens[0]
        ouput_tokens -= 1
        if ouput_tokens <= 0:
            print("max reached")
            break
        input_ids = torch.concat([input_ids, next_tokens_id], dim=1)
    return response

@click.command
@click.argument("model_path")
@click.argument("prompt")
def run(model_path: str, prompt: str):
    path = pathlib.Path(model_path)
    model_instance, tokenizer = llm_loader.load(path)
    sampler = GenericSampler(0.6, 10, 3)
    response = ask_llm(model_instance, tokenizer, sampler, prompt, ouput_tokens=1024)
    print(response)

if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    run()


