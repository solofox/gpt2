import click
import torch
import pathlib

import llm_loader

class GenericSampler():
    def __init__(self, temperature, top_p, top_k):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    def sample(self, logits):
        return torch.argmax(logits, dim=1)
    
def ask_llm(model_instance, tokenizer, sampler, prompt, ouput_tokens: int = 0):
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
    while True:
        logits = model_instance.forward(input_ids)
        next_tokens_id = sampler.sample(logits)
        next_tokens = tokenizer.convert_ids_to_tokens(next_tokens_id)[0]        
        if next_tokens_id[0] == model_instance.eos_token_id:
            print("eos")
            break
        response += next_tokens[0]
        ouput_tokens -= 1
        if ouput_tokens <= 0:
            print("max reached")
            break
        next_tokens_id = next_tokens_id.unsqueeze(dim=1)
        input_ids = torch.concat([input_ids, next_tokens_id], dim=1)
    return response

@click.command
@click.argument("model_path")
@click.argument("prompt")
def run(model_path: str, prompt: str):
    path = pathlib.Path(model_path)
    model_instance, tokenizer = llm_loader.load(path)
    sampler = GenericSampler(0.6, 10, 3)
    response = ask_llm(model_instance, tokenizer, sampler, prompt, ouput_tokens=10)
    print(response)

if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    run()


