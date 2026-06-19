import torch
from llm_types import Sampler

DEFAULT_TEMPERATURE = 0.7

class GreedySampler(Sampler):
    def __init__(self):
        pass

    def sample(self, logits: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
class NucleusSampler(Sampler):
    def __init__(self, temperature = DEFAULT_TEMPERATURE, top_k: int = 0, top_p: float = 1.0):
        if top_k < 0:
            raise ValueError(f"top_k value {top_k} is not valid")
        if top_p > 1.0 or top_p < 0.0:
            raise ValueError(f"top_p value {top_p} is not valid")

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def sample(self, logits: torch.Tensor) -> torch.LongTensor:
        # logits: [B, N], B = batch, N = token numbers
        B, N = logits.shape

        if self.temperature is not None:
            logits = logits / self.temperature
        
        # it's very confusing when both top_k and top_p is specified. I think option 1 is OK, because top-p is well defined.
        # option 1: top-k change the token probability distribution of top-p.
        # option 2: top-k doesn't change the token probability distribution.
        
        if self.top_k > 0 and self.top_k < N:
            values, indices = torch.topk(logits, k=self.top_k, dim=1)
            min_values = values[..., -1:]
            logits = torch.where(logits < min_values, float('-inf'), logits)

        probs = torch.softmax(logits, dim=1)
        if self.top_p < 1.0:
            sorted_probs, indices = torch.sort(probs, dim=1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            mask = (cumsum - sorted_probs) > self.top_p
            sorted_probs[mask] = 0
            probs.scatter_(dim=1, index=indices, src=sorted_probs)
            probs = probs / torch.sum(sorted_probs, dim=1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)
    