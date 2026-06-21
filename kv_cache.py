import torch
from llm_types import KVCacheEntry, Model

def allocate(model: Model, input_ids: torch.Tensor, output_tokens: int) -> KVCacheEntry:
    B, T = input_ids.shape
    cache = model.allocate_kvcache_for_batch(B, T + output_tokens)
    return KVCacheEntry(
        cached_len = 0,
        k_cache = cache[0],
        v_cache = cache[1],
    )
