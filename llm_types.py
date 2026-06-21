import abc
import torch
import transformers
from dataclasses import dataclass
from typing import Tuple, Optional

class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.LongTensor:
        '''
        Do a sample based unnormalized logits
        
        Input shape: [batch_size, vocab_size]
        Output shape: [batch_size]
        '''
        pass

@dataclass
class KVCacheEntry:
    # KVCacheEntry is in batch granulity, not request granulity
    # q shape: [B, H, T, d_k]
    # shape: [L, B, H, W, d_k]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cached_len: int = 0

class Model(abc.ABC):
    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor, kvcache_entry: Optional[KVCacheEntry], use_cache: bool = True) -> torch.Tensor:
        '''
        a LLM model.

        input_ids shape: [batch_size, seq_len]
        Output shape: [batch_size, vocab_size]
        '''
        pass

    @abc.abstractmethod
    def allocate_kvcache_for_batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        '''
        allocate a kvcache tensor for batch.
        dim0's size must be 2, [0] will be k-cache, [1] will be v-cache
        '''
        pass

type Tokenizer = "transformers.PreTrainedTokenizerBase"
