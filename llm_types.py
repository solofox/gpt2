import abc
import torch
import transformers

class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.LongTensor:
        '''
        Do a sample based unnormalized logits
        
        Input shape: [batch_size, vocab_size]
        Output shape: [batch_size]
        '''
        pass

class Model(abc.ABC):
    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor, batch_id: int) -> torch.Tensor:
        '''
        a LLM model.

        input_ids shape: [batch_size, seq_len]
        Output shape: [batch_size, vocab_size]
        '''
        pass

type Tokenizer = "transformers.PreTrainedTokenizerBase"