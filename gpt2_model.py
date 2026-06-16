
import math
import torch
import torch.nn.functional as F
import contextlib
import threading

def gelu_new(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

def layer_norm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    return F.layer_norm(input, normalized_shape=(input.shape[-1],), weight=weight, bias=bias, eps=eps)
    # var, mean = torch.var_mean(input, dim=-1, keepdim=True, correction=0)
    # d = (input - mean)
    # n = torch.rsqrt(var + eps)
    # return d * n * weight + bias

class Embed():
    def __init__(self, d_model, vocab_size):
        self.device = None
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.wte_weight = state_dict.pop('wte.weight')
        self.wpe_weight = state_dict.pop('wpe.weight')

    def to(self, device: torch.device):
        self.device = device
        self.wte_weight = self.wte_weight.to(device)
        self.wpe_weight = self.wpe_weight.to(device)

    def forward(self, input_ids, offset=0):
        # x: [batch, seq]
        seq_len = input_ids.shape[-1]
        x = self.wte_weight[input_ids]
        x = x + self.wpe_weight[offset : offset + seq_len, :]
        return x
    
class LMHead():
    def __init__(self, d_model, vocab_size):
        self.device = None
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.weight = state_dict.pop('wte.weight').T

    def to(self, device: torch.device):
        self.weight = self.weight.to(device)
    
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        return x

class DecoderLayer():
    def __init__(self, layer_id, d_model, h, max_seq_len, layernorm_eps: 1e-05):
        self.device = None
        self.layer_id = layer_id
        self.d_model = d_model
        self.h = h
        self.layernorm_eps = layernorm_eps
        self.max_seq_len = max_seq_len

    def load_state_dict(self, state_dict):
        # layer norm
        self.ln_1_weight = state_dict.pop('ln_1.weight')
        self.ln_1_bias = state_dict.pop('ln_1.bias')
        self.ln_2_weight = state_dict.pop('ln_2.weight')
        self.ln_2_bias = state_dict.pop('ln_2.bias')
        # atteion 
        self.attn_weight = state_dict.pop('attn.c_attn.weight')
        self.attn_bias = state_dict.pop('attn.c_attn.bias')
        self.attn_proj_weight = state_dict.pop('attn.c_proj.weight')
        self.attn_proj_bias = state_dict.pop('attn.c_proj.bias')
        # MLP
        self.c_fc_weight = state_dict.pop('mlp.c_fc.weight')
        self.c_fc_bias = state_dict.pop('mlp.c_fc.bias')
        self.c_proj_weight = state_dict.pop('mlp.c_proj.weight')
        self.c_proj_bias = state_dict.pop('mlp.c_proj.bias')
        # 剩下一个 attn.bias 没啥用
        attn_mask_bias = state_dict.pop('attn.bias')
    
    def to(self, device: torch.device):
        self.device = device
        self.ln_1_weight = self.ln_1_weight.to(device)
        self.ln_1_bias = self.ln_1_bias.to(device)
        self.ln_2_weight = self.ln_2_weight.to(device)
        self.ln_2_bias = self.ln_2_bias.to(device)
        self.attn_weight = self.attn_weight.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.attn_proj_weight = self.attn_proj_weight.to(device)
        self.attn_proj_bias = self.attn_proj_bias.to(device)
        self.c_fc_weight = self.c_fc_weight.to(device)
        self.c_fc_bias = self.c_fc_bias.to(device)
        self.c_proj_weight = self.c_proj_weight.to(device)
        self.c_proj_bias = self.c_proj_bias.to(device)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        x = layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, eps=self.layernorm_eps)
        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias

        scale = torch.rsqrt(torch.tensor([self.d_model / self.h], dtype=x.dtype))
        attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)
        attn_bias = torch.zeros(seq_len, seq_len, dtype=x.dtype)
        attn_bias = attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        # split q, k, v
        q, k, v = qkv_merged.split(self.d_model, dim=-1)
        q = q.reshape((-1, seq_len, self.h, self.d_model // self.h))
        q = q.transpose(-2, -3)
        k = k.reshape((-1, seq_len, self.h, self.d_model // self.h))
        k = k.transpose(-2, -3)
        v = v.reshape((-1, seq_len, self.h, self.d_model // self.h))
        v = v.transpose(-2, -3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal mask
        scores += attn_bias
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, v)
        # merges back
        x = scores.transpose(-2, -3)
        x = x.reshape((-1, seq_len, self.d_model))

        x = torch.matmul(x, self.attn_proj_weight) + self.attn_proj_bias
        return x

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        # MLP
        x = layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias, eps=self.layernorm_eps)

        x = torch.matmul(x, self.c_fc_weight) + self.c_fc_bias
        x = gelu_new(x)
        x = torch.matmul(x, self.c_proj_weight) + self.c_proj_bias
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        a = self.attention(x)
        x = a + residual
        residual = x
        m = self.mlp(x)
        x = m + residual
        return x

class Chatgpt2Model():
    def __init__(self, config):
        self.device = None
        self.N = config['n_layer']           
        self.d_model = config['n_embd'] 
        self.h = config['n_head']             
        self.vocab_size = config['vocab_size']
        self.eos_token_id = config['eos_token_id']
        self.context_window = config['n_ctx']
        self.layernorm_eps = config.get('layer_norm_epsilon', 1e-05)

        self.decoders = []
        for layer_id in range(self.N):
            self.decoders.append(
                DecoderLayer(layer_id, self.d_model, self.h, self.context_window, layernorm_eps=self.layernorm_eps)
            )
        self.lm_head = LMHead(self.d_model, self.vocab_size)
        self.embed = Embed(self.d_model, self.vocab_size)

    def load_state_dict(self, state_dict):
        metadata = state_dict.pop('__metadata__', {})
        for layer_no in range(self.N):
            prefix = f"h.{layer_no}."
            sub_state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items() if key.startswith(prefix)
            }
            for key in sub_state_dict.keys():
                state_dict.pop(prefix + key)
            self.decoders[layer_no].load_state_dict(sub_state_dict)
            if sub_state_dict:
                raise Exception(f"Unknown parameters for decoder layer: {sub_state_dict.keys()}")
        
        wte_weight = state_dict.pop('wte.weight')
        wpe_weight = state_dict.pop('wpe.weight')
        self.embed.load_state_dict({'wpe.weight': wpe_weight, 'wte.weight': wte_weight})
        self.lm_head.load_state_dict({'wte.weight': wte_weight})

        self.ln_f_weight = state_dict.pop('ln_f.weight')
        self.ln_f_bias = state_dict.pop('ln_f.bias')

        if state_dict:
            raise Exception(f"Unknown parameters: {state_dict.keys()}")

    def to(self, device: torch.device):
        self.device = device
        self.ln_f_weight = self.ln_f_weight.to(device)
        self.ln_f_bias = self.ln_f_bias.to(device)
        self.embed.to(device)
        self.lm_head.to(device)
        for decoder in self.decoders:
            decoder.to(device)
        
    def forward(self, input_ids: torch.Tensor, batch_id: None):
        # input_ids: [batch, seq]
        x = self.embed.forward(input_ids)
        # x: [batch, seq, d_model]
        for decoder in self.decoders:
            x = decoder.forward(x)
        # x: [batch, seq, d_model]
        x = layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, eps=self.layernorm_eps)
        # the next-token prediction only needs the last item in each sample
        x = x[:, -1, :]
        # x: [batch, d_model]
        logits: torch.Tensor = self.lm_head.forward(x)
        # logits: [batch, vocab_size]
        return logits

