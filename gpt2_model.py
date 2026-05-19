
import math
import torch
import torch.nn.functional as F

import utils as debugging

from typing import Tuple

def gelu_new(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

def layer_norm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    return F.layer_norm(input, normalized_shape=(input.shape[-1],), weight=weight, bias=bias, eps=eps)
    var, mean = torch.var_mean(input, dim=-1, keepdim=True, correction=0)
    d = (input - mean)
    n = torch.rsqrt(var + eps)
    return d * n * weight + bias

class Embed():
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.wte_weight = state_dict.pop('wte.weight')
        self.wpe_weight = state_dict.pop('wpe.weight')

    def forward(self, input_ids):
        # x: [batch, seq]
        seq_len = input_ids.shape[-1]
        # advanced index
        x = self.wte_weight[input_ids]
        debugging.save_tensors(f"trace/my/embed.safetensors", embed=x)
        debugging.save_tensors(f"trace/my/pos.safetensors", pos=self.wpe_weight[:seq_len, :])
        x = x + self.wpe_weight[:seq_len, :]
        return x
    
class LMHead():
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.weight = state_dict.pop('wte.weight').T

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        #x = torch.softmax(x, dim=-1)
        return x

class DecoderLayer():
    def __init__(self, layer_id, d_model, h, max_seq_len, layernorm_eps: 1e-05):
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

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        x = layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, eps=self.layernorm_eps)
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.ln1.safetensors", x=x)

        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.qkv.safetensors", x=qkv_merged)
        qkv_splited = torch.split(qkv_merged, self.d_model // self.h, dim=-1)
        assert len(qkv_splited) == 3 * self.h

        q_splited = qkv_splited[ : self.h]
        k_splited = qkv_splited[self.h : 2 * self.h]
        v_splited = qkv_splited[2 * self.h : ]

        scale = torch.rsqrt(torch.tensor([self.d_model / self.h], dtype=x.dtype))

        attn_bias = torch.zeros(self.max_seq_len, self.max_seq_len, dtype=x.dtype)
        attn_mask = torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool).tril(diagonal=0)
        attn_bias = attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        # 这里的优化空间需要分析，for里面的内容尽量少
        heads = []
        for i in range(self.h):
            qk_similarities = torch.matmul(q_splited[i], k_splited[i].transpose(-2, -1)) * scale
            qk_similarities += attn_bias[:seq_len, :seq_len]
            qk_similarities = F.softmax(qk_similarities, dim=-1)
            headi = torch.matmul(qk_similarities, v_splited[i])
            heads.append(headi)
        x = torch.concat(heads, dim=-1)
        x = torch.matmul(x, self.attn_proj_weight) + self.attn_proj_bias
        return x
    
    def attention_optimized(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        x = layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, eps=self.layernorm_eps)
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.ln1.safetensors", x=x)

        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.qkv.safetensors", x=qkv_merged)

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
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.input.safetensors", x=x)

        residual = x
        a = self.attention_optimized(x)
        x = a + residual

        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.after-attention.safetensors", x=x)

        residual = x
        m = self.mlp(x)
        x = m + residual

        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.output.safetensors", x=x)
        return x

class Chatgpt2Model():
    def __init__(self, config):
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
                key[len(prefix):]: value['tensor']
                for key, value in state_dict.items() if key.startswith(prefix)
            }
            for key in sub_state_dict.keys():
                state_dict.pop(prefix + key)
            self.decoders[layer_no].load_state_dict(sub_state_dict)
            if sub_state_dict:
                raise Exception(f"Unknown parameters for decoder layer: {sub_state_dict.keys()}")
        
        wte_weight = state_dict.pop('wte.weight')['tensor']
        wpe_weight = state_dict.pop('wpe.weight')['tensor']
        self.embed.load_state_dict({'wpe.weight': wpe_weight.clone().detach(), 'wte.weight': wte_weight})
        self.lm_head.load_state_dict({'wte.weight': wte_weight.clone().detach()})

        self.ln_f_weight = state_dict.pop('ln_f.weight')['tensor']
        self.ln_f_bias = state_dict.pop('ln_f.bias')['tensor']

        if state_dict:
            raise Exception(f"Unknown parameters: {state_dict.keys()}")
        
    def forward(self, input_ids: torch.Tensor):
        # input_ids: [batch, seq]
        debugging.save_tensors(f"trace/my/input_ids.safetensors", input_ids=input_ids)
        x = self.embed.forward(input_ids)
        debugging.save_tensors(f"trace/my/pos_embed.safetensors", pos_embed=x)
        # x: [batch, seq, d_model]
        for decoder in self.decoders:
            x = decoder.forward(x)
        # x: [batch, seq, d_model]
        x = layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias, eps=self.layernorm_eps)
        # 提取最后一个维度的向量
        x = x[:, -1, :]
        # x: [batch, d_model]
        logits: torch.Tensor = self.lm_head.forward(x)
        # logits: [batch, vocab_size]
        return logits
