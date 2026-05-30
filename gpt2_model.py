
import math
import torch
import torch.nn.functional as F
import contextlib
import threading

import utils as debugging

from typing import Tuple

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
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.wte_weight = state_dict.pop('wte.weight')
        self.wpe_weight = state_dict.pop('wpe.weight')

    def forward(self, input_ids, offset=0):
        # x: [batch, seq]
        seq_len = input_ids.shape[-1]
        # advanced index
        x = self.wte_weight[input_ids]
        debugging.save_tensors(f"trace/my/embed.safetensors", embed=x)
        debugging.save_tensors(f"trace/my/pos.safetensors", pos=self.wpe_weight[:seq_len, :])
        x = x + self.wpe_weight[offset : offset + seq_len, :]
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

class KVCache(threading.local):
    __slots__ = ['date']
    def __init__(self):
        self.data = None

current_kvcache = KVCache()

@contextlib.contextmanager
def bind_kvcache(data):
    assert isinstance(data, dict), "kvcache data must be a dict"

    try:
        current_kvcache.data = data
        yield
    except:
        raise
    finally:
        current_kvcache.data = None

def save_layer_kvcache(layer_id, k, v):
    data = current_kvcache.data 
    if data is None:
        return
    
    batch_size = k.shape[0]
    seq_len = k.shape[-2]
    if layer_id == 0:
        data['seq_len'] = seq_len
        data['batch_size'] = batch_size
        data.setdefault('kv', {})
    else:
        assert data['seq_len'] == seq_len and data['batch_size'] == batch_size, "inconsistent kv cache"
    data['kv'][layer_id] = { 'k': k, 'v': v }

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
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.ln1.safetensors", x=x)

        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.qkv.safetensors", x=qkv_merged)

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
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.q.safetensors", x=q.contiguous())
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.k.safetensors", x=k.contiguous())
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.v.safetensors", x=v.contiguous())

        save_layer_kvcache(self.layer_id, k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal mask
        scores += attn_bias
        scores = F.softmax(scores, dim=-1)
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.self.safetensors", x=scores.contiguous())
        scores = torch.matmul(scores, v)
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.ah.safetensors", x=scores.contiguous())
        # merges back
        x = scores.transpose(-2, -3)
        x = x.reshape((-1, seq_len, self.d_model))
        debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{seq_len}.a.safetensors", x=x.contiguous())

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

        decoder_class = config.get('decoder_class', DecoderLayer)
        self.decoders = []
        for layer_id in range(self.N):
            self.decoders.append(
                decoder_class(layer_id, self.d_model, self.h, self.context_window, layernorm_eps=self.layernorm_eps)
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
        
    def forward(self, input_ids: torch.Tensor, batch_id: None):
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


class DecoderLayerWithKVCache(DecoderLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def attention_optimized(self, x: torch.Tensor) -> torch.Tensor:        
        if self.layer_id not in current_kvcache.data.get('kv', {}):
            return super().attention_optimized(x)
        
        seq_len = x.shape[-2]
        k_cache = current_kvcache.data['kv'][self.layer_id]['k']
        v_cache = current_kvcache.data['kv'][self.layer_id]['v']
        full_seqlen = k_cache.shape[-2] + seq_len

        #import pdb; pdb.set_trace()

        x = layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, eps=self.layernorm_eps)

        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias

        scale = torch.rsqrt(torch.tensor([self.d_model / self.h], dtype=x.dtype))

        # split q, k, v
        q, k, v = qkv_merged.split(self.d_model, dim=-1)
        q = q.reshape((-1, seq_len, self.h, self.d_model // self.h))
        q = q.transpose(-2, -3)
        k = k.reshape((-1, seq_len, self.h, self.d_model // self.h))
        k = k.transpose(-2, -3)
        v = v.reshape((-1, seq_len, self.h, self.d_model // self.h))
        v = v.transpose(-2, -3)

        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.q.safetensors", x=q.contiguous())
        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.k.safetensors", x=k.contiguous())
        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.v.safetensors", x=v.contiguous())

        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.kcache.safetensors", x=k_cache.contiguous())
        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.vcache.safetensors", x=v_cache.contiguous())

        #import pdb; pdb.set_trace()
        full_k = torch.concat([k_cache, k], dim=-2)
        full_v = torch.concat([v_cache, v], dim=-2)

        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.fullk.safetensors", x=full_k.contiguous())
        # debugging.save_tensors(f"trace/my/decoder.{self.layer_id}.{full_seqlen}.fullv.safetensors", x=full_v.contiguous())

        save_layer_kvcache(self.layer_id, full_k, full_v)
    
        scores = torch.matmul(q, full_k.transpose(-2, -1)) * scale
        # causal mask
        scores = F.softmax(scores, dim=-1)
        scores = torch.matmul(scores, full_v)
        # merges back
        x = scores.transpose(-2, -3)
        x = x.reshape((-1, seq_len, self.d_model))

        x = torch.matmul(x, self.attn_proj_weight) + self.attn_proj_bias
        return x


class Chatgpt2ModelWithKVCache(Chatgpt2Model):
    def __init__(self, config):
        config['decoder_class'] = DecoderLayerWithKVCache
        super().__init__(config)
        self.kv_cache = {}

    def check_cache(self, batch_id):
        data = self.kv_cache[batch_id]
        
        if 'seq_len' not in data or 'batch_size' not in data or 'kv' not in data:
            del self.kv_cache[batch_id]
            print(f"invalid seq_len/batch_size, kv")
            return False
        
        seq_len = data['seq_len']
        batch_size = data['batch_size']
        kv = data['kv']
        for layer_id in range(self.N):
            if layer_id not in kv:
                print(f"missing layer {layer_id}")
                del self.kv_cache[batch_id]
                return False
            lkv = kv[layer_id]
            if 'k' not in lkv or 'v' not in lkv:
                print(f"missing k or v")
                del self.kv_cache[batch_id]
                return False
            k = lkv['k']
            v = lkv['v']
            if k.shape != (batch_size, self.h, seq_len, self.d_model // self.h) or v.shape != (batch_size, self.h, seq_len, self.d_model // self.h):
                print(f"inconsistent k/v shape")
                return False
        
        return True
    
    def remove_cache(self, batch_id):
        if batch_id in self.kv_cache:
            del self.kv_cache[batch_id]

    def forward(self, input_ids: torch.Tensor, batch_id: None):
        if not batch_id:
            return super().forward(input_ids, None)
        
        if batch_id not in self.kv_cache:
            batch_cache = {}
            self.kv_cache[batch_id] = batch_cache
        else:
            batch_cache = self.kv_cache[batch_id]
            if batch_cache['batch_size'] != input_ids.shape[0] or (batch_cache['seq_len'] + 1) != input_ids.shape[1]:
                batch_cache = {}
                self.kv_cache[batch_id] = batch_cache  
                
        with bind_kvcache(batch_cache):
            try:
                if not batch_cache:
                    return super().forward(input_ids, None)

                x = self.embed.forward(input_ids[:, -1], offset=input_ids.size(-1) - 1)

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
            finally:
                self.check_cache(batch_id)

