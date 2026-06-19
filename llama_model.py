import math
import contextlib
import threading
import enum
import torch
import abc
import torch.nn.functional as F


class RopeImpl(enum.IntEnum):
    Complex = 1
    Interleaved = 2
    Halved = 3

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


def rms_norm(input, weight: torch.Tensor | None, eps: float=1e-6):
    dtype = input.dtype
    rms = torch.rsqrt(input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = (input * rms).to(dtype)
    if weight is not None:
        normed = normed * weight
    return normed
    #return F.rms_norm(input, normalized_shape=(input.shape[0],), weight=weight, eps=eps)

class Llama1Embed():
    def __init__(self, d_model, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = d_model

    def load_state_dict(self, state_dict):
        self.embed_tokens_weight = state_dict.pop('model.embed_tokens.weight')
        assert self.embed_tokens_weight.shape[-1] == self.d_model
        assert self.embed_tokens_weight.shape[-2] == self.vocab_size
    
    def forward(self, input_ids):
        # input_ids: [B, S] => output: [B, S, d_model]
        return self.embed_tokens_weight[input_ids]

class Llama1LMHead():
    def __init__(self, d_model, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = d_model

    def load_state_dict(self, state_dict):
        self.lm_head_weight = state_dict.pop('lm_head.weight')
        assert self.lm_head_weight.shape[-1] == self.d_model
        assert self.lm_head_weight.shape[-2] == self.vocab_size
    
    def forward(self, x):
        # x: [B, H] -> [B, vocab_size]
        return F.linear(x, self.lm_head_weight)

class Rope(abc.ABC):
    def __init__(self, dim: int, head: int, context_window: int, theta: int = 10000):
        assert dim % 2 == 0
        self.dim = dim
        self.context_window = context_window
        self.head = head
    
    def adjust(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    @abc.abstractmethod
    def forward(self, x, offset=0):
        ...
    
class ComplexRope(Rope):
    def __init__(self, dim: int, head: int, context_window: int, theta: int = 10000):
        super().__init__(dim, head, context_window, theta)
        pos = torch.arange(context_window).unsqueeze(dim=1)
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        m = pos * inv_freqs
        self.rope_weight = torch.polar(torch.ones_like(m), m)

    def forward(self, x, offset=0):
        shape = x.shape
        assert shape[-1] == self.dim
        x_as_complex = torch.view_as_complex(x.view(*shape[:-1], self.dim // 2, 2))
        rotated_x = x_as_complex * self.rope_weight[offset: offset + shape[-2], :]
        return torch.view_as_real(rotated_x).reshape(shape).type_as(x)

class InterleavedRope(Rope):
    def __init__(self, dim: int, head: int, context_window: int, theta: int = 10000):
        super().__init__(dim, head, context_window, theta)
        pos = torch.arange(context_window).unsqueeze(dim=1)
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        m = pos * inv_freqs
        self.rope_angles = torch.zeros(context_window, dim)
        self.rope_angles[:, 0::2] = m
        self.rope_angles[:, 1::2] = m
    
    def forward(self, x, offset=0):
        assert x.shape[-1] == self.dim
        rope_angles = self.rope_angles[offset:offset+x.shape[-2], :]

        even_features = x[..., 0::2]
        odd_features = x[..., 1::2]
        swapped = torch.stack([-odd_features, even_features], dim=-1).reshape(x.shape)
        rx = x * rope_angles.cos() + swapped * rope_angles.sin()
        return rx.type_as(x)

class HalvedRope(Rope):
    def __init__(self, dim: int, head: int, context_window: int, theta: int = 10000):
        super().__init__(dim, head, context_window, theta)

        pos = torch.arange(context_window).unsqueeze(dim=1)
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        m = pos * inv_freqs
        self.rope_angles = torch.zeros(context_window, dim)
        self.rope_angles[:, :dim//2] = m
        self.rope_angles[:, dim//2:] = m

    def adjust(self, weight: torch.Tensor) -> torch.Tensor:
        # weight.shape[0] == feature dim
        # weight.shape: [out-features, in-features], out-features & in-features = hidden_size
        assert weight.shape == (self.dim * self.head, self.dim * self.head)

        shape = weight.shape
        # x: [dim, dim] => (head, head_dim // 2, 2, dim)
        x = weight.reshape(self.head, self.dim // 2, 2, self.dim * self.head)
        # in each head: move even together, move odd together
        y = x.transpose(1, 2).reshape(shape)
        return y.contiguous()
    
    def forward(self, x, offset=0):
        assert x.shape[-1] == self.dim
        rope_angles = self.rope_angles[offset:offset+x.shape[-2], :]

        first_half = x[..., 0:self.dim//2]
        second_half = x[..., self.dim//2:]
        swapped_half = torch.cat([-second_half, first_half], dim=-1)
        return x * rope_angles.cos() + swapped_half * rope_angles.sin()

class Llama1Decoder():
    def __init__(self, layer_id, d_model, h, context_window, intermediate_size, rms_norm_eps = 1e-6, rope_impl: RopeImpl = RopeImpl.Complex):
        self.layer_id = layer_id
        self.d_model = d_model
        self.h = h
        self.head_dim = d_model // h
        self.context_window = context_window
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.inv_square_dk = 1.0 / math.sqrt( self.head_dim )

        self.attention_mask = torch.tril(torch.ones((context_window, context_window), dtype=torch.int)).to(torch.bool).logical_not()

        if rope_impl == RopeImpl.Complex:
            self.rope = ComplexRope(self.head_dim, self.context_window)
        else:
            raise Exception(f"Unknown ROPE implementation required f{rope_impl}")
    
    def load_state_dict(self, state_dict):
        # "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
        # "model.layers.0.self_attn.rotary_emb.inv_freq": "model-00001-of-00002.safetensors",
        self.input_layernorm_weight = state_dict.pop('input_layernorm.weight')
        self.mlp_up_proj_weight = state_dict.pop('mlp.up_proj.weight')
        self.mlp_gate_proj_weight = state_dict.pop('mlp.gate_proj.weight')
        self.mlp_down_proj_weight = state_dict.pop('mlp.down_proj.weight')
        self.attn_q_proj_weight = self.rope.adjust(state_dict.pop('self_attn.q_proj.weight'))
        self.attn_k_proj_weight = self.rope.adjust(state_dict.pop('self_attn.k_proj.weight'))
        self.attn_v_proj_weight = state_dict.pop('self_attn.v_proj.weight')
        self.attn_o_proj_weight = state_dict.pop('self_attn.o_proj.weight')

        self.post_attn_layernorm_weight = state_dict.pop('post_attention_layernorm.weight')
        self.rope_inv_freq = state_dict.pop('self_attn.rotary_emb.inv_freq')

    def do_rope(self, x: torch.Tensor, offset=0):
        # x: [B, H, S, head_dim]
        x_shape = x.shape
        seq_len = x_shape[-2]
        # x reshape to: [B, H, S, head_dim / 2, 2]
        # xc: [B, H, S, head_dim / 2] as complex
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], x.shape[-1] // 2,  2))
        # rope_weight: [context_window, head_dim / 2], complex as element
        rc = xc * self.rope_weight[offset: offset + seq_len, :]
        return torch.view_as_real(rc).reshape(x_shape)

    def attention_nocache(self, x):
        #import pdb; pdb.set_trace()
        seq_len = x.size(1)

        x = rms_norm(x, self.input_layernorm_weight, self.rms_norm_eps)
        # q, k, v: [B, S, d_model]
        q = F.linear(x, self.attn_q_proj_weight)
        k = F.linear(x, self.attn_k_proj_weight)
        v = F.linear(x, self.attn_v_proj_weight)
        # q, k, v: [B, S, h, d_model // h]
        q = q.reshape(-1, seq_len, self.h, self.d_model // self.h)
        k = k.reshape(-1, seq_len, self.h, self.d_model // self.h)
        v = v.reshape(-1, seq_len, self.h, self.d_model // self.h)
        # q, k, v: [B, h, S, d_model //h]
        q.transpose_(1, 2)
        k.transpose_(1, 2)
        v.transpose_(1, 2)
        # rope
        q = self.rope.forward(q)
        k = self.rope.forward(k)

        save_layer_kvcache(self.layer_id, k, v)

        # 计算 attention
        # scores: [B, h, S, S]
        #print(f"q.shape={q.shape}")
        #print(f"k.shape={k.shape}")
        #scores = F.linear(q, k) * self.inv_square_dk
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_square_dk
        #print("myllama-similarities=", scores)
        similarities = F.softmax( scores.masked_fill(self.attention_mask[:seq_len, :seq_len], float('-inf')) , dim=-1)
        #print("similarities=", similarities)
        #x = F.linear(similarities, v)
        x = torch.matmul(similarities, v)
        # 合并
        x.transpose_(1, 2)
        x = x.reshape(-1, seq_len, self.d_model)
        # 输出
        x = F.linear(x, self.attn_o_proj_weight)
        return x

    def attention(self, x):
        if self.layer_id not in current_kvcache.data.get('kv', {}):
            return self.attention_nocache(x)
        
        seq_len = x.size(1)
        k_cache = current_kvcache.data['kv'][self.layer_id]['k']
        v_cache = current_kvcache.data['kv'][self.layer_id]['v']
        start_offset = k_cache.shape[-2]

        x = rms_norm(x, self.input_layernorm_weight, self.rms_norm_eps)
        # q, k, v: [B, S, d_model]
        q = F.linear(x, self.attn_q_proj_weight)
        k = F.linear(x, self.attn_k_proj_weight)
        v = F.linear(x, self.attn_v_proj_weight)
        # q, k, v: [B, S, h, d_model // h]
        q = q.reshape(-1, seq_len, self.h, self.d_model // self.h)
        k = k.reshape(-1, seq_len, self.h, self.d_model // self.h)
        v = v.reshape(-1, seq_len, self.h, self.d_model // self.h)
        # q, k, v: [B, h, S, d_model //h]
        q.transpose_(1, 2)
        k.transpose_(1, 2)
        v.transpose_(1, 2)
        # rope
        q = self.rope.forward(q, offset=start_offset)
        k = self.rope.forward(k, offset=start_offset)
        # kv cache
        k = torch.cat([k_cache, k], dim=-2)
        v = torch.cat([v_cache, v], dim=-2)
        save_layer_kvcache(self.layer_id, k, v)
        # 计算 attention
        # scores: [B, h, S, S]
        #print(f"q.shape={q.shape}")
        #print(f"k.shape={k.shape}")
        #scores = F.linear(q, k) * self.inv_square_dk
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_square_dk
        #print("myllama-similarities=", scores)
        similarities = F.softmax( scores.masked_fill(self.attention_mask[:seq_len, :seq_len], float('-inf')) , dim=-1)
        #print("similarities=", similarities)
        #x = F.linear(similarities, v)
        x = torch.matmul(similarities, v)
        # 合并
        x.transpose_(1, 2)
        x = x.reshape(-1, seq_len, self.d_model)
        # 输出
        x = F.linear(x, self.attn_o_proj_weight)
        return x
    
    def mlp(self, x):
        # x: [B, S, d_dmodel]
        x = rms_norm(x, self.post_attn_layernorm_weight, self.rms_norm_eps)
        fx = F.linear(x, self.mlp_up_proj_weight)
        gx = F.linear(x, self.mlp_gate_proj_weight)
        gx = F.silu(gx)
        x = fx * gx
        x = F.linear(x, self.mlp_down_proj_weight)
        return x
    
    def forward(self, x):
        # attention
        y = self.attention(x)
        x = x + y
        # mlp
        y = self.mlp(x)
        x = x + y
        return x

class Llama1Model():
    def __init__(self, config):
        self.d_model = config['hidden_size']
        self.h = config['num_attention_heads']
        self.N = config['num_hidden_layers']
        self.context_window = config['max_sequence_length']
        self.bos_token_id = config['bos_token_id']
        self.eos_token_id = config['eos_token_id']
        self.pad_token_id = config['pad_token_id']
        self.norm_eps = config['rms_norm_eps']
        self.max_seq_len = config['max_sequence_length']
        self.vocab_size = config['vocab_size']
        self.kv_cache = {}

        self.decoders = []
        for i in range(self.N):
            decoder = Llama1Decoder(i, self.d_model, self.h, self.max_seq_len, config['intermediate_size'], self.norm_eps)
            self.decoders.append(decoder)

        self.embed = Llama1Embed(self.d_model, self.vocab_size)
        self.lm_head = Llama1LMHead(self.d_model, self.vocab_size)
        
    def load_state_dict(self, state_dict):
        for layer_id in range(self.N):
            layer_state_dict = {}
            layer_params = set()
            layer_param_prefix = f"model.layers.{layer_id}."
            for key in state_dict.keys():
                if key.startswith(layer_param_prefix):
                    layer_params.add(key)
                    layer_state_dict[key[len(layer_param_prefix):]] = state_dict[key]
            for key in layer_params:
                state_dict.pop(key)
            self.decoders[layer_id].load_state_dict(layer_state_dict)
        self.embed.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)
        self.norm_weight = state_dict.pop('model.norm.weight')
        assert not state_dict, "Unknown parameters in state dict."

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
    
    def forward(self, input_ids: torch.Tensor, batch_id: int = 0, disable_kvcache: bool = False):
        start_offset = 0
        if disable_kvcache:
            batch_cache = {}
        elif batch_id not in self.kv_cache:
            batch_cache = {}
            self.kv_cache[batch_id] = batch_cache
        else:
            batch_cache = self.kv_cache[batch_id]
            if batch_cache['batch_size'] != input_ids.shape[0] or (batch_cache['seq_len'] + 1) != input_ids.shape[1]:
                batch_cache = {}
                self.kv_cache[batch_id] = batch_cache 
            else:
                # hit cache only in this branch
                start_offset = batch_cache['seq_len']
        
        with bind_kvcache(batch_cache):
            try:
                if start_offset:
                    input_ids = input_ids[:, -1:]
                
                # input_ids: [B, S]
                x = self.embed.forward(input_ids)      # using ROPE, here no offset needed
                # x: [B, S, d_model]
                for decoder in self.decoders:
                    x = decoder.forward(x)
                # x: [B, S, d_model]
                x = x[:, -1, :]
                # x: [B, d_model]
                x = rms_norm(x, self.norm_weight, self.norm_eps)
                logits = self.lm_head.forward(x)
                # x: [B, vocab_size]
                return logits        
            finally:
                self.check_cache(batch_id)

def load(config):
    assert config['model_type'] == 'llama'

    model = None 

    train_window = config['max_position_embeddings']
    if train_window == 2048:
        assert config['hidden_act'] == "silu"
        model = Llama1Model(config)
    else:
        raise Exception(f"Unknown max_position_embeddings value {train_window}")
    
    return model
    
