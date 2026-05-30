import math
import torch
import torch.nn.functional as F

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

class Llama1Decoder():
    def __init__(self, layer_id, d_model, h, context_window, intermediate_size, rms_norm_eps = 1e-6):
        self.layer_id = layer_id
        self.d_model = d_model
        self.h = h
        self.head_dim = d_model // h
        self.context_window = context_window
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.inv_square_dk = 1.0 / math.sqrt( self.head_dim )

        self.attention_mask = torch.tril(torch.ones((context_window, context_window), dtype=torch.int)).to(torch.bool).logical_not()

        self.rope_base = 10000
        # 计算 rope_weight: [context_window, head_dim/2]
        pos = torch.arange(0, self.context_window).unsqueeze(dim=1)
        freqs = 1 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)).unsqueeze(dim=0)
        rope_angles = pos * freqs
        self.rope_weight = torch.polar(torch.ones_like(rope_angles), rope_angles)

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
        self.attn_q_proj_weight = state_dict.pop('self_attn.q_proj.weight')
        self.attn_k_proj_weight = state_dict.pop('self_attn.k_proj.weight')
        self.attn_v_proj_weight = state_dict.pop('self_attn.v_proj.weight')
        self.attn_o_proj_weight = state_dict.pop('self_attn.o_proj.weight')

        self.post_attn_layernorm_weight = state_dict.pop('post_attention_layernorm.weight')
        self.rope_inv_freq = state_dict.pop('self_attn.rotary_emb.inv_freq')

    def do_rope(self, x: torch.Tensor):
        # x: [B, H, S, head_dim]
        x_shape = x.shape
        seq_len = x_shape[-2]
        # x reshape to: [B, H, S, head_dim / 2, 2]
        # xc: [B, H, S, head_dim / 2] as complex
        xc = torch.view_as_complex(x.reshape(*x.shape[:-1], x.shape[-1] // 2,  2))
        # rope_weight: [context_window, head_dim / 2], complex as element
        rc = xc * self.rope_weight[: seq_len, :]
        return torch.view_as_real(rc).reshape(x_shape)

    def attention(self, x):
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
        q = self.do_rope(q).to(x.dtype)
        k = self.do_rope(k).to(x.dtype)
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

    def forward(self, input_ids: torch.Tensor, batch_id: int = 0):
        # input_ids: [B, S]
        x = self.embed.forward(input_ids)
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
    