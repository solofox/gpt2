
import math
import torch
import torch.nn.functional as F

from typing import Tuple

def gelu_new(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

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
        x.add(self.wpe_weight[:seq_len, :])
        return x
    
class LMHead():
    def __init__(self, d_model, vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size

    def load_state_dict(self, state_dict):
        self.weight = state_dict.pop('wpe.weight').T

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.softmax(x, dim=-1)
        return x

class DecoderLayer():
    def __init__(self, d_model, h, layernorm_eps: 1e-05):
        self.d_model = d_model
        self.h = h
        #self.scale = torch.sqrt(d_model / h)
        self.layernorm_eps = layernorm_eps

    def load_state_dict(self, state_dict):
        # h.0.attn.c_attn.weight
        # h.0.attn.c_attn.bias
        # h.0.attn.bias           [1,1,1024,1024]
        # h.0.attn.c_proj.weight
        # h.0.attn.c_proj.bias
        self.attn_weight = state_dict.pop('attn.c_attn.weight')
        self.attn_bias = state_dict.pop('attn.c_attn.bias')
        self.attn_proj_weight = state_dict.pop('attn.c_proj.weight')
        self.attn_proj_bias = state_dict.pop('attn.c_proj.bias')

        # layer norm
        self.ln_1_weight = state_dict.pop('ln_1.weight')
        self.ln_1_bias = state_dict.pop('ln_1.bias')
        self.ln_2_weight = state_dict.pop('ln_2.weight')
        self.ln_2_bias = state_dict.pop('ln_2.bias')

        # MLP
        self.c_fc_weight = state_dict.pop('mlp.c_fc.weight')
        self.c_fc_bias = state_dict.pop('mlp.c_fc.bias')
        self.c_proj_weight = state_dict.pop('mlp.c_proj.weight')
        self.c_proj_bias = state_dict.pop('mlp.c_proj.bias')

        # 剩下一个 attn.bias 是啥？
        attn_bias = state_dict.pop('attn.bias')
        print("Dropping paramter h.*.attn.bias")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = F.layer_norm(x, normalized_shape=(self.d_model,), weight=self.ln_1_weight, bias=self.ln_1_bias, eps=self.layernorm_eps)
        qkv_merged = torch.matmul(x, self.attn_weight) + self.attn_bias
        qkv_splited = torch.split(qkv_merged, self.d_model // self.h, dim=-1)
        assert len(qkv_splited) == 3 * self.h

        q_splited = qkv_splited[0 : self.h]
        k_splited = qkv_splited[self.h : 2 * self.h]
        v_splited = qkv_splited[2 * self.h : ]

        scale = math.sqrt(self.d_model / self.h)
        # attention
        heads = []
        for i in range(self.h):
            qk_similarities = torch.matmul(q_splited[i], k_splited[i].transpose(-2, -1)) / scale
            # TODO: do a self mask
            qk_similarities = F.softmax(qk_similarities, dim=-1)
            headi = torch.matmul(qk_similarities, v_splited[i])
            heads.append(headi)
        x = torch.concat(heads, dim=-1)
        x = torch.matmul(x, self.attn_proj_weight) + self.attn_proj_bias

        # residual connection
        x = x + residual

        residual = x
        
        x = F.layer_norm(x, normalized_shape=(self.d_model,), weight=self.ln_2_weight, bias=self.ln_2_bias, eps=self.layernorm_eps)

        # FFN
        x = torch.matmul(x, self.c_fc_weight) + self.c_fc_bias
        x = gelu_new(x)
        x = torch.matmul(x, self.c_proj_weight) + self.c_proj_bias

        # residual connection
        x = x + residual

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
        for _ in range(self.N):
            self.decoders.append(
                DecoderLayer(self.d_model, self.h, layernorm_eps=self.layernorm_eps)
            )
        self.lm_head = LMHead(self.d_model, self.vocab_size)
        self.embed = Embed(self.d_model, self.vocab_size)

    def load_state_dict(self, state_dict):
        metadata = state_dict.pop('__metadata__')
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
        self.lm_head.load_state_dict({'wpe.weight': wpe_weight.clone().detach()})

        self.ln_f_weight = state_dict.pop('ln_f.weight')['tensor']
        self.ln_f_bias = state_dict.pop('ln_f.bias')['tensor']

        if state_dict:
            raise Exception(f"Unknown parameters: {state_dict.keys()}")
        
    def forward(self, input_ids: torch.Tensor):
        # input_ids: [batch, seq]
        x = self.embed.forward(input_ids)
        # x: [batch, seq, d_model]
        for decoder in self.decoders:
            x = decoder.forward(x)
        # x: [batch, seq, d_model]
        x = F.layer_norm(x, normalized_shape=(self.d_model,), weight=self.ln_f_weight, bias=self.ln_f_bias, eps=self.layernorm_eps)
        # 提取最后一个维度的向量
        x = x[:, -1, :]
        # x: [batch, d_model]
        logits: torch.Tensor = self.lm_head.forward(x)
        # logits: [batch, vocab_size]
        return logits
