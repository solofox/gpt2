import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import os
import pathlib
import copy
import json
import math
import utils

import llama_model as myllama
import official_llama.llama.model as metallama

MODEL_PATH = pathlib.Path("./models.cache/llama-7B/")
with open(MODEL_PATH / 'config.json') as f:
    MODEL_ARGS = json.load(f)
BATCH = 1
SEQLEN = 10
HIDDEN_SIZE = MODEL_ARGS['hidden_size']
CONTEXT_WINDOW = MODEL_ARGS['max_sequence_length']
VOCAB_SIZE = MODEL_ARGS['vocab_size']

def asMetaModelArgs():
    meta_args = metallama.ModelArgs()
    meta_args.dim = HIDDEN_SIZE
    meta_args.n_layers = MODEL_ARGS['num_hidden_layers']
    meta_args.n_heads = MODEL_ARGS['num_attention_heads'] 
    meta_args.n_kv_heads = meta_args.n_heads
    meta_args.vocab_size = VOCAB_SIZE
    meta_args.norm_eps = MODEL_ARGS['rms_norm_eps']
    return meta_args

def load_model_parameters(model_path):
    tensors_indexjson_file = model_path / 'model.safetensors.index.json'
    parameters = {}
    with open(tensors_indexjson_file, "rt") as f:
        indexjson = json.loads(f.read())
        for filename in set(indexjson['weight_map'].values()):
            partial = utils.load_safetensors(model_path / filename)
            parameters.update(partial)
    return parameters

def switch_to_high_resolution_floating_point(params): 
    for key, pdict in params.items():
        if 'tensor' not in pdict:
            continue
        dtype = pdict['tensor'].dtype 
        if dtype.is_floating_point and dtype.itemsize < 4:
            pdict['tensor'] = pdict['tensor'].to(torch.float)

def compare_tensor(title, name, x, y, show_data = False):
    print(title)
    is_good = True
    if show_data:
        print("x: ", x)
        print("y: ", y)

    def squeeze_high(shape: list):
        while len(shape) > 1 and shape[0] == 1:
            shape.pop(0)
    shape1 = squeeze_high(list(x.shape))
    shape2 = squeeze_high(list(y.shape))
    if shape1 != shape2:
        print(f"  Tensor {name}: inconsistent shape, {x.shape} vs {y.shape}")
        is_good = False
    elif x.dtype != y.dtype:
        print(f"  Tensor {name}: inconsistent type, {x.dtype} vs {y.dtype}")
        is_good = False
    else:
        is_good = torch.allclose(x, y, rtol=1e-4, atol=1e-5)
        diff = x - y
        distance = torch.sqrt(torch.sum(diff ** 2)).item()
        normalized_distance =  distance / diff.numel()
        print(diff)
        print(f"  Tensor {name}: distance={distance}, normalized distance={normalized_distance}, tensor.allclose={is_good}, diffmin={torch.min(diff).item()}, diffmax={torch.max(diff).item()}. shape={x.shape}, dtype={y.dtype}")
    return is_good

def test_embedding(weight):
    batch = (torch.rand(BATCH, SEQLEN) * VOCAB_SIZE).to(torch.int)

    embed = myllama.Llama1Embed(HIDDEN_SIZE, VOCAB_SIZE)
    embed.load_state_dict({"model.embed_tokens.weight": weight})
    output1 = embed.forward(batch)

    mod = metallama.ParallelEmbedding(VOCAB_SIZE, HIDDEN_SIZE, init_method=lambda x: x)
    mod.load_state_dict({'weight': weight})
    output2 = mod.forward(batch).to(output1.dtype)
    compare_tensor("compare embedding", "embed_tokens.weight", output1, output2)
    
def test_rms(weight, scale=1):
    batch = torch.rand((BATCH, SEQLEN, HIDDEN_SIZE)) * scale

    output1 = myllama.rms_norm(batch, weight)
    mod = metallama.RMSNorm(dim=HIDDEN_SIZE)
    mod.load_state_dict({'weight': weight})
    output2 = mod.forward(batch)
    compare_tensor("compare rms_norm", "norm.weight", output1, output2)

def test_rope():
    HEAD = 8
    batch = torch.rand((BATCH, HEAD, SEQLEN, HIDDEN_SIZE // HEAD))
    rope1 = myllama.ComplexRope(HIDDEN_SIZE // HEAD, HEAD, 2 * SEQLEN)
    rope2 = myllama.InterleavedRope(HIDDEN_SIZE // HEAD, HEAD, 2 * SEQLEN)

    output1 = rope1.forward(batch, offset=0)
    output2 = rope2.forward(batch, offset=0)
    compare_tensor("compare rope complex vs interleaved", "offset=0", output1, output2)

    output1 = rope1.forward(batch, offset=7)
    output2 = rope2.forward(batch, offset=7)
    compare_tensor("compare rope complex vs interleaved", "offset=13", output1, output2)

def test_rope_attention(rope1_impl: myllama.RopeImpl = myllama.RopeImpl.Complex, rope2_impl: myllama.RopeImpl = myllama.RopeImpl.Halved):
    HEAD = 8
    x = torch.rand((BATCH, SEQLEN, HIDDEN_SIZE))

    q_weight = torch.randn((HIDDEN_SIZE, HIDDEN_SIZE))
    k_weight = torch.randn((HIDDEN_SIZE, HIDDEN_SIZE))

    def select_rope(impl: myllama.RopeImpl):
        if impl == myllama.RopeImpl.Complex:
            return myllama.ComplexRope(HIDDEN_SIZE // HEAD, HEAD, 2 * SEQLEN)
        elif impl == myllama.RopeImpl.Interleaved:
            return myllama.InterleavedRope(HIDDEN_SIZE // HEAD, HEAD, 2 * SEQLEN)
        elif impl == myllama.RopeImpl.Halved:
            return myllama.HalvedRope(HIDDEN_SIZE // HEAD, HEAD, 2 * SEQLEN)
        else:
            raise Exception(f"Unsupport rope1 implementaion {impl}")

    rope1 = select_rope(rope1_impl)
    rope2 = select_rope(rope2_impl)

    q_weight_rope1 = rope1.adjust(q_weight)
    k_weight_rope1 = rope1.adjust(k_weight)
    q_weight_rope2 = rope2.adjust(q_weight)
    k_weight_rope2 = rope2.adjust(k_weight)

    scale = 1.0 / math.sqrt(HIDDEN_SIZE / HEAD)
    mask = torch.full((SEQLEN, SEQLEN), float('-inf')).triu(diagonal=1)
    def attention(x, q_weight, k_weight, rope):
        # split header
        q0 = F.linear(x, q_weight)
        k0 = F.linear(x, k_weight)
        q = F.linear(x, q_weight).reshape(*x.shape[:-1], HEAD, HIDDEN_SIZE // HEAD).transpose(1, 2)
        k = F.linear(x, k_weight).reshape(*x.shape[:-1], HEAD, HIDDEN_SIZE // HEAD).transpose(1, 2)
        rq = rope.forward(q)
        rk = rope.forward(k)
        scores = rq @ rk.transpose(-1, -2)
        return F.softmax(scores * scale + mask, dim=-1)
    
    result1 = attention(x, q_weight_rope1, k_weight_rope1, rope1)
    result2 = attention(x, q_weight_rope2, k_weight_rope2, rope2)
    compare_tensor(f"compare rope {rope1_impl.name} vs {rope2_impl.name}", "attention", result1, result2)


def test_transformblock(parameters):

    batch = (torch.rand(BATCH, SEQLEN) * VOCAB_SIZE).to(torch.int)
    
    embed = myllama.Llama1Embed(HIDDEN_SIZE, VOCAB_SIZE)
    embed.load_state_dict({"model.embed_tokens.weight": parameters['model.embed_tokens.weight']['tensor']})
    x = embed.forward(batch)

    layer_id = 0
    layer_state_dict = {}
    layer_param_prefix = f"model.layers.{layer_id}."
    for key in parameters.keys():
        if key.startswith(layer_param_prefix):
            layer_state_dict[key[len(layer_param_prefix):]] = parameters[key]['tensor']
    
    myblock = myllama.Llama1Decoder(0, HIDDEN_SIZE, MODEL_ARGS['num_attention_heads'], CONTEXT_WINDOW, MODEL_ARGS['intermediate_size'], MODEL_ARGS['rms_norm_eps'])
    myblock.load_state_dict(copy.deepcopy(layer_state_dict))

    mod = metallama.TransformerBlock(0, asMetaModelArgs())
    mod_state_dict = {
        'attention.wq.weight':    layer_state_dict['self_attn.q_proj.weight'],
        'attention.wk.weight':    layer_state_dict['self_attn.k_proj.weight'],
        'attention.wv.weight':    layer_state_dict['self_attn.v_proj.weight'],
        'attention.wo.weight':    layer_state_dict['self_attn.o_proj.weight'],
        'feed_forward.w1.weight': layer_state_dict['mlp.gate_proj.weight'],
        'feed_forward.w2.weight': layer_state_dict['mlp.down_proj.weight'],
        'feed_forward.w3.weight': layer_state_dict['mlp.up_proj.weight'],
        'attention_norm.weight':  layer_state_dict['input_layernorm.weight'],
        'ffn_norm.weight':        layer_state_dict['post_attention_layernorm.weight'],
    }
    mod.load_state_dict(mod_state_dict)
    start_pos = 0
    freqs_cis = metallama.precompute_freqs_cis(HIDDEN_SIZE // MODEL_ARGS['num_attention_heads'], CONTEXT_WINDOW)[start_pos : start_pos + SEQLEN]
    mask = torch.full((1, 1, SEQLEN, SEQLEN), float("-inf"))
    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(x)

    output1 = myblock.forward(x)
    output2 = mod.forward(x, start_pos, freqs_cis, mask).type_as(output1)

    compare_tensor("transform block", 'block', output1, output2)

def test():
    torch.set_grad_enabled(False)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group("gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(1)
    # seed must be the same in all processes
    torch.manual_seed(42)

    # parameters = load_model_parameters(MODEL_PATH)
    # if True: 
    #     switch_to_high_resolution_floating_point(parameters)
    #     print("switch floating point to >= float32")
    print("Loaded")

    #test_embedding(parameters['model.embed_tokens.weight']['tensor'])
    #test_rms(parameters['model.norm.weight']['tensor'])
    #test_rope()
    #test_rope_attention(myllama.RopeImpl.Complex, myllama.RopeImpl.Interleaved)
    test_rope_attention(myllama.RopeImpl.Complex)
    test_rope_attention(myllama.RopeImpl.Interleaved)
    #test_transformblock(parameters)

if __name__ == "__main__":
    test()

    