from os import PathLike
import json
from transformers import AutoTokenizer

import gpt2_model
import llama_model
import utils

from typing import Tuple, Any

def load(model_path: PathLike) -> Tuple[Any, Any]:

    with open(model_path / 'config.json', 'rt') as f:
        model_config = json.loads(f.read())        

    model_type = model_config.get('model_type', 'unknown')
    if model_type == 'gpt2':
        model = gpt2_model.Chatgpt2Model(model_config)
    elif model_type == 'llama':
        model = llama_model.load(model_config)
    else:
        raise Exception(f"Unknown supported model for type {model_type}")

    # load parameters
    tensors_indexjson_file = model_path / 'model.safetensors.index.json'
    model_safetensors_file = model_path / 'model.safetensors'
    if tensors_indexjson_file.exists() and tensors_indexjson_file.is_file():
        parameters = {}
        with open(tensors_indexjson_file, "rt") as f:
            indexjson = json.loads(f.read())
            for filename in set(indexjson['weight_map'].values()):
                partial = utils.load_safetensors(model_path / filename)
                parameters.update(partial)
    elif model_safetensors_file.exists() and model_safetensors_file.is_file():
        parameters = utils.load_safetensors(model_safetensors_file)
    else:
        raise Exception("No model parameters files found")
    
    model.load_state_dict(parameters)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

