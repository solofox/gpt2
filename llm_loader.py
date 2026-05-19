from os import PathLike
import json
from transformers import AutoTokenizer

import gpt2_model
import utils

from typing import Tuple, Any

def load(model_path: PathLike) -> Tuple[Any, Any]:

    with open(model_path / 'config.json', 'rt') as f:
        model_config = json.loads(f.read())        

    model_type = model_config.get('model_type', 'unknown')
    if model_type == 'gpt2':
        model = gpt2_model.Chatgpt2Model(model_config)
    else:
        raise Exception(f"Unknown supported model for type {model_type}")

    # load parameters
    parameters = utils.load_safetensors(model_path / 'model.safetensors')
    model.load_state_dict(parameters)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(tokenizer, type(tokenizer))
    return model, tokenizer

