import pathlib
import time
import click
from functools import partial
from modelscope import snapshot_download

MODELS = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

FILES = ['config.json', 'tokenizer_config.json', 'README.md', 'tokenizer.json', 'vocab.json', 'merges.txt', '*.safetensors']

MODELS_INFO = {
    'gpt2-small': {
        'repo': 'openai-community/gpt2',
    },
    'gpt2-medium': {
        'repo': 'openai-community/gpt2-medium',
    },
    'gpt2-large': {
        'repo': 'openai-community/gpt2-large',
    },
    'gpt2-xl': {
        'repo': 'openai-community/gpt2-xl',
    },
}

@click.command(help=f"Download GPT2 models from modelscope, models are {' '.join(MODELS)}")
@click.option("--cache-dir", "--dir", default="./models.cache", help="", type=str)
@click.argument('models', required=True, nargs=-1)
def download(models: list[str], cache_dir: str):
    cache_dir = pathlib.Path(cache_dir)
    for model in models:
        if model not in MODELS:
            print(f"[ERROR] Unknown model {model}, please select from {MODELS}")
            continue
        model_dir = cache_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Start to download model {model}")
        snapshot_download(
            repo_id = MODELS_INFO[model]['repo'],
            repo_type = 'model',
            local_dir = model_dir,
            allow_patterns = FILES,
            #resume_download = True,
        )
        
if __name__ == "__main__":
    download()

