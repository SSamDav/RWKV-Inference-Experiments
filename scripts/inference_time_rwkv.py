import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

import requests
from tokenizers import Tokenizer
from pathlib import Path

import subprocess
import pandas as pd
import torch
from rwkv.model import RWKV
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

DATA_PATH = Path(__file__).parent / '../data'
DATA_PATH.mkdir(exist_ok=True)

# getting the tokenizer
r = requests.get("https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json")
with open("20B_tokenizer.json", "w") as fp:
  fp.write(r.text)

# Loading Tokenizer
TOKENIZER = Tokenizer.from_file("20B_tokenizer.json")

def git(*args):
    return subprocess.check_call(['git'] + list(args))

def sample(outputs):
    probs = nn.functional.softmax(outputs, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens

strategies = ['cpu fp32', 'cuda fp32']
recompute_all_models = False
models = [
    "BlinkDL/rwkv-4-pile-169m",
    "BlinkDL/rwkv-4-pile-430m",
    "BlinkDL/rwkv-4-pile-1b5",
    "BlinkDL/rwkv-4-pile-3b",
    "BlinkDL/rwkv-4-pile-7b",
    "BlinkDL/rwkv-4-pile-14b",
]

model_mapping = {
    "BlinkDL/rwkv-4-pile-169m": "RWKV-4-Pile-169M-20220807-8023.pth",
    "BlinkDL/rwkv-4-pile-430m": "RWKV-4-Pile-430M-20220808-8066.pth",
    "BlinkDL/rwkv-4-pile-1b5": "RWKV-4-Pile-1B5-20220903-8040.pth",
    "BlinkDL/rwkv-4-pile-3b": "RWKV-4-Pile-3B-20221008-8023.pth",
    "BlinkDL/rwkv-4-pile-7b": "RWKV-4-Pile-7B-20221115-8047.pth",
    "BlinkDL/rwkv-4-pile-14b": "RWKV-4-Pile-14B-20230213-8019.pth",

}

num_tokens = 1024
num_samples = 1
prompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'
tokenized_prompt = TOKENIZER.encode(prompt).ids

data = []
if (DATA_PATH / 'inference_results_rwkv.csv').exists() and not recompute_all_models:
    data = pd.read_csv(DATA_PATH / 'inference_results_rwkv.csv').to_dict('records')

for strategy in strategies:
    for model_name in models:
        if any(d["model_name"] == model_name and d["strategy"] == strategy for d in data):
            continue

        try:
            if not Path(model_name.split("/")[-1]).exists():
                hf_hub_download(repo_id="model_name", filename=model_mapping[model_name], local_dir=f"./{model_name}")

            

            state = None
            tokenized_prompt = torch.tensor(tokenized_prompt)
            next_token = tokenized_prompt
            full_text = tokenized_prompt
            model_weights = Path(model_name.split("/")[-1]) / model_mapping[model_name]
            model = RWKV(model=model_weights.as_posix(), strategy=strategy)
            model_size = sum(p.numel() for p in model.parameters())

            for tok_idx in range(num_tokens):
                if state is not None:
                    state = [s.to(strategy.split()[0]) for s in state]

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
                    with record_function("model_inference"):
                        output, state = model.forward(next_token, state=state)

                full_profile = next(event for event in prof.key_averages() if event.key == 'model_inference')
                next_token = sample(output).cpu()
                full_text = torch.cat([full_text, next_token], dim=-1)
                gen_text = TOKENIZER.decode(full_text.tolist())
                data.append({
                    "model_name": model_name,
                    "model_size": model_size,
                    "token_id": tok_idx,
                    "final_text": gen_text,
                    "strategy": strategy,
                    "cpu_time": full_profile.cpu_time,
                    "cuda_time": full_profile.cuda_time,
                    "cpu_memory_usage": full_profile.cpu_memory_usage,
                    "cuda_memory_usage": full_profile.cuda_memory_usage,
                    "self_cpu_memory_usage": full_profile.self_cpu_memory_usage,
                    "self_cuda_memory_usage": full_profile.self_cuda_memory_usage

                })

                pd.DataFrame(data).to_csv(DATA_PATH / 'inference_results_rwkv.csv')
        
        except:
            continue