from datasets import load_dataset
from tokenizers import Tokenizer
from rwkv.model import RWKV
from huggingface_hub import hf_hub_download
from pathlib import Path
import torch.nn.functional as F
from tqdm.auto import tqdm

import requests
import torch
import json
import os

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

strategies = ['cuda fp32'] # 'cpu fp32', 
models = [
    # "BlinkDL/rwkv-4-pile-169m",
    # "BlinkDL/rwkv-4-pile-430m",
    # "BlinkDL/rwkv-4-pile-1b5",
    # "BlinkDL/rwkv-4-pile-3b",
    "BlinkDL/rwkv-4-pile-7b",
    # "xiaol/RWKV-claude-4-World-7B-65k",
    # "BlinkDL/rwkv-4-pile-14b",
]

model_mapping = {
    "xiaol/RWKV-claude-4-World-7B-65k": "RWKV-claude-4-World-7B-20230805-ctx65k.pth",
    "BlinkDL/rwkv-4-pile-169m": "RWKV-4-Pile-169M-20220807-8023.pth",
    "BlinkDL/rwkv-4-pile-430m": "RWKV-4-Pile-430M-20220808-8066.pth",
    "BlinkDL/rwkv-4-pile-1b5": "RWKV-4-Pile-1B5-20220903-8040.pth",
    "BlinkDL/rwkv-4-pile-3b": "RWKV-4-Pile-3B-20221008-8023.pth",
    "BlinkDL/rwkv-4-pile-7b": "RWKV-4-Pile-7B-20230406-ctx8192-test949.pth",
    "BlinkDL/rwkv-4-pile-14b": "RWKV-4-Pile-14B-20230313-ctx8192-test1050.pth",

}


def tokenize(example):
    ids = tokenizer.encode(example["text"]).ids
    return {"ids": ids[:128000], "length": len(ids)}


def compute_window_perplexity(model, input, context_size):
    seq_len = len(input)
    prev_end_loc = 0 
    nlls = []
    for begin_loc in tqdm(list(range(0, seq_len, 256)), leave=False, desc=f"Ctx: {context_size}"):
        end_loc = min(begin_loc + context_size, seq_len)
        trg_len = end_loc - prev_end_loc
        window_ids = input[begin_loc:end_loc]
        
        window_loss = []
        inputs_ids = torch.tensor(window_ids[:-trg_len])
        target_ids = torch.tensor(window_ids[-trg_len:])
        if len(inputs_ids) > 0:
            output, state = model.forward(inputs_ids, state=None)
            for token_id in target_ids:
                window_loss.append(F.cross_entropy(output.cpu().unsqueeze(0), torch.tensor([token_id])).tolist())
                output, state = model.forward(token_id.unsqueeze(0), state=state)
                
            nlls.append(torch.tensor(window_loss).mean().float())
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
        
    return float(torch.exp(torch.stack(nlls).mean()).float().cpu())


r = requests.get("https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json")
with open("20B_tokenizer.json", "w") as fp:
  fp.write(r.text)

# Loading Tokenizer
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# Loading Dataset
dataset = load_dataset('hoskinson-center/proof-pile', split='test', streaming=True)
tokenized_dataset = dataset.map(tokenize).filter(lambda example: example['length'] >= 128000)


for strategy in strategies:
    for model_name in models:
        processed_name = model_name.split("/")[-1].replace("-", "_")
        if not Path(model_name).exists():
            hf_hub_download(repo_id=model_name, filename=model_mapping[model_name], local_dir=f"./{model_name}")
            
        model_weights = Path(model_name) / model_mapping[model_name]
        model = RWKV(model=model_weights.as_posix(), strategy=strategy)
        pbar = tqdm(total=5)
        with torch.no_grad():
            for doc_id, doc in enumerate(tokenized_dataset):
                if doc_id >= 5: break
                
                with open(f"perplexity_by_context_{processed_name}_docid_{doc_id}.jsonl", "w") as fp:
                    for ctx_size in range(2048, 128000, 2048):
                        perplexity = compute_window_perplexity(model, doc["ids"], ctx_size)  
                        json.dump(
                            {
                                "context_length": ctx_size,
                                "perplexity": perplexity,
                                "doc_id": doc_id
                            }, fp
                        )
                        fp.write("\n")
                        
                tqdm.update(1)