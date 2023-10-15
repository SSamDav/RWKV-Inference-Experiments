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


strategies = ['cuda fp32'] # 'cpu fp32', 
models = [
    # "BlinkDL/rwkv-4-pile-169m",
    # "BlinkDL/rwkv-4-pile-430m",
    # "BlinkDL/rwkv-4-pile-1b5",
    # "BlinkDL/rwkv-4-pile-3b",
    "BlinkDL/rwkv-4-pile-7b",
    "BlinkDL/rwkv-4-pile-14b",
]

model_mapping = {
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


def calculate_perplexity(logits, targets):
    loss = F.cross_entropy(logits, targets)
    perplexity = torch.exp(loss)
    return perplexity


r = requests.get("https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json")
with open("20B_tokenizer.json", "w") as fp:
  fp.write(r.text)

# Loading Tokenizer
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

# Loading Dataset
dataset = load_dataset('hoskinson-center/proof-pile', split='train', streaming=True)
tokenized_dataset = dataset.map(tokenize).filter(lambda example: example['length'] >= 128000)


for strategy in strategies:
    for model_name in models:
        processed_name = model_name.split("/")[-1].replace("-", "_")
        if not Path(model_name).exists():
            hf_hub_download(repo_id=model_name, filename=model_mapping[model_name], local_dir=f"./{model_name}")
            
        model_weights = Path(model_name) / model_mapping[model_name]
        model = RWKV(model=model_weights.as_posix(), strategy=strategy)
        with torch.no_grad():
            with open(f"perplexity_by_context_{processed_name}.jsonl", "w") as fp:
                for doc_id, doc in enumerate(tokenized_dataset):
                    if doc_id >= 10: break

                    state, previous_token = None, None
                    for token_id, token in enumerate(tqdm(doc["ids"], leave=True)):
                        if token_id == 0:
                            previous_token = token
                            continue

                        if state is not None:
                            state = [s.to(strategy.split()[0]) for s in state]

                        token_tensor = torch.tensor([previous_token])
                        output, state = model.forward(token_tensor, state=state)
                        cross_entropy = F.cross_entropy(output.cpu().unsqueeze(0), torch.tensor([token])).tolist()
                        json.dump(
                            {
                                "context_length": token_id,
                                "cross_entropy": cross_entropy,
                                "doc_id": doc_id
                            }, fp
                        )
                        fp.write("\n")
                        previous_token = token