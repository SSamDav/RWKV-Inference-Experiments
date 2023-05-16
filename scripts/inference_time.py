from pathlib import Path

import pandas as pd
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RwkvForCausalLM, LogitsProcessorList
from torch import nn
import torch
import gc

DATA_PATH = Path(__file__).parent / '../data'
LOGITS_PROCESSOR = LogitsProcessorList()

def process_outputs(input_ids, outputs):
    next_token_logits = outputs.logits[:, -1, :]
    next_token_scores = LOGITS_PROCESSOR(input_ids, next_token_logits)
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
    if hasattr(outputs, 'state'):
        state = outputs.state
        return input_ids, state
    
    return input_ids

devices = ['cpu', 'cuda']
models = [
    "EleutherAI/pythia-160m",
    "EleutherAI/gpt-neo-125m",
    "facebook/opt-125m",
    "RWKV/rwkv-4-169m-pile",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-neox-20b",
    "facebook/opt-13b",
    "RWKV/rwkv-4-14b-pile"
]
num_tokens = 100
num_samples = 1
prompt = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.'

data = []
for device in devices:
    for model_name in tqdm(models):
        model_cls = AutoModelForCausalLM if 'rwkv' not in model_name.lower() else RwkvForCausalLM
        model = model_cls.from_pretrained(model_name)
        model = model.to(device)
        model_size = sum(p.numel() for p in model.parameters())

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}

        for _ in range(num_samples):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
                with record_function("model_inference"):
                    tokens = model.generate(**tokenized_prompt, max_new_tokens=num_tokens, do_sample=True)

            full_profile = next(event for event in prof.key_averages() if event.key == 'model_inference')
            data.append({
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

        del model
        gc.collect()
        torch.cuda.empty_cache() 

        pd.DataFrame(data).to_csv(DATA_PATH / f'inference_results_{device}.csv')
