from transformers import AutoModelForCausalLM, AutoTokenizer, RwkvForCausalLM
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import time

DATA_PATH = Path(__file__).parent / 'data'

devices = ['cpu', 'gpu']
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
num_samples = 5
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
            start_time = time.time()
            tokens = model.generate(**tokenized_prompt, max_new_tokens=num_tokens, do_sample=True)
            total_time = time.time() - start_time
            data.append({
                "model": model_name,
                "model_size": model_size,
                'device': device,
                'total_time': total_time
            })

pd.DataFrame(data).to_csv(DATA_PATH / f'inference_results_{device}.csv')
