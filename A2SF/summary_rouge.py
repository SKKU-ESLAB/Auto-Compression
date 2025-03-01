import json
from tqdm import tqdm
import copy
import torch
import os


from rouge import Rouge

from transformers import AutoTokenizer

def load_json(file_path, key):
    with open(file_path, "r") as f:
        file = f.readlines()
        result = list()
        for line in file:
            data = json.loads(line)
            result.append(data[key])
    return result

rouge = Rouge()

answer_path = "data/xsum-3shot.jsonl"
data_folder_path = "results/original_5shot_150"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Data Load
with open(answer_path, "r") as f:
    file = f.readlines()
    prompts, input_idses, answers, input_lengths = list(), list(), list(), list()
    for line in file:
        data = json.loads(line)
        prompt = data["article"]
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
        answer = data["summary_gt"]
        seq_len = input_ids.numel()
        
        if seq_len > 4096:
            continue
        
        answers.append(answer)

full_model = load_json(os.path.join(data_folder_path, "full_model.jsonl"), "response")
h2o_model = load_json(os.path.join(data_folder_path, "h2o_model.jsonl"), "response")
    
for factor in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for mode in ["Local", "No Local"]:
        print(f"Factor: {factor} / Mode: {mode}")

        a2sf_model = load_json(os.path.join(data_folder_path, f"a2sf_{factor}_{mode}_model.jsonl"), "response")
        
        full_cache_score, h2o_score, a2sf_score = list(), list(), list()
        
        for rouge_type in ["1", "2", "l"]:
            for answer, full_cache, h2o, a2sf in zip(answers, full_model, h2o_model, a2sf_model):
                
                if full_cache != "":
                    full_cache_score.append(rouge.get_scores(answer, full_cache)[0][f"rouge-{rouge_type}"]["f"])
                else:
                    full_cache_score.append(0)
                
                if h2o != "":
                    h2o_score.append(rouge.get_scores(answer, h2o)[0][f"rouge-{rouge_type}"]["f"])
                else:
                    h2o_score.append(0)
                
                if a2sf != "":
                    a2sf_score.append(rouge.get_scores(answer, a2sf)[0][f"rouge-{rouge_type}"]["f"])
                else:
                    a2sf_score.append(0)
            
            print(rouge_type)
            print(f"Full Cache Rouge Score: {sum(full_cache_score)/len(full_cache_score)}")
            print(f"H2O Rouge Score: {sum(h2o_score)/len(h2o_score)}")
            print(f"A2SF Rouge Score: {sum(a2sf_score)/len(a2sf_score)}")
        print()
