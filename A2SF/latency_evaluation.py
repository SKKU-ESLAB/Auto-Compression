import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention

for model_name in ["meta-llama/Llama-2-7b-hf"]:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # for policy in ["h2o", "a2sf"]:
    for policy in ["a2sf"]:
        # for input_length, batch_size in [(2048, 2), (1024, 4), (512, 8)]:
        for input_length, batch_size in [(1024, 1)]:
            if model_name == "huggyllama/llama-7b" and input_length == 2048:
                continue
            
            config.scoring_policy = policy
            config.hh_size = math.ceil(input_length*0.1)
            config.recent_size = math.ceil(input_length*0.1)
            if policy == "h2o":
                config.forgetting_factor = 1.0
            elif policy == "a2sf":
                config.forgetting_factor = 0.4

            model = H2OLlamaForCausalLM.from_pretrained(model_name, config=config)
            # model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            torch.cuda.empty_cache()
            model.eval().half().cuda()

            vocab_size = tokenizer.vocab_size

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            result_list = []

            for _ in tqdm(range(10)):
                input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, input_length)).to(model.device)

                starter.record()
                model.generate(input_ids=input_ids, max_length=2*input_length)
                ender.record()
                torch.cuda.synchronize()
                
                for name, m in model.named_modules():
                    if isinstance(m, H2OLlamaAttention):
                        m._clean_cache()

            result_list.append(starter.elapsed_time(ender))
            averaged_result = sum(result_list)/len(result_list)

            print(f"{model_name} {config.scoring_policy} average latency(batch: {batch_size} / prompt : {input_length} / generation : {input_length}) : {averaged_result/1000:.2f}s")