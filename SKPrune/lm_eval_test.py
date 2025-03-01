import copy
import torch
import argparse
from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks
from transformers import AutoModelForCausalLM

initialize_tasks()

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with different models and tasks.")
    
    parser.add_argument('--task_list', type=str, default="openbookqa,piqa,arc_challenge,arc_easy,mathqa", 
                        help="Comma-separated list of tasks to evaluate.")
    parser.add_argument('--model_list', type=str, default="", 
                        help="Comma-separated list of model names.")
    parser.add_argument('--fewshot_list', type=str, default="1,5", 
                        help="Comma-separated list of fewshot values.")
    
    return parser.parse_args()

args = parse_args()

task_list = args.task_list.split(',')
model_list = ["facebook/opt-2.7b"] + args.model_list.split(',')
fewshot_list = [int(x) for x in args.fewshot_list.split(',')]

lm = huggingface.HFLM("facebook/opt-2.7b", batch_size=16)

for model_name in model_list:
    print(f"model: {model_name}")
    
    if model_name != "facebook/opt-2.7b":
        lm.model.cpu()
        torch.cuda.empty_cache()
        tmp_model = AutoModelForCausalLM.from_pretrained(model_name)
        lm.model.load_state_dict(tmp_model.state_dict())
        lm.model.cuda()
    
    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        results = evaluator.evaluate(lm, tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation"))

        print(evaluator.make_table(results))
        if "groups" in results:
            print(evaluator.make_table(results, "groups"))
        print("=====================================================================================================================================")
