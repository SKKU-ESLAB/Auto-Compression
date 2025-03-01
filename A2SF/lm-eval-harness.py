import copy
import torch
import argparse

from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

initialize_tasks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on various tasks.")
    
    parser.add_argument("--task_list", nargs="+", help="List of tasks to evaluate")
    parser.add_argument("--model_list", nargs="+", help="List of model names")
    parser.add_argument("--fewshot_list", nargs="+", type=int, help="List of fewshot values")
    parser.add_argument("--ratio_list", nargs="+", type=float, help="List of ratio values")

    args = parser.parse_args()

    task_list = args.task_list
    model_list = args.model_list
    fewshot_list = args.fewshot_list
    ratio_list = args.ratio_list

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        full_model = False
        for model_name in model_list:
            
            lm = huggingface.HFLM(model_name, device="cpu", batch_size=16)
            check_point = copy.deepcopy(lm.model.state_dict())
            lm._device = "cuda"
            lm.model.cuda()

            print(f"model: {model_name}")

            for ratio in ratio_list:
                config = {
                    # "Full": (0.0, 1.0, 0.0, 1.0, False),
                    # "STREAMING_LLM": (ratio/2, 0.0, ratio/2, 1.0, False),
                    # "H2O": (0.0, ratio/2, ratio/2, 1.0, False),
                    "A2SF": (0.0, ratio/2, ratio/2, 0.3, False),
                }

                for method, (streaming, selecting, recent, factor, ideal) in config.items():
                        
                    if method != "Full":
                        lm_model(
                            model_name=model_name,
                            lm=lm,
                            check_point=check_point,
                            device="cuda",
                            streaming_ratio=streaming,
                            selecting_ratio=selecting,
                            recent_ratio=recent,
                            forgetting_factor=factor,
                            ideal=ideal,
                        )
                    else:
                        if full_model:
                            continue
                        full_model = True
                    
                    print(f"================={method} streaming : {streaming:.2f} / selecting : {selecting:.2f} / recent : {recent:.2f} / factor : {factor:.2f}=================")
                    
                    results = evaluator.evaluate(lm, tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation"))

                    print(evaluator.make_table(results))
                    if "groups" in results:
                        print(evaluator.make_table(results, "groups"))
                    print("=====================================================================================================================================")
                            
            lm.model.cpu()
