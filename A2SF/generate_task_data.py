import argparse
import json
import os

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor
from lm_eval.tasks import initialize_tasks

initialize_tasks()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_list', nargs="+", type=str, help="List of tasks to evaluate")
    parser.add_argument("--fewshot_list", nargs="+", type=int, help="List of fewshot values")
    args = parser.parse_args()

    seq = 1024
    total_batch = 1
    pe = 'fixed'

    if not os.path.exists("data"):
        os.makedirs("data")

    for task_name in args.task_list:
        for fewshot in args.fewshot_list:
            output_path = os.path.join("data", f"{task_name}-{fewshot}shot.jsonl")
            
            with open(output_path, 'w') as f:
                pass

            class DryRunner:
                def eval(self, batch):

                    with open(output_path, 'a') as f:

                        for text in batch['text']:
                            item = {
                                "prompt": text, 
                            }

                            f.write(json.dumps(item) + '\n')

                    out = {
                        'mask_loss': [1.0] * len(batch),
                        'each_correct': [True] * len(batch),
                    }
                    return out

            t = DryRunner()
            adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")
            results = evaluator.evaluate(adaptor, tasks.get_task_dict([task_name], num_fewshot=fewshot, fewshot_split="validation"))
    print('Finished')