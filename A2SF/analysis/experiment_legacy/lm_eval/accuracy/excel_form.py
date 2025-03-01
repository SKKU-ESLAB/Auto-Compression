import os

path = os.path.dirname(__file__)

with open(os.path.join(path, "excel_form.txt"), "r") as f:
    file = f.readlines()

model = ["LLaMA 2 7B", "LLaMA 7B", "OPT 6.7B", "OPT 2.7B"]
fewshot = ["1", "0"]
model_fewshots = []
for i in model:
    for j in fewshot:
        model_fewshots.append(f"{i} | {j}-shot")
datasets = ["openbookqa", "winogrande", "piqa", "copa", "mathqa", "arc_easy", "arc_challenge"]

final_result = ""
for j in range(len(datasets)):
    for i in range(len(model_fewshots)):
        model_fewshot = model_fewshots[i]
        dataset = datasets[j]
        
        idx = i*7 + j
        
        text = file[idx].strip().split("\t")
        
        result = ""
        
        result += 7*f"{text[0]}\t" + "\n"
        
        for a in range(3):
            for b in range(7):
                result += f"{text[3*b + a + 1]}\t"
            result += "\n"
        
        final_result += f"{model_fewshot} | {dataset}\n"
        final_result += result + "\n"

with open(os.path.join(path,"result.txt"), "w") as f:
    f.write(final_result)