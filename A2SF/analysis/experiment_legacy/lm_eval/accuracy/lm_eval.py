import re

with open("lm_eval.txt", "r") as f:
    file = f.readlines()

result = list()

for i in file:
    if "|acc_norm" not in i:
        if "|acc" in i:
            values = re.findall(r"\|acc\s*\|\s*(\d+\.\d+)", i)
            result.append(f"{values[0]}\n")

str_joined = "".join(result)

numbers = re.findall(r"\d+\.\d+", str_joined)

grouped_list = [numbers[i:i+7] for i in range(0, len(numbers), 7)]

final_text = ""

for i in range(len(grouped_list[0])):
    tmp = ""
    for j in range(len(grouped_list)):
        try:
            tmp += f"{grouped_list[j][i]}\t"
        except:
            import pdb; pdb.set_trace()
    final_text += f"{tmp}\n"

with open("test.txt", "w") as f:
    f.write(final_text)