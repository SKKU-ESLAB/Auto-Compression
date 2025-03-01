import re
import os

dir_path = os.path.dirname(__file__)

with open(os.path.join(dir_path, "factor_030.txt"), "r") as f:
    file = f.readlines()

result = list()

for i in file:
    if "|acc_norm" not in i:
        if "|acc" in i:
            values = re.findall(r"\|acc\s*\|\s*(\d+\.\d+)", i)
            result.append(f"{values[0]}\n")

str_joined = "".join(result)

numbers = re.findall(r"\d+\.\d+", str_joined)

num_column = 10
num_dataset = 7

grouped_list = [numbers[i:i+num_dataset] for i in range(0, len(numbers), num_dataset)]

num_data = len(grouped_list)

final_text = ""

for i in range(int(num_data/num_column)):
    for j in range(num_dataset):
        tmp = ""
        for k in range(num_column):
            tmp += f"{grouped_list[i*num_column+k][j]}\t"
        final_text += f"{tmp}\n"

with open(os.path.join(dir_path, "excel_form.txt"), "w") as f:
    f.write(final_text)