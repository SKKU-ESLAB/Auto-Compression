import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.dirname(__file__)

file = open(os.path.join(path, "result.txt")).readlines()

labels = ["Full", "Local", "H2O", "A2SF"]
x_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

for idx in range(7*2*4):

    line_num = len(labels)+2
    data = file[line_num*idx:line_num*(idx+1)-1]

    data_list = []
    for i in range(1, len(labels)+1, 1):
        tmp = [j.strip() for j in data[i].strip().split("\t")]
        tmp = list(map(float, tmp))
        data_list.append(tmp)

    title = data[0].strip()
    dataset = title.split("|")[-1].strip()
    y_values = np.array(data_list)

    plt.figure(figsize=(5, 4))
    plt.title(title, fontsize=16)
    for i, label in enumerate(labels):
        if label == "A2SF":
            max_val_index = np.argmax(y_values[i])
            plt.plot(x_values, y_values[i], label=label, marker="o", markersize=5)
            plt.plot(x_values[max_val_index], y_values[i][max_val_index], marker="*", markersize=10)
        else:
            plt.plot(x_values, y_values[i], linestyle="dashed", label=label)

    plt.xlabel("Forget Factor", fontsize=16)
    plt.xticks(fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.grid(True, linestyle="dashed")
    plt.tight_layout()
    
    dir_path = os.path.join(path, "plot", dataset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, f"{title}.png"))
    plt.close()