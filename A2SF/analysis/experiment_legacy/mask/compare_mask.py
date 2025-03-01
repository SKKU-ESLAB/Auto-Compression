import os
import numpy as np
import matplotlib.pyplot as plt

def compare_mask(mask1, mask2):
    result = list()
    
    for i in range(mask1.shape[1]):
        mask_a = mask1[0, i, :, :]
        mask_b = mask2[0, i, :, :]
        
        for j in range(mask_a.shape[-2]):
            vec_a = mask_a[j, :]
            vec_b = mask_b[j, :]

            product = np.dot(vec_a, vec_b)
            similar = product/(np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 0.0001)

            result.append(similar)
    
    result = np.array(result)
    
    return np.mean(result)

for dataset in ["winogrande", "piqa", "openbookqa", "arc_e", "mathqa"]:
    
    mask_list = {
        "ideal": [],
        # "local": [],
        "h2o": [],
        "a2sf_000": [],
        "a2sf_010": [],
        "a2sf_050": []
    }
    
    print(dataset)
    for layer in range(32):
        
        dir_path = os.path.dirname(__file__)
        attention_path = os.path.join(dir_path, dataset, "no_pruning", f"{layer}.npy")

        ideal_mask = np.load(attention_path) # 1, 32, 25, 25
        
        for mask_name in mask_list.keys():
            mask_path = os.path.join(dir_path, dataset, mask_name, f"{layer}.npy")
            mask = np.load(mask_path) # 1, 32, 25, 25
            
            if mask_name == "local":
                mask = np.triu(mask, -(np.count_nonzero(mask[0,0,-1])-2))
            
            mask_list[mask_name].append(compare_mask(ideal_mask, mask))

    plt.figure(figsize=(7.5,6))
    for a, b in mask_list.items():
        if b != []:
            plt.plot(b, label=a)
            
            mean = sum(b)/len(b)
            print(f"{mean:.3f}")

    plt.legend()
    plt.title("Average Cosine Similarity of Heads")
    plt.savefig(os.path.join(dir_path, dataset, "Similarity.png"))
    plt.xlabel("Layer Number")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.close()