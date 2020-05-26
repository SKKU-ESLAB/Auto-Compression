import torch
import torch.nn as nn
from scipy.stats import norm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bit', type=int, default='4')
parser.add_argument('--prune', action='store_true')
parser.add_argument('--data', action='store', default=None)
args = parser.parse_args()

m = torch.load(args.data)
b_min = 2
b_max = 4
count = 0
index = []
params = 0
num_layer = 0
for i in m.keys():
    if 'weight' in i:
            #if count % 2 == 1 and count < 1000:
            #    count += 1
            #    continue
            if count == 7:
                break
            print(i)
            w = m[i]
            ch = w.shape[0]
            w_numpy = w.cpu().numpy()
            length = w_numpy.shape[0]
            w_numpy = w_numpy.reshape(w_numpy.shape[0], -1)
            size = w_numpy.shape[1]
            max = np.max(np.absolute(w_numpy), axis=1)
            min = np.min(np.absolute(w_numpy), axis=1)
            interval = max - min
            w_sum = np.sum(w_numpy*w_numpy)
            ch_sum = np.sum(w_numpy*w_numpy, axis=1)
            ch_sum = ch_sum / w_sum
            tmp = interval * ch_sum 
            tmp = (tmp - tmp.min())/(tmp.max()-tmp.min()) * (b_max -b_min) + b_min
            tmp = np.round(tmp)
            tmp2 = tmp*size
            params += np.sum(tmp2)
            sorted_tmp = tmp.argsort()
            idx = 0
            last = 0
            a = []
            for i in range(b_max - b_min + 1):
                if (i+b_min) == b_max:
                    index.append(sorted_tmp[last:].tolist())
                    break
                else:
                    while idx < ch:
                        if (i+b_min) != tmp[sorted_tmp[idx]]:
                            break
                        idx += 1
                index.append(sorted_tmp[last:idx].tolist())
                last = idx
                if idx == ch:
                    break
            #index.append(a)
            count+=1
            num_layer += 1
with open('index.txt', 'w') as f:
    for item in index:
        f.write("%s\n" % item)
