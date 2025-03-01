import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import json
import random

def get_wikitext2(nsamples, seqlen, tokenizer, batch_size=1):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    
    # Generate samples from training set
    trainloader = []
    for _ in range(nsamples):
        inps = torch.zeros((batch_size, seqlen)).long()
        tars = torch.zeros_like(inps)
        for b in range(batch_size):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            inps[b] = inp[0]
            tars[b] = tar[0]

        trainloader.append((inps, tars))

    return trainloader

class KAttentionDataset(Dataset):
    def __init__(self, K, Attention):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass

class ForgetGate(nn.Module):
    def __init__(self, num_head, dim_head):
        super().__init__()
        
        self.forget_gate_weight = nn.Parameter(torch.zeros(num_head, 2*dim_head, 1), requires_grad=True)
        self.forget_gate_bias = nn.Parameter(torch.zeros(1, num_head, 1, 1), requires_grad=True)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, keys):
        last_vector = keys[:, :, -1, :]
        cated_keys = torch.cat((keys, last_vector.unsqueeze(2).expand(-1, -1, keys.size(2), -1)), dim=-1)
        
        forget_factors = torch.matmul(cated_keys, self.forget_gate_weight) + self.forget_gate_bias
        forget_factors = self.sigmoid(forget_factors)
        
        return forget_factors.squeeze(-1)
        

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = AutoConfig.from_pretrained("gpt2")
gate = ForgetGate(config.n_head, int(config.n_embd/config.n_head))

nsamples = 10
seqlen = 512
batch_size = 4
budget = 200

trainloader = get_wikitext2(
    nsamples = nsamples,
    seqlen = seqlen,
    tokenizer = tokenizer,
    batch_size = batch_size
)

criterion = nn.MSELoss()
optimizer = optim.Adam(gate.parameters, lr=0.001)

model.eval().cuda()
gate.cuda()

for epoch in range(100): # epochs
    for li in range(config.n_layer):
        outputs = model(trainloader[epoch][0].to("cuda"), output_attentions=True)
        
        ks = outputs.past_key_values[li][0]
        ats = outputs.attentions[li]
        
        mask = torch.ones((batch_size, config.n_head, seqlen), device="cuda")
        masked_ats = ats.clone()
        
        first_step = True
        scores = torch.zeros((batch_size, config.n_head, seqlen), device="cuda")
        
        for step in range(budget, seqlen-1):
            if first_step:
                for i in range(budget):
                    forget_factor = gate(ks[:, :, :i+1, :])
                    scores += masked_ats[:, :, i, :] * F.pad(forget_factor, (0, seqlen - i - 1))
                first_step = False
            else:
                forget_factor = gate(ks[:, :, :step+1, :])
                scores += masked_ats[:, :, step, :] * F.pad(forget_factor, (0, seqlen - step - 1))
        

            min_idx = torch.argmin(scores + 100 * ( 1 - mask ), dim=-1)
            mask = mask.scatter(-1, min_idx.unsqueeze(-1), 0)
            
            masked_ats[:, :, step+1, :] = masked_ats[:, :, step+1, :] * mask
        
        ats - masked_ats