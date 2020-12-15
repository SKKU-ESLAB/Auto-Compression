## Models

|              | FLOPs     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| ResNet20  |  M     |   M     | -     | -                                                            |
| ResNet32  |  M     |     M     | -     | -                                                            |
| ResNet44  |  M     |  M     | -     | - |
| ResNet56  |  M     |  M     |  -        | - |


## Training Example

```python
# resnet20 / global threshold / group-level pruning / sparsity:0.5
python3 train.py --model=ResNet-20 --prune_method=global --prune_type=group --sparsity=0.5 
--save_folder=resnet18_model_path --batch_size=64 --epochs=160

```
## Data Pre-processing

I used the following code for data pre-processing on CIFAR10 & CIFAR100:

```python
# cifar10
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
# cifar100
normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])

# train
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
    batch_size=batch_size, shuffle=True, num_workers=n_worker)

# validation
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])),
    batch_size=batch_size, shuffle=False, num_workers=n_worker)
            

```

## Optimizer & Scheduler

```python
epoch = 160

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum= 0.9, weight_decay= 1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
            

```
