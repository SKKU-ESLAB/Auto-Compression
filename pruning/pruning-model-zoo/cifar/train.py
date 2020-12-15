import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.nn.utils.prune as prune
from train_argument import parser, print_args

from time import time
from utils import *
from models.resnet import *
from trainer import *


def main(args):
    save_folder = args.save_folder

    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(model_folder)

    setattr(args, 'model_folder', model_folder)

    logger = create_logger(model_folder, 'train', 'info')
    print_args(args, logger)

    
    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    if "ResNet" in args.model :
        depth_ = args.model.split('-')[1]
        
        # when it is dst, resnet block is special
        if args.prune_method!='dst' : p_type=None

        res_dict = { '20' : resnet20(num_classes=int(args.dataset.split('-')[1]), prune_type = p_type),
                '32' : resnet32(num_classes=int(args.dataset.split('-')[1]), prune_type = p_type),
                '44': resnet44(num_classes=int(args.dataset.split('-')[1]), prune_type = p_type),
                '56': resnet56(num_classes=int(args.dataset.split('-')[1]), prune_type = p_type)
                }
            
        net = res_dict[depth_]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # set trainer
    trainer = Trainer(args, logger)

    # loss
    loss = nn.CrossEntropyLoss()

    # dataloader
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.dataset == 'cifar-10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar-100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    # optimizer & scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    # pruning
    if args.prune_method=='global':
        if args.prune_type=='group':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]==3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    append_size = 4 - tmp_pruned.shape[1] % 4
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, 4)
                    tmp_pruned = tmp_pruned.abs().mean(2, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.2
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]==3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    append_size = 4 - tmp_pruned.shape[1] % 4
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, 4)
                    tmp_pruned = tmp_pruned.abs().mean(2, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmp_pruned = tmp_pruned.ge(threshold)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size)

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)
        elif args.prune_type =='filter':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]==3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.5
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]==3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmp_pruned = tmp_pruned.ge(threshold)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size)

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)        

        print(f'model pruned!!(sparsity : {args.sparsity : .2f}, prune_method : {args.prune_method}, prune_type : {args.prune_type}-level pruning')
    
    elif args.prune_method=='uniform':
        assert False, 'uniform code is not ready'

    elif args.prune_method =='dst':
        print(f'model pruned!!(prune_method : {args.prune_method}, prune_type : {args.prune_type}-level pruning')

    
    # Training
    trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)



if __name__ == '__main__':
    args = parser()
    #print_args(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
