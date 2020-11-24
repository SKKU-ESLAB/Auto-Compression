import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from train_argument import parser, print_args

from time import time
from utils import *
from models import *
from trainer import *


def main(args):
    save_folder = args.affix

    #log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    #makedirs(log_folder)
    makedirs(model_folder)

    #setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(model_folder, 'train', 'info')
    print_args(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    if "WideResNet" in args.model:
        depth_ = args.model.split('-')[1]
        factor_ = args.model.split('-')[2]
        net = WideResNet(depth=depth_, num_classes=args.dataset == 'cifar10' and 10 or 100, widen_factor=factor_)
        if args.mask:
            net = GroupedMaskedWideResNet(depth=depth_, num_classes=args.dataset == 'cifar10' and 10 or 100, widen_factor=factor_)
    
    elif 'MobileNetV3' in args.model:# e.g. args.model == MobileNetv3-small-0.75
        mode_ = args.model.split('-')[1]
        width_ = args.model.split('-')[2]
        if mode_ =='small':
            net = mobilenetv3_small(prune_type=args.prune_type, width_mult=width_)
        elif mode_ == 'large':
            net = mobilenetv3_large(prune_type=args.prune_type, width_mult=width_)
        
        # load pretrain weights
        net.load_state_dict(torch.load(f'./saved_models/mobilenetv3-{mode_}-{width_}.pth'))
        print('load pretrain weights!')

    elif "ResNet" in args.model :
        depth_ = args.model.split('-')[1]
        if 'cifar' in args.dataset:
            
            if args.mask: # pruning baseline
                res_dict2 = { '20' : resnet20(num_classes=10, prune_type = args.prune_type),
                        '32' : resnet32(num_classes=10, prune_type = args.prune_type),
                        '44': resnet44(num_classes=10, prune_type = args.prune_type),
                        '56': resnet56(num_classes=10, prune_type = args.prune_type)
                        }
                    
                net = res_dict2[depth_]

            else: # dense baseline
                res_dict = {'18': resnet18(pretrained=False, progress=True, prune_type = args.prune_type),
                        '34': resnet34(pretrained=False, progress=True, prune_type = args.prune_type),
                        '50': resnet50(pretrained=False, progress=True, prune_type = args.prune_type),
                        '101': resnet101(pretrained=False, progress=True, prune_type = args.prune_type)
                        }

                net = res_dict[depth_]

        # pretrain load
        state_dict = torch.load(f'./saved_models/[resnet{depth_}_pretrain]/best_acc_model.pth')
        temp_dict = OrderedDict()
        for name,parms in net.named_parameters():
            temp_dict[name] = parms
        
        for name in state_dict:
            temp_dict[name] = state_dict[name]

        net.load_state_dict(temp_dict)
        print('load pretrain weights!')

    
    #
    net.to(device)

    trainer = Trainer(args, logger)

    loss = nn.CrossEntropyLoss()


    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if args.dataset == 'cifar10':
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
            batch_size=100, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=100, shuffle=True, **kwargs)

    elif args.dataset =='imagenet':
        train_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('../../../imagenet-torchvision/', split='train',download=False, transform=transforms.Compose([
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
        #
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('../../../imagenet-torchvision/', split='val',download=False, transform=transforms.Compose([
                            transforms.Resize(int(224/0.875)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])),
            batch_size=args.batch_size, shuffle=False, num_workers=4)


    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)



if __name__ == '__main__':
    args = parser()
    #print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
