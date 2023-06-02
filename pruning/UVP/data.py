import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from auto_augment import AutoAugment, Cutout
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(args):
    # Data loading code
    if args.dataset == "imagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    #AutoAugment(),
                    #Cutout(),
                    transforms.ToTensor(),
                    normalize,
                    ]))

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, persistent_workers=True,
                worker_init_fn=seed_worker)

        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                    ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, persistent_workers=True)
    elif args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2612])

        train_dataset = datasets.CIFAR10(
                args.data,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    #AutoAugment(),
                    #Cutout(),
                    transforms.ToTensor(),
                    normalize,
                    ]))

        val_dataset = datasets.CIFAR10(
                args.data,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ]))

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])

        train_dataset = datasets.CIFAR100(
                args.data,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    #AutoAugment(),
                    #Cutout(),
                    transforms.ToTensor(),
                    normalize,
                    ]))

        val_dataset = datasets.CIFAR100(
                args.data,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ]))

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader
