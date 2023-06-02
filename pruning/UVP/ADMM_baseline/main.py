import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.models as models
import numpy as np

import wandb

from models.mobilenet_v1 import MobileNetV1
import math
from utils import get_admm_loss, initialize_perm_list, initialize_Z_and_U, update_X, update_Z, update_U, print_prune, apply_prune
from data import get_dataset
from get_model_from_gluoncv import get_pretrained_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Added
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--lr-scheduler', default='multistep', type=str)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--warmup-lr', default=0, type=float)
parser.add_argument('--vector-size', default=1, type=int)
parser.add_argument('--unaligned', dest='unaligned', action='store_true')
parser.add_argument('--width-mult', default=1.0, type=float)
parser.add_argument('--sparsity-method', default='gt', choices=['gt', 'uniform'])
parser.add_argument('--target-sparsity', default=0.8, type=float)
parser.add_argument('--rho', default=0.001, type=float)
parser.add_argument('--group-norm', default='l1', choices=['l1', 'l2'])

parser.add_argument('--cp-alpha', default=0.1, type=float)
parser.add_argument('--cp-beta', default=0.8, type=float)

parser.add_argument('--name', default="mobilenet_v1", type=str)
parser.add_argument('--admm-epochs', default=100, type=int)
parser.add_argument('--ft-epochs', default=100, type=int)
parser.add_argument('--cp', dest='cp', action='store_true')
parser.add_argument('--repeat', dest='repeat', action='store_true')

parser.add_argument('--cp-ft', dest='cp_ft', action='store_true')


best_acc1 = 0


def cosine_calc_learning_rate(args, total_epochs, epoch, batch=0, nBatch=None):
    T_total = total_epochs * nBatch
    T_cur = epoch * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    return lr


def cosine_adjust_learning_rate(args, optimizer, total_epochs, epoch, batch=0, nBatch=None):
    new_lr = cosine_calc_learning_rate(args, total_epochs, epoch, batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def cosine_warmup_adjust_learning_rate(args, optimizer, T_total, nBatch, epoch,
                                       batch=0, warmup_lr=0):
    T_cur = epoch * nBatch + batch + 1
    new_lr = T_cur / T_total * (args.lr - warmup_lr) + warmup_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def main():
    args = parser.parse_args()
    groups = f"{args.arch}_{args.width_mult}_ts{args.target_sparsity}_admm{args.admm_epochs}_ft{args.ft_epochs}_v{args.vector_size}"
    if args.unaligned:
        groups = groups + "_u"
    else:
        groups = groups + "_a"
    if args.cp:
        groups = groups + "_cp"
    if args.repeat:
        groups = groups + "_repeat"
    if args.cp_ft:
        groups = groups + "_cpft"
    args.name = groups
    wandb.init(project="UVP", name=groups)
    wandb.config.update(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.arch == "mobilenet_v1":
        print("=> creating model 'mobilenet_v1'")
        if args.dataset == "imagenet":
            num_classes = 1000
            input_size = 224
        elif args.dataset == "cifar10":
            num_classes = 10
            input_size = 32
        elif args.dataset == "cifar100":
            num_classes = 100
            input_size = 32
        model = MobileNetV1(num_classes=num_classes,
                            input_size=input_size,
                            width_mult=args.width_mult)

    elif args.arch == "mobilenet_v2":
        model = models.mobilenet_v2(width_mult=args.width_mult, dropout=0.0)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    if args.pretrained:
        pretrained_model = f"{args.arch}_{args.width_mult}.pth.tar"
        if not os.path.isfile(pretrained_model):
            get_pretrained_model(args.arch, args.width_mult)
        print("=> loading checkpoint '{}'".format(pretrained_model))
        if args.gpu is None:
            checkpoint = torch.load(pretrained_model)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(pretrained_model, map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])

    if not os.path.isdir(args.name):
        os.makedirs(args.name)

    admm_epoch = 0
    ft_epoch = 0
    perm_list = None
    mask = None
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            admm_epoch = checkpoint['admm_epoch']
            ft_epoch = checkpoint['ft_epoch']
            best_acc1 = checkpoint['best_acc1']
            perm_list = checkpoint['perm_list']
            mask = checkpoint['mask']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader, val_loader = get_dataset(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if perm_list is None:
        perm_list = initialize_perm_list(model, args)

    Z, U = initialize_Z_and_U(model, args)
    X = update_X(model, args)
    Z, score_diff_dict = update_Z(X, U, args, perm_list, args.cp)
    U = update_U(U, X, Z)
    print(score_diff_dict)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(admm_epoch, args.admm_epochs):
        # train for one epoch
        train_log = admm_train(train_loader, model, criterion, optimizer, epoch, args, Z, U, scaler)
        X = update_X(model, args)
        Z, score_diff_dict = update_Z(X, U, args, perm_list, args.repeat)
        U = update_U(U, X, Z)
        if args.repeat:
            print(score_diff_dict)

        # evaluate on validation set
        acc1, val_log = validate(val_loader, model, criterion, args)

        wandb.log({**train_log, **val_log, **score_diff_dict}, step=epoch+1)

        save_checkpoint({
            'admm_epoch': epoch + 1,
            'ft_epoch': 0,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'perm_list': perm_list,
            'mask': None,
        }, False, dirs=args.name, filename="admm.pth.tar")

    if mask is None:
        mask = apply_prune(model, args, perm_list)
        print_prune(model, args)
        acc1 = validate(val_loader, model, criterion, args)

        save_checkpoint({
            'admm_epoch': args.admm_epochs,
            'ft_epoch': 0,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'perm_list': perm_list,
            'mask': mask,
        }, False, dirs=args.name, filename="post_admm.pth.tar")

    # finetuning stage
    best_acc1 = 0
    optimizer.param_groups[0]['weight_decay'] = 0.
    for epoch in range(ft_epoch, args.ft_epochs):
        # train for one epoch
        train_log = finetune(train_loader, model, criterion, optimizer, epoch, args, mask, scaler)

        # evaluate on validation set
        acc1, val_log = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        wandb.log({**train_log, **val_log}, step=args.admm_epochs+epoch+1)

        save_checkpoint({
            'admm_epoch': args.admm_epochs,
            'ft_epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'perm_list': perm_list,
            'mask': mask,
        }, is_best, dirs=args.name, filename='finetune.pth.tar')
        print("Best accuracy: {:.3f}".format(best_acc1.item()))

def admm_train(train_loader, model, criterion, optimizer, epoch, args, Z, U, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cls_losses = AverageMeter('cls_Loss', ':.4e')
    admm_losses = AverageMeter('admm_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cls_losses, admm_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            cls_loss = criterion(output, target)
            admm_loss = get_admm_loss(args, model, Z, U)
            loss = cls_loss + admm_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        cls_losses.update(cls_loss.item(), images.size(0))
        admm_losses.update(admm_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    log_dict = {
        "lr": optimizer.param_groups[0]['lr'],
        "train_loss": losses.avg,
        "admm_cls_loss": cls_losses.avg,
        "admm_admm_loss": admm_losses.avg,
        "train_top1": top1.avg,
        "train_top5": top5.avg,
    }
    return log_dict


def finetune(train_loader, model, criterion, optimizer, epoch, args, mask, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.lr_scheduler == 'cosine':
            nBatch = len(train_loader)
            if epoch < args.warmup_epochs:
                cosine_warmup_adjust_learning_rate(
                    args, optimizer, args.warmup_epochs * nBatch, nBatch,
                    epoch, i, args.warmup_lr)
            else:
                cosine_adjust_learning_rate(
                    args, optimizer, args.ft_epochs, epoch - args.warmup_epochs, i, nBatch)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    log_dict = {
        "lr": optimizer.param_groups[0]['lr'],
        "train_loss": losses.avg,
        "train_top1": top1.avg,
        "train_top5": top5.avg,
    }
    return log_dict


def train(train_loader, model, criterion, optimizer, epoch, args, scaler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.lr_scheduler == 'cosine':
            nBatch = len(train_loader)
            if epoch < args.warmup_epochs:
                cosine_warmup_adjust_learning_rate(
                    args, optimizer, args.warmup_epochs * nBatch,
                    nBatch, epoch, i, args.warmup_lr)
            else:
                cosine_adjust_learning_rate(
                    args, optimizer, args.ft_epochs, epoch - args.warmup_epochs, i, nBatch)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    log_dict = {
        "lr": optimizer.param_groups[0]['lr'],
        "train_loss": losses.avg,
        "train_top1": top1.avg,
        "train_top5": top5.avg,
    }
    return log_dict


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    log_dict = {
        "val_loss": losses.avg,
        "val_top1": top1.avg,
        "val_top5": top5.avg,
    }

    return top1.avg, log_dict


def save_checkpoint(state, is_best, dirs=".", filename='checkpoint.pth.tar'):
    filename = os.path.join(dirs, filename)
    torch.save(state, filename)
    if filename != 'checkpoint.pth.tar':
        shutil.copyfile(filename, os.path.join(dirs, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(filename, os.path.join(dirs, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
