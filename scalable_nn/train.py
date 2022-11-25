import importlib
import os
import time
import random
import math
import copy

import torch
from torchvision import datasets, transforms
import numpy as np

from utils.config import FLAGS
from utils.meters import AverageMeter, accuracy
from utils.loss_ops import CrossEntropyLossSoft, CrossEntropyLossSmooth
from utils.model_profiling import model_profiling
from models.group_level_ops import *

from collections import defaultdict
from string import Template
import wandb
import ComputePostBN

def get_model():
    # import NN model
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    if getattr(FLAGS, 'dataparallel', False):
        model_wrapper = torch.nn.DataParallel(model).cuda()
    else:
        model_wrapper = model.cuda()
    return model, model_wrapper

def data_transforms():
    ## import data_trasnforms depends on dataset
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cifar':
        if getattr(FLAGS, 'normalize', False):
            mean = FLAGS.mean
            std = FLAGS.std
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms

def dataset(train_transforms, val_transforms, test_transforms):
    """get dataset for classification"""
    if FLAGS.dataset == 'imagenet1k_basic':
        train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'CIFAR10':
        train_set = datasets.CIFAR10(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        val_set = datasets.CIFAR10(
            FLAGS.dataset_dir,
            train=False,
            transform = val_transforms,
            download=True)
        test_set = None
    elif FLAGS.dataset == 'CIFAR100':
        train_set = datasets.CIFAR100(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        val_set = datasets.CIFAR100(
            FLAGS.dataset_dir,
            train=False,
            transform = val_transforms,
            download=True)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset_dir))
    return train_set, val_set, test_set

def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    # infer batch size
    if not getattr(FLAGS, 'batch_size', False):
        raise ValueError('batch size is not defined')

    batch_size = FLAGS.batch_size
        
    if FLAGS.data_loader in ['imagenet1k_basic', 'cifar']:
        if getattr(FLAGS, 'distributed', False):
            if FLAGS.test_only:
                train_sampler = None
            else:
                train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(val_set)
        else:
            train_sampler = None
            val_sampler = None
        train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                drop_last=getattr(FLAGS, 'drop_last', False))
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            num_workers=FLAGS.data_loader_workers,
            drop_last=getattr(FLAGS, 'drop_last', False))
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    if train_loader is not None:
        FLAGS.data_size_train = len(train_loader.dataset)
    if val_loader is not None:
        FLAGS.data_size_val = len(val_loader.dataset)
    if test_loader is not None:
        FLAGS.data_size_test = len(test_loader.dataset)
    return train_loader, val_loader, test_loader

def profiling(model, use_cuda):
    print('Start model profiling, use_cuda: {}.'.format(use_cuda))    
    flops, params = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size, use_cuda=use_cuda,
            verbose=getattr(FLAGS, 'profiling_verbose', True))
    return flops, params

def get_lr_scheduler(optimizer):
    """get learning rate"""
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - warmup_epochs) / num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = (
                1. + math.cos(
                    math.pi * (i - warmup_epochs) / num_epochs)) / 2.
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                'Learning rate scheduler {} is not yet implemented.'.format(
                    FLAGS.lr_scheduler))
    return lr_scheduler

def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0

            '''
            if getattr(FLAGS, 'gp_training', False):
                weight_decay = 0
            '''

            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def lr_schedule_per_iteration(optimizer, epoch, batch_idx=0):
    """ function for learning rate scheuling per iteration """
    warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    num_epochs = FLAGS.num_epochs - warmup_epochs
    iters_per_epoch = FLAGS.data_size_train / FLAGS.batch_size
    current_iter = epoch * iters_per_epoch + batch_idx + 1
    if getattr(FLAGS, 'lr_warmup', False) and epoch < warmup_epochs:
        linear_decaying_per_step = FLAGS.lr/warmup_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_iter * linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'linear_decaying':
        linear_decaying_per_step = FLAGS.lr/num_epochs/iters_per_epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] -= linear_decaying_per_step
    elif FLAGS.lr_scheduler == 'cosine_decaying':
        mult = (
            1. + math.cos(
                math.pi * (current_iter - warmup_epochs * iters_per_epoch)
                / num_epochs / iters_per_epoch)) / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = FLAGS.lr * mult
    else:
        pass
    
def run_one_epoch(model, loader, criterion, optimizer, epoch, phase='train', soft_criterion=None, eval_width=None, eval_density=None):
    top1_label = "width=${width}, density=${density} ${phase} / top1"
    loss_label = "width=${width}, density=${density} ${phase} / loss"
    
    top1_fmt = Template(top1_label)
    loss_fmt = Template(loss_label)
    
    top1_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)
    
    t_start = time.time()
    
    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    train = phase == 'train'
    if train:
        model.train()
        gradient_accum = getattr(FLAGS, 'gradient_accum', 1)
    else:
        model.eval()
        gradient_accum = 1
        if phase == 'cal':
            model.apply(bn_calibration_init)
    _density_list = copy.deepcopy(FLAGS.density_list)
    max_density = max(_density_list)
    min_density = min(_density_list)
    
    for batch_idx, (input, target) in enumerate(loader):
        if phase == 'cal':
            if batch_idx == len(loader):
                break
            #if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
            #    break
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        batch_size = input.size(0)
        # train part
        if train:
            # Dynamic Sparse Train part
            if getattr(FLAGS, 'DST_TRAIN', False):
                # sandwish learning rule
                num_sample = getattr(FLAGS, 'num_samples_training', 2) - 2
                width_mult_train = [FLAGS.width_mult] * (num_sample + 2)
                density_train = [max_density, min_density] + random.sample(_density_list, num_sample)
                for width_mult, density in zip(width_mult_train, density_train):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    model.apply(lambda m: setattr(m, 'density', density))
                    if FLAGS.pruner == 'local':
                        model.apply(conv_change_mask)
                    elif FLAGS.pruner == 'global':
                        global_pruning_update(model, density)
                    elif FLAGS.pruner == 'global_normal':
                        global_normal_pruning_update(model, density)
                    
                    output = model(input)
                    if density == max_density:
                        loss = torch.mean(criterion(output, target)) / gradient_accum
                        soft_target = torch.nn.functional.softmax(output, dim=1)
                        top1, top5 = accuracy(output, target, topk=(1, 5))
                        
                        top1_meters[top1_fmt.substitute(width=width_mult, density=density, phase=phase)].update(top1.item(), batch_size)
                        loss_meters[loss_fmt.substitute(width=width_mult, density=density, phase=phase)].update(loss.item(), batch_size)
                    else:
                        if getattr(FLAGS, 'IPKD', False):
                            loss = torch.mean(soft_criterion(output, soft_target.detach())) / gradient_accum
                            top1 = accuracy(output, target)[0]
                            top1_meters[top1_fmt.substitute(width=width_mult, density=density, phase=phase)].update(top1.item(), batch_size)
                            loss_meters[loss_fmt.substitute(width=width_mult, density=density, phase=phase)].update(loss.item(), batch_size)
                        else:
                            loss = torch.mean(criterion(output, target)) / gradient_accum
                            top1 = accuracy(output, target)[0]
                            top1_meters[top1_fmt.substitute(width=width_mult, density=density, phase=phase)].update(top1.item(), batch_size)
                            loss_meters[loss_fmt.substitute(width=width_mult, density=density, phase=phase)].update(loss.item(), batch_size)
                    loss.backward()
            
            # Single Training
            else:
                if batch_idx == 0:
                    print('Single Model Training')
                width_mult = FLAGS.width_mult
                density = FLAGS.density_mult

                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                model.apply(lambda m: setattr(m, 'density', density))
                output = model(input)
                loss = torch.mean(criterion(output, target))
                top1, top5 = accuracy(output, target, topk=(1, 5))
                top1_meters[top1_fmt.substitute(width=width_mult, density=density, phase=phase)].update(top1.item(), batch_size)
                loss_meters[loss_fmt.substitute(width=width_mult, density=density, phase=phase)].update(loss.item(), batch_size)
                loss.backward()
            if ((batch_idx + 1) % gradient_accum == 0) or (batch_idx + 1 == len(loader)): 
                optimizer.step()
                for param in model.parameters():
                    param.grad = None
            lr_schedule_per_iteration(optimizer, epoch, batch_idx)
        
        # evaluation
        else:
            assert (eval_width is not None) and (eval_density is not None), 'no train need config'
            output = model(input)
            loss = torch.mean(criterion(output, target))
            top1, top5 = accuracy(output, target, topk=(1, 5))
            top1_meters[top1_fmt.substitute(width=eval_width, density=eval_density, phase=phase)].update(top1.item(), batch_size)
            loss_meters[loss_fmt.substitute(width=eval_width, density=eval_density, phase=phase)].update(loss.item(), batch_size)
            
    if phase != 'cal':
        print(
            '{:.1f}s\t{}\t{}/{}: '.format(
            time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
            ', '.join('{}: {:.3f}\n'.format(k, v.avg) for k, v in top1_meters.items()))
    if phase in ['train', 'val']:
        #log wandb
        log = {}
        for metric in [top1_meters, loss_meters]:
            for what, meter in metric.items():
                log[what] = meter.avg
        wandb.log(log, step=epoch)
        
    result = sum([v.avg for k, v in top1_meters.items()]) / len(top1_meters)
    return result

def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # seed
    set_random_seed()

    model, model_wrapper = get_model()
    
    if getattr(FLAGS, 'label_smoothing', 0):
        criterion = CrossEntropyLossSmooth(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if getattr(FLAGS, 'IPKD', False):
        soft_criterion = CrossEntropyLossSoft(reduction='none')
    else:
        soft_criterion = None

    # check pretrained
    if getattr(FLAGS, 'pretrained', False):
        # load all tensors onto the CPU
        checkpoint = torch.load(
            FLAGS.pretrained, map_location=lambda storage, loc: storage)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                print('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        print('Loaded model {}.'.format(FLAGS.pretrained))

    optimizer = get_optimizer(model_wrapper)
    
    # check resume training
    if os.path.exists(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(
            os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'),
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=False)
        model_wrapper = torch.nn.DataParallel(model).cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = get_lr_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        print('Loaded checkpoint {} at epoch {} / best_val={}.'.format(
            FLAGS.log_dir, last_epoch, best_val))
    else:
        lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = -1.0
        # if start from scratch, print model and do profiling
        print(model_wrapper)
        if getattr(FLAGS, 'profiling', False):
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)
            if getattr(FLAGS, 'profiling_only', False):
                return

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)
    
    if getattr(FLAGS, 'test_only', False):
        print('start test')
        with torch.no_grad():
            density_mult_eval = FLAGS.density_list
            width_mult_eval = [FLAGS.width_mult] * len(FLAGS.density_list)
            for width_eval, density_eval in zip(width_mult_eval, density_mult_eval):
                model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_eval))
                model_wrapper.apply(lambda m: setattr(m, 'density', density_eval))
                if FLAGS.pruner == 'local':
                    model_wrapper.apply(conv_change_mask)
                elif FLAGS.pruner == 'global':
                    global_pruning_update(model_wrapper, density_eval)
                elif FLAGS.pruner == 'global_normal':
                    global_normal_pruning_update(model_wrapper, density_eval)
                
                model_wrapper = ComputePostBN.ComputeBN(model_wrapper, train_loader)
                run_one_epoch(model_wrapper, val_loader, criterion, optimizer, -1, phase='val', 
                                eval_width=width_eval, eval_density=density_eval)
        return
        
    print('Start group-level pruning version training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        run_one_epoch(model_wrapper, train_loader, criterion, optimizer, epoch, phase='train', soft_criterion=soft_criterion)
        
        acc_list = []
        if epoch % 10 == 1:
            with torch.no_grad():
                density_mult_eval = FLAGS.density_list
                width_mult_eval = [FLAGS.width_mult] * len(density_mult_eval)
                for width_eval, density_eval in zip(width_mult_eval, density_mult_eval):
                    model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_eval))
                    model_wrapper.apply(lambda m: setattr(m, 'density', density_eval))
                    if FLAGS.pruner == 'local':
                        model_wrapper.apply(conv_change_mask)
                    elif FLAGS.pruner == 'global':
                        global_pruning_update(model_wrapper, density_eval)
                    elif FLAGS.pruner == 'global_normal':
                        global_normal_pruning_update(model_wrapper, density_eval)
                
                    model_wrapper = ComputePostBN.ComputeBN(model_wrapper, train_loader)
                    acc_list.append(run_one_epoch(model_wrapper, val_loader, criterion, optimizer, epoch, phase='val', 
                                eval_width=width_eval, eval_density=density_eval))
        else:
            with torch.no_grad():
                density_mult_eval = [1.0, 0.10]
                width_mult_eval = [FLAGS.width_mult] * 2
                for width_eval, density_eval in zip(width_mult_eval, density_mult_eval):
                    model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_eval))
                    model_wrapper.apply(lambda m: setattr(m, 'density', density_eval))
                    if FLAGS.pruner == 'local':
                        model_wrapper.apply(conv_change_mask)
                    elif FLAGS.pruner == 'global':
                        global_pruning_update(model_wrapper, density_eval)
                    elif FLAGS.pruner == 'global_normal':
                        global_normal_pruning_update(model_wrapper, density_eval)
                
                    model_wrapper = ComputePostBN.ComputeBN(model_wrapper, train_loader)
                    acc_list.append(run_one_epoch(model_wrapper, val_loader, criterion, optimizer, epoch, phase='val', 
                                eval_width=width_eval, eval_density=density_eval))
        acc = sum(acc_list)/len(acc_list)
        if  acc > best_val:
            best_val = acc
            if not os.path.exists(FLAGS.log_dir):
              os.makedirs(FLAGS.log_dir)
            if getattr(FLAGS, 'dataparallel', False):
                torch.save({
                    'model' : model_wrapper.module.state_dict(),
                    },
                    os.path.join(FLAGS.log_dir, 'best_model.pt')
                    )
            else:
                torch.save({
                    'model' : model_wrapper.state_dict(),
                    },
                    os.path.join(FLAGS.log_dir, 'best_model.pt')
                    )
            print('New best validation top1_meters error {:.3f}'.format(best_val))
        
        if getattr(FLAGS, 'dataparallel', False):
            torch.save(
                {
                    'model': model_wrapper.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                    'best_val': best_val,
                },
                os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))
        else:
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'last_epoch': epoch,
                    'best_val': best_val,
                },
                os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))           
        
        with torch.no_grad():
            if epoch % 5 == 1:
                density_mult_test = FLAGS.density_list
                width_mult_test = [FLAGS.width_mult] * len(density_mult_test)
                
                for width_eval, density_eval in zip(width_mult_test, density_mult_test):
                    model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_eval))
                    model_wrapper.apply(lambda m: setattr(m, 'density', density_eval))
                    if FLAGS.pruner == 'local':
                        model_wrapper.apply(conv_change_mask)
                    elif FLAGS.pruner == 'global':
                        global_pruning_update(model_wrapper, density_eval)
                    elif FLAGS.pruner == 'global_normal':
                        global_normal_pruning_update(model_wrapper, density_eval)

                    change_in_mask(model_wrapper, density_eval, epoch)
                    recored_sparsity(model_wrapper, density_eval, epoch)
                    
                    
    if getattr(FLAGS, 'calibrate_bn', False):
        if getattr(FLAGS, 'DST_TRAIN', False):
            density_mult_list = FLAGS.density_list.copy()
            width_mult_list = [FLAGS.width_mult] * len(density_mult_list)
            new_model, new_model_wrapper = get_model()
            new_model_wrapper.load_state_dict(
                model_wrapper.state_dict(), strict=False)
            model_wrapper = new_model_wrapper
            print('Start validation after calibration.')
            for width_cal, density_cal in zip(width_mult_list, density_mult_list):
                model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_cal))
                model_wrapper.apply(lambda m: setattr(m, 'density', density_cal))
                if FLAGS.pruner == 'local':
                    model_wrapper.apply(conv_change_mask)
                elif FLAGS.pruner == 'global':
                    global_pruning_update(model_wrapper, density_cal)
                elif FLAGS.pruner == 'global_normal':
                    global_normal_pruning_update(model_wrapper, density_cal)
                run_one_epoch(model_wrapper, train_loader, criterion, optimizer, -1, phase='cal', 
                              eval_width=width_cal, eval_density=density_cal)
                
            with torch.no_grad():
                for width_cal, density_cal in zip(width_mult_list, density_mult_list):
                    model_wrapper.apply(lambda m: setattr(m, 'width_mult', width_cal))
                    model_wrapper.apply(lambda m: setattr(m, 'density', density_cal))
                    if FLAGS.pruner == 'local':
                        model_wrapper.apply(conv_change_mask)
                    elif FLAGS.pruner == 'global':
                        global_pruning_update(model_wrapper, density_cal)
                    elif FLAGS.pruner == 'global_normal':
                        global_normal_pruning_update(model_wrapper, density_cal)
                    result = run_one_epoch(model_wrapper, test_loader, criterion, optimizer, -1, phase='test', 
                              eval_width=width_cal, eval_density=density_cal)
                torch.save({
                    'model' : model_wrapper.state_dict(),
                    },
                    os.path.join(FLAGS.log_dir, 'bn_cal.pt')
                    )
    return
            

def main():
    wandb.init(project='DST ImageNet Training', name=FLAGS.log_dir, config=FLAGS.yaml(), resume='allow')
    train_val_test()
    
main() 
