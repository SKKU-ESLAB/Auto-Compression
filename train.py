import importlib
import os
import time
import random
import math

from functools import wraps

import sys
import copy
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
from torch import multiprocessing
from torchvision import datasets, transforms
from torch.nn.modules.utils import _pair

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.transforms import ImageFolderLMDB
from ultron_io import UltronIO
from utils.config import FLAGS
from utils.meters import *
from utils.model_profiling import compare_models
from models.quantizable_ops import EMA
from models.quantizable_ops import QuantizableConv2d, QuantizableLinear
#import wandb
import datetime
import torch.cuda.amp as amp

#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--log_interval', type=int, default=100)
#args = parser.parse_args()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        if True: # is_master():
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            print('func:{!r} took: {:2.4f} sec'.format(f.__name__, te-ts))
        else:
            result = f(*args, **kw)
        return result
    return wrap


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    #if getattr(FLAGS, 'distributed', False):
    #    gpu_id = init_dist()
    #    if getattr(FLAGS, 'distributed_all_reduce', False):
    #        model_wrapper = AllReduceDistributedDataParallel(model.cuda())
    #    else:
    #        model_wrapper = torch.nn.parallel.DistributedDataParallel(
    #            model.cuda(), [gpu_id], gpu_id)
    #else:
    model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper


def data_transforms():
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
            if getattr(FLAGS, 'normalize', False):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                mean = [0.0, 0.0, 0.0]
                std = [1.0, 1.0, 1.0]
            #crop_scale = 0.08
            #jitter_param = 0.4
            #lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            if getattr(FLAGS, 'normalize', False):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                mean = [0.0, 0.0, 0.0]
                std = [1.0, 1.0, 1.0]
            #crop_scale = 0.25
            #jitter_param = 0.4
            #lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),# scale=(crop_scale, 1.0)),
            #transforms.ColorJitter(
            #    brightness=jitter_param, contrast=jitter_param,
            #    saturation=jitter_param),
            #Lighting(lighting_param),
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
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        ### me !! ###
        train_transforms = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms == 'cinic':
        if getattr(FLAGS, 'normalize', False):
            mean = [0.4789, 0.4723, 0.4305]
            std = [0.2421, 0.2383, 0.2587]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
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
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'imagenet1k_lmdb':
        if not FLAGS.test_only:
            train_set = ImageFolderLMDB(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = ImageFolderLMDB(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'imagenet1k_val50k':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
            seed = getattr(FLAGS, 'random_seed', 0)
            random.seed(seed)
            val_size = 50000
            random.shuffle(train_set.samples)
            train_set.samples = train_set.samples[val_size:]
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        test_set = None
    elif FLAGS.dataset == 'CINIC10':
        if not FLAGS.test_only:
            train_set = datasets.ImageFolder(
                os.path.join(FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'valid'),
            transform=val_transforms)
        test_set = datasets.ImageFolder(
            os.path.join(FLAGS.dataset_dir, 'test'),
            transform=val_transforms)
    elif FLAGS.dataset == 'CIFAR10':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR10(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        else:
            train_set = None
        val_set = datasets.CIFAR10(
            FLAGS.dataset_dir,
            train=False,
            transform = val_transforms,
            download=True)
        test_set = None
    elif FLAGS.dataset == 'CIFAR100':
        if not FLAGS.test_only:
            train_set = datasets.CIFAR100(
                FLAGS.dataset_dir,
                transform = train_transforms,
                download=True)
        else:
            train_set = None
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
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset))
    return train_set, val_set, test_set


def data_loader(train_set, val_set, test_set):
    """get data loader"""
    train_loader = None
    val_loader = None
    test_loader = None
    if getattr(FLAGS, 'batch_size', False):
        if getattr(FLAGS, 'batch_size_per_gpu', False):
            assert FLAGS.batch_size == (FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
        else:
            assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
            FLAGS.batch_size_per_gpu = (FLAGS.batch_size // FLAGS.num_gpus_per_job)
    elif getattr(FLAGS, 'batch_size_per_gpu', False):
        FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    else:
        raise ValueError('batch size (per gpu) is not defined')
    batch_size = int(FLAGS.batch_size)# / get_world_size())
    if FLAGS.data_loader in ['imagenet1k_basic','cifar', 'cinic']:
        #if getattr(FLAGS, 'distributed', False):
        #    if FLAGS.test_only:
        #        train_sampler = None
        #    else:
        #        train_sampler = DistributedSampler(train_set)
        #    val_sampler = DistributedSampler(val_set)
        #else:
        train_sampler = None
        val_sampler = None
        if not FLAGS.test_only:
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


def lr_func(x, fun='cos'):
    if fun == 'cos':
        return math.cos( x * math.pi ) / 2 + 0.5
    if fun == 'exp':
        return math.exp( - x * 8 )
    if fun == 'gaussian':
        return ( math.exp( - x**2 * 8 ) + 0.02 ) / 1.02
    if fun == 'butterworth':
        return ( 1 / ( ( x * 3 ) ** 10 + 1 ) ** 0.5 + 0.02 ) / 1.02
    if fun == 'mixed':
        return ( math.cos( x * math.pi ) / 2 + 0.5 ) / ( ( x * 1.5 ) ** 20 + 1 ) ** 0.5


def get_lr_scheduler(optimizer, nBatch=None):
    """get learning rate"""
    #warmup_epochs = getattr(FLAGS, 'lr_warmup_epochs', 0)
    if FLAGS.lr_scheduler == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma)
    elif FLAGS.lr_scheduler == 'exp_decaying':
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            elif i % getattr(FLAGS, 'exp_decaying_period', 1) == 0:
                lr_dict[i] = lr_dict[i-1] * FLAGS.exp_decaying_lr_gamma
            else:
                lr_dict[i] = lr_dict[i-1]
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'exp_decaying_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
            #lr_dict[i] = math.exp(-(i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters) * 8)
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters), 'exp')
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'gaussian_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
            #lr_dict[i] = math.exp(-(i - FLAGS.warmup_iters)**2 / (FLAGS.num_iters - FLAGS.warmup_iters)**2 * 8)
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters), 'gaussian')
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'butterworth_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters), 'butterworth')
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'mixed_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
            lr_dict[i] = lr_func((i - FLAGS.warmup_iters) / (FLAGS.num_iters - FLAGS.warmup_iters), 'mixed')
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'linear_decaying':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - FLAGS.warmup_epochs) / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing':
        num_epochs = FLAGS.num_epochs - FLAGS.warmup_epochs
        lr_dict = {}
        for  i in range(FLAGS.num_epochs):
            lr_dict[i] = (1.0 + math.cos( (i - FLAGS.warmup_epochs) * math.pi / num_epochs)) / 2
        lr_lambda = lambda epoch: lr_dict[epoch] # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing_iter':
        FLAGS.num_iters = FLAGS.num_epochs * nBatch
        FLAGS.warmup_iters = FLAGS.warmup_epochs * nBatch
        lr_dict = {}
        for i in range(FLAGS.warmup_iters):
            bs_ratio = 256 / FLAGS.batch_size
            lr_dict[i] = (1 - bs_ratio) / FLAGS.warmup_iters * i + bs_ratio
        if getattr(FLAGS, 'warm_restart', False):
            T = 10
            T_iter = T * nBatch
            start_iter = FLAGS.warmup_iters
            while True:
                if start_iter >= FLAGS.num_iters:
                    break
                T_iter = min(T_iter, FLAGS.num_iters - start_iter)
                for i in range(start_iter, start_iter + T_iter):
                    if i >= FLAGS.num_iters:
                        break
                    lr_dict[i] = (1.0 + math.cos((i - start_iter) * math.pi / T_iter)) / 2
                start_iter += T_iter
                T_iter *= 2
        else:
            for i in range(FLAGS.warmup_iters, FLAGS.num_iters):
                lr_dict[i] = (1.0 + math.cos((i - FLAGS.warmup_iters) * math.pi / (FLAGS.num_iters - FLAGS.warmup_iters))) / 2
        lr_lambda = lambda itr: lr_dict[itr] # noqa: E731
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
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, alpha=FLAGS.optim_decay, eps=FLAGS.optim_eps, weight_decay=FLAGS.weight_decay, momentum=FLAGS.momentum)
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
    print('seed for random sampling: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#@master_only
def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


def profiling(model, use_cuda):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    flops, params, bitops, bitops_max, bytesize, energy, latency = model_profiling(
        model, FLAGS.image_size, FLAGS.image_size,
        verbose=getattr(FLAGS, 'model_profiling_verbose', False))
    return bitops, bytesize


def get_experiment_setting():
    experiment_setting = 'ema_decay_{ema_decay}/fp_pretrained_{fp_pretrained}/bit_list_{bit_list}'.format(ema_decay=getattr(FLAGS, 'ema_decay', None), fp_pretrained=getattr(FLAGS, 'fp_pretrained_file', None) is not None,  bit_list='_'.join([str(i) for i in getattr(FLAGS, 'bits_list', None)]))
    if getattr(FLAGS, 'act_bits_list', False):
        experiment_setting = os.path.join(experiment_setting, 'act_bits_list_{}'.format('_'.join([str(i) for i in FLAGS.act_bits_list])))
    if getattr(FLAGS, 'double_side', False):
        experiment_setting = os.path.join(experiment_setting, 'double_side_True')
    if not getattr(FLAGS, 'rescale', False):
        experiment_setting = os.path.join(experiment_setting, 'rescale_False')
    if not getattr(FLAGS, 'calib_pact', False):
        experiment_setting = os.path.join(experiment_setting, 'calib_pact_False')
    experiment_setting = os.path.join(experiment_setting, 'kappa_{kappa}'.format(kappa=getattr(FLAGS, 'kappa', 1.0)))
    if getattr(FLAGS, 'target_bitops', False):
        experiment_setting = os.path.join(experiment_setting, 'target_bitops_{}'.format(getattr(FLAGS, 'target_bitops', False)))
    if getattr(FLAGS, 'target_size', False):
        experiment_setting = os.path.join(experiment_setting, 'target_size_{}'.format(getattr(FLAGS, 'target_size', False)))
    if getattr(FLAGS, 'init_bit', False):
        experiment_setting = os.path.join(experiment_setting, 'init_bit_{}'.format(getattr(FLAGS, 'init_bit', False)))
    if getattr(FLAGS, 'unbiased', False):
        experiment_setting = os.path.join(experiment_setting, f'unbiased_True')
    print('Experiment settings: {}'.format(experiment_setting))
    return experiment_setting


def forward_loss(model, criterion, inputs, targets, meter):
    """forward model and return loss"""
    if getattr(FLAGS, 'normalize', False):
        inputs = inputs #(128 * inputs).round_().clamp_(-128, 127)
    else:
        inputs = (255 * inputs).round_()
    outputs = model(inputs)
    loss = torch.mean(criterion(outputs, targets))
    # topk
    _, pred = outputs.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    res = torch.cat(correct_k, dim=0)
    #if getattr(FLAGS, 'distributed', False) and getattr(FLAGS, 'distributed_all_reduce', False):
    #    res = dist_all_reduce_tensor(res)
    res = res.cpu().detach().numpy()
    bs = (res.size - 1) // len(FLAGS.topk)
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1. - res[i*bs:(i+1)*bs])
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(loss.tolist())
    return loss


def bit_discretizing(model):
    print('hard offset', FLAGS.hard_offset)
    for m in model.modules():
        if hasattr(m, 'bit_discretizing'):
          print('bit discretized for ', m)
          m.bit_discretizing()


def get_comp_cost_loss(model):
    loss = 0.0
    for m in model.modules():
        loss += getattr(m, 'comp_cost_loss', 0.0)
    target_bitops = getattr(FLAGS, 'target_bitops', False)
    if target_bitops:

        loss = torch.abs(loss - target_bitops)
    return loss


def get_model_size_loss(model):
    loss = 0.0
    for m in model.modules():
        loss += getattr(m, 'model_size_loss', 0.0)
    target_size = getattr(FLAGS, 'target_size', False)
    if target_size:
        loss = torch.abs(loss - target_size)
    return loss


torch.autograd.set_detect_anomaly(True)
@timing
def run_one_epoch(
        epoch, loader, model, criterion, optimizer, meters, phase='train', ema=None, scheduler=None, scaler=None):
    """run one epoch for train/val/test/cal"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test', 'cal'], "phase not be in train/val/test/cal."
    train = phase == 'train'
    log_dir = FLAGS.log_dir
    if train:
        model.train()
    else:
        model.eval()
    
    eval_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    n_layer = 53
    len_loader = len(loader) // FLAGS.log_interval

    for batch_idx, (inputs, targets) in enumerate(loader):
        ######### FAST TEST Start ###########
        # if batch_idx == len_loader :
        #     break
        ######### FAST TEST End ###########

        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        targets = targets.cuda(non_blocking=True)
        if train:
            if FLAGS.lr_scheduler == 'linear_decaying':
                linear_decaying_per_step = (
                    FLAGS.lr/FLAGS.num_epochs/len(loader.dataset)*FLAGS.batch_size)
                for param_group in optimizer.param_groups:
                    param_group['lr'] -= linear_decaying_per_step
            
            
            ############ [Question] what is this block for? #############
            ##                      8-bit quantization?                      
            if getattr(FLAGS, 'normalize', False):
                inputs = inputs #(128 * inputs).round_().clamp_(-128, 127)
            else:
                inputs = (255 * inputs).round_()
            #############################################################
            if getattr(FLAGS, 'amp', False):
                with amp.autocast():
                    outputs = model(inputs)
                    loss = torch.mean(criterion(outputs, targets))
                    if epoch >= FLAGS.warmup_epochs and not getattr(FLAGS,'hard_assignment', False):
                        if getattr(FLAGS,'weight_only', False):
                            loss += getattr(FLAGS, 'kappa', 1.0) * get_model_size_loss(model)
                        else:
                            loss += getattr(FLAGS, 'kappa', 1.0) * get_comp_cost_loss(model)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                outputs = model(inputs)
                loss = torch.mean(criterion(outputs, targets))
                if epoch >= FLAGS.warmup_epochs and not getattr(FLAGS,'hard_assignment', False):
                    if getattr(FLAGS,'weight_only', False):
                        loss += getattr(FLAGS, 'kappa', 1.0) * get_model_size_loss(model)
                    else:
                        loss += getattr(FLAGS, 'kappa', 1.0) * get_comp_cost_loss(model)
                loss.backward()
                optimizer.step()

            if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
                scheduler.step()

            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            eval_loss.update(loss.item(), inputs.size(0))

            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            ################# for removing wandb ##############
            '''
            if getattr(FLAGS, 'log_bitwidth', False):
                log_dict = {'acc1_iter': acc1.item(), 
                            'acc1_avg': top1.avg}
                cnt = 0
                for name, m in model.named_modules():
                    if hasattr(m, 'lamda_w'):
                        log_dict[f'{cnt}_lambda_w'] = m.lamda_w.item()
                        log_dict[f'{cnt}_lambda_a'] = m.lamda_a.item()
                        cnt += 1
                log_dict['loss'] = loss.item()
                wandb.log(log_dict)
            '''
            #################################################
            if (batch_idx) % FLAGS.log_interval == 0:
                curr = batch_idx * len(inputs)
                total = len(loader.dataset)
                print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Train Epoch: {epoch:4d}  Phase: {phase}  Process: {curr:5d}/{total:5d} '\
                    f'Loss: {eval_loss.avg:.4f} | '\
                    f'top1.avg: {top1.avg:.4f} % | '\
                    f'top5.avg: {top5.avg:.4f} % | ')

        
        else: #not train
            if ema:
                print('ema apply')
                ema.shadow_apply(model)
            forward_loss(model, criterion, inputs, targets, meters)
            outputs = model(inputs)
            if ema:
                print('ema recover')
                ema.weight_recover(model)

    val_top1 = None
    try:
        print('{:.1f}s\t{}\t{}/{}: '.format(
        time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
        ', '.join('{}: {}'.format(k, v) for k, v in results.items()))
        val_top1 = results['top1_error']
    except:
        val_top1 = top1.avg
    return val_top1


@timing
def train_val_test():
    if not getattr(FLAGS, 'gumbel', True):
        from models.quantizable_ops_nogumbel import QuantizableConv2d, QuantizableLinear
    if getattr(FLAGS, 'amp', False):
        print()
        print('--------------------------------------')
        print('==> AUTOMATIC MIXED PRECISION Training')
        print('--------------------------------------\n\n')
    """train and val"""
    torch.backends.cudnn.benchmark = False
    scaler = torch.cuda.amp.GradScaler()
    if True: #else:
        set_random_seed()

    ####### DEBUG MSG 
    interp_method = 'simple_interpolation (ours)' if getattr(FLAGS, 'simple_interpolation', False) else 'fracbits_original'
    print(f'\n==> Interpolation method: {interp_method}\n')
    if getattr(FLAGS, 'bitwidth_direct', False):
        print('==> Direct learning of bitwidth (This should be shown)\n')


    # experiment setting
    experiment_setting = get_experiment_setting()

    # model
    model, model_wrapper = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    if getattr(FLAGS, 'profiling_only', False):
        if 'gpu' in FLAGS.profiling:
            profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            profiling(model, use_cuda=False)
        return
    
    # ema_decay : not used
    ema_decay = getattr(FLAGS, 'ema_decay', None)
    if ema_decay:
        ema = EMA(ema_decay)
        ema.shadow_register(model_wrapper)
    else:
        ema = None
    
    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    train_set, val_set, test_set = dataset(
        train_transforms, val_transforms, test_transforms)
    train_loader, val_loader, test_loader = data_loader(
        train_set, val_set, test_set)

    log_dir = FLAGS.log_dir
    log_dir = os.path.join(log_dir, experiment_setting)


    # full precision pretrained
    if getattr(FLAGS, 'fp_pretrained_file', None):
        if not os.path.isfile(FLAGS.fp_pretrained_file):
            pretrain_dir = os.path.dirname(FLAGS.fp_pretrained_file)
            print(FLAGS.fp_pretrained_file)
            os.system(f"wget -P {pretrain_dir} https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
        checkpoint = torch.load(FLAGS.fp_pretrained_file)
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
        model_dict = model_wrapper.state_dict()
        
        #checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        # remove unexpected keys
        for k in list(checkpoint.keys()):
            if k not in model_dict.keys():
                checkpoint.pop(k)
        model_dict.update(checkpoint)
        model_wrapper.load_state_dict(model_dict)
        print('Loaded full precision model {}.'.format(FLAGS.fp_pretrained_file))

    # check pretrained ----------------------------------
    if FLAGS.pretrained_file and FLAGS.pretrained_dir:
        pretrained_dir = FLAGS.pretrained_dir
        #pretrained_dir = os.path.join(pretrained_dir, experiment_setting)
        pretrained_file = os.path.join(pretrained_dir, FLAGS.pretrained_file)
        #checkpoint = io.torch_load(
        #    pretrained_file, map_location=lambda storage, loc: storage)
        checkpoint = torch.load(pretrained_file)
        # update keys from external models
        #if type(checkpoint) == dict and 'model' in checkpoint:
        #    checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                print('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        # filter lamda_w and lamda_a args:
        pretrained_dict = {}
        for k,v in checkpoint['model'].items():
            if 'lamda_w' in k or 'lamda_a' in k:
                checkpoint['model'][k] = v.repeat(model_wrapper.state_dict()[k].size())
        model_wrapper.load_state_dict(checkpoint['model'])
        print('Loaded model {}.'.format(pretrained_file))
    optimizer = get_optimizer(model_wrapper)

    if FLAGS.test_only and (test_loader is not None):
        print('Start testing.')
        ema = checkpoint.get('ema', None)
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(
                -1, test_loader,
                model_wrapper, criterion, optimizer,
                test_meters, phase='test', ema=ema, scaler=scaler)
        return

    # check resume training ------------------------------
    if os.path.isfile(os.path.join(log_dir, 'latest_checkpoint.pt')):
        checkpoint = torch.load(os.path.join(log_dir, 'latest_checkpoint.pt'))
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
            lr_scheduler = get_lr_scheduler(optimizer, len(train_loader))
            lr_scheduler.last_epoch = last_epoch * len(train_loader)
        else:
            lr_scheduler = get_lr_scheduler(optimizer)
            lr_scheduler.last_epoch = last_epoch
        best_val = checkpoint['best_val']
        train_meters, val_meters = checkpoint['meters']
        ema = checkpoint.get('ema', None)
        print('Loaded checkpoint {} at epoch {}.'.format(
            log_dir, last_epoch))
    else:
        if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
            lr_scheduler = get_lr_scheduler(optimizer, len(train_loader))
        else:
            lr_scheduler = get_lr_scheduler(optimizer)
        last_epoch = lr_scheduler.last_epoch
        best_val = 1.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        # if start from scratch, print model and do profiling
        #print(model_wrapper)
        if getattr(FLAGS, 'profiling', False):
            if 'gpu' in FLAGS.profiling:
                profiling(model, use_cuda=True)
            if 'cpu' in FLAGS.profiling:
                profiling(model, use_cuda=False)

    if getattr(FLAGS, 'log_dir', None):
        try:
            os.makedirs(log_dir)
        except OSError:
            pass
    #################### for removing wandb #################
    #PROJECT_NAME='LBQv2'
    #wandb.init(project=PROJECT_NAME, dir=FLAGS.log_dir)
    #wandb.config.update(FLAGS)
    #########################################################
    print('Start training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
            lr_sched = lr_scheduler
        else:
            lr_sched = None
            # For PyTorch 1.1+, comment the following line
            #lr_scheduler.step()
        # train ---------------------------------------------
        print(' train '.center(40, '*')) 
        train_top1 = run_one_epoch(
          epoch, train_loader, model_wrapper, criterion, optimizer,
          train_meters, phase='train', ema=ema, scheduler=lr_sched, scaler=scaler)
        #print(f'{train_top1} <-> {flush_scalar_meters(train_meters)["top1_error"]}') -> 안되넴

        # val -----------------------------------------------
        print(' validation '.center(40, '~'))
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            if epoch == getattr(FLAGS,'hard_assign_epoch', float('inf')):
                #################
                if getattr(FLAGS, 'nlvs_direct', False):
                    FLAGS.nlvs_direct = False
                    for m in model.modules():
                        if hasattr(m, 'nlvs_w'):
                            m.quant_type = 'simple_interpolation'
                            m.lamda_w.data = torch.log2(m.nlvs_w)
                            print(f'm.lamda_w.data = {m.lamda_w.data}')
                        if hasattr(m, 'nlvs_a'):
                            m.quant_type = 'simple_interpolation'
                            m.lamda_a.data = torch.log2(m.nlvs_a)
                            print(f'm.lamda_a.data = {m.lamda_a.data}')
                print('Start to use hard assigment')
                setattr(FLAGS, 'hard_assignment', True)
                lower_offset = -1
                higher_offset = 0
                setattr(FLAGS, 'hard_offset', 0)

                with_ratio = 0.01
                bitops, bytesize = profiling(model, use_cuda=True)
                search_trials = 10
                trial = 0
                if getattr(FLAGS,'weight_only', False):
                    target_bytesize = getattr(FLAGS, 'target_size', 0)
                    while trial < search_trials:
                        trial += 1
                        if bytesize - target_bytesize > with_ratio * target_bytesize:
                            higher_offset = FLAGS.hard_offset
                        elif bytesize - target_bytesize < -with_ratio * target_bytesize:
                            lower_offset = FLAGS.hard_offset
                        else:
                            break
                        FLAGS.hard_offset = (higher_offset + lower_offset) /2
                        bitops, bytesize = profiling(model, use_cuda=True)
                else:
                    target_bitops = getattr(FLAGS, 'target_bitops',0)
                    while trial < search_trials:
                        trial += 1
                        if bitops - target_bitops > with_ratio *target_bitops:
                            higher_offset = FLAGS.hard_offset
                        elif bitops - target_bitops < -with_ratio * target_bitops:
                            lower_offset = FLAGS.hard_offset
                        else:
                            break
                        FLAGS.hard_offset = (higher_offset + lower_offset) /2
                        bitops, bytesize = profiling(model, use_cuda=True)
                bit_discretizing(model_wrapper)
                setattr(FLAGS,'hard_offset', 0)
            top1_error = run_one_epoch(
                epoch, val_loader, model_wrapper, criterion, optimizer,
                val_meters, phase='val', ema=ema, scaler=scaler)

        if top1_error < best_val:
            best_val = top1_error
            torch.save(
                {
                    'model': model_wrapper.state_dict(),
                },
                os.path.join(log_dir, 'best_model.pt'))
            print('New best validation top1 error: {:.3f}'.format(best_val))

        # save latest checkpoint
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
                'best_val': best_val,
                'meters': (train_meters, val_meters),
                'ema': ema,
            },
            os.path.join(log_dir, 'latest_checkpoint.pt'))

        # For PyTorch 1.0 or earlier, comment the following two lines
        if FLAGS.lr_scheduler not in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
            lr_scheduler.step()

    profiling(model, use_cuda=True)
    for m in model.modules():
        if hasattr(m, 'alpha'):
            print(m, m.alpha)
        if hasattr(m, 'lamda_w'):
            print(m, m.lamda_w)
        if hasattr(m, 'lamda_a'):
            print(m, m.lamda_a)
    return


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    """train and eval model"""
    #init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()
