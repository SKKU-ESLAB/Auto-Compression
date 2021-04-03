import importlib
import os
import time
import random
import math

from functools import wraps

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from pysnooper import snoop
from memory_profiler import profile

import torch
import torch.nn as nn
from torch import multiprocessing
# from torch.distributed import all_gather, get_world_size, is_initialized
from torchvision import datasets, transforms
# from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.utils import _pair

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.transforms import ImageFolderLMDB
# from utils.distributed import init_dist, master_only, is_master
# from utils.distributed import get_rank, get_world_size
# from utils.distributed import dist_all_reduce_tensor
# from utils.distributed import master_only_print as mprint
# from utils.distributed import AllReduceDistributedDataParallel, allreduce_grads

from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters

from models.quantizable_ops import EMA
from models.quantizable_ops import QuantizableConv2d, QuantizableLinear
from models.quantizable_ops import bn_calibration

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        if is_master():
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            mprint('func:{!r} took: {:2.4f} sec'.format(f.__name__, te-ts))
        else:
            result = f(*args, **kw)
        return result
    return wrap


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
#   if getattr(FLAGS, 'distributed', False):
#      gpu_id = init_dist()
#        if getattr(FLAGS, 'distributed_all_reduce', False):
#            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
##        else:
#            model_wrapper = torch.nn.parallel.DistributedDataParallel(
#                model.cuda(), [gpu_id], gpu_id)
#    else:
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
    batch_size = int(FLAGS.batch_size / get_world_size())
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
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1. - (i - warmup_epochs) / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]  # noqa: E731
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == 'cos_annealing':
        num_epochs = FLAGS.num_epochs - warmup_epochs
        lr_dict = {}
        for  i in range(FLAGS.num_epochs):
            lr_dict[i] = (1.0 + math.cos( (i - warmup_epochs) * math.pi / num_epochs)) / 2
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


@master_only
def get_meters(phase, single_sample=False):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    if single_sample:
        meters = get_single_meter(phase)
    elif getattr(FLAGS, 'quantizable_training', False):
        meters = {}
        for bits in FLAGS.bits_list:
            meters[str(bits)] = get_single_meter(phase, str(bits))
        #if getattr(FLAGS, 'stoch_valid', False):
        #    for sample_idx in range(getattr(FLAGS, 'sample_iter', 8)):
        #        meters[f'sample_{sample_idx}'] = get_single_meter(phase, f'sample_{sample_idx}')
    else:
        meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


@master_only
def profiling(model, use_cuda, single_sample=False):
    """profiling on either gpu or cpu"""
    mprint('Start model profiling, use_cuda:{}.'.format(use_cuda))
    if getattr(FLAGS, 'quantizable_training', False) and not single_sample:
        for bits in FLAGS.bits_list:
            model.apply(
                lambda m: setattr(m,'bits', bits))
            mprint('Model profiling with {} bits.'.format(bits))
            flops, params, bitops, bytesize, energy, latency = model_profiling(
                model, FLAGS.image_size, FLAGS.image_size,
                verbose=getattr(FLAGS, 'model_profiling_verbose', False))
    else:
        flops, params, bitops, bytesize, energy, latency = model_profiling(
            model, FLAGS.image_size, FLAGS.image_size,
            verbose=getattr(FLAGS, 'model_profiling_verbose', False))
    return flops, params


def get_experiment_setting():
    experiment_setting = 'ema_decay_{ema_decay}/fp_pretrained_{fp_pretrained}/bit_list_{bit_list}'.format(ema_decay=getattr(FLAGS, 'ema_decay', None), fp_pretrained=getattr(FLAGS, 'fp_pretrained_file', None) is not None,  bit_list=getattr(FLAGS, 'bits_list', None))
    experiment_setting = os.path.join(experiment_setting, 'kappa_{kappa}'.format(kappa=getattr(FLAGS, 'kappa', 1.0)))
    if getattr(FLAGS, 'unbiased', False):
        experiment_setting = os.path.join(experiment_setting, f'unbiased_True')
    mprint('Experiment settings: {}'.format(experiment_setting))
    return experiment_setting


#@snoop()
def forward_loss(model, criterion, input, target, meter):
    """forward model and return loss"""
    if getattr(FLAGS, 'normalize', False):
        input = input #(128 * input).round_().clamp_(-128, 127)
    else:
        input = (255 * input).round_()
    output = model(input)
    loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    res = torch.cat([loss.view(1)] + correct_k, dim=0)
    #if getattr(FLAGS, 'distributed', False) and getattr(FLAGS, 'distributed_all_reduce', False):
    #    res = dist_all_reduce_tensor(res)
    res = res.cpu().detach().numpy()
    bs = (res.size - 1) // len(FLAGS.topk)
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1. - res[1+i*bs:1+(i+1)*bs])
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(res[0])
    return loss


@timing
#@snoop(depth=2)
def run_one_epoch(
        epoch, loader, model, criterion, optimizer, meters, phase='train', ema=None, scheduler=None, eta=None, epoch_dict=None, single_sample=False):
    """run one epoch for train/val/test/cal"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test', 'cal'], "phase not be in train/val/test/cal."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
        #if getattr(FLAGS, 'bn_calib', False) and phase == 'val' and epoch < FLAGS.num_epochs - 10:
        #    model.apply(bn_calibration)
        #if getattr(FLAGS, 'bn_calib_stoch_valid', False):
        #    model.apply(bn_calibration)
        if phase == 'cal':
            model.apply(bn_calibration)

    if getattr(FLAGS, 'distributed', False): ###################### What does this line do?? ##########################
        loader.sampler.set_epoch(epoch)

    scale_dict = {}
    if getattr(FLAGS, 'switch_lr', False):
        scale_dict = {32: 1.0, 16: 1.0, 8: 1.0, 6: 1.0, 5: 1.0, 4: 1.02, 3: 1.08, 2: 1.62, 1: 4.83}

    for batch_idx, (input, target) in enumerate(loader):
        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        target = target.cuda(non_blocking=True)
        if train:
            if FLAGS.lr_scheduler == 'linear_decaying':
                linear_decaying_per_step = (
                    FLAGS.lr/FLAGS.num_epochs/len(loader.dataset)*FLAGS.batch_size)
                for param_group in optimizer.param_groups:
                    param_group['lr'] -= linear_decaying_per_step
            # For PyTorch 1.1+, comment the following two line
            #if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
            #    scheduler.step()
            optimizer.zero_grad()
            if getattr(FLAGS, 'quantizable_training', False) and not single_sample:
                for bits_idx, bits in enumerate(FLAGS.bits_list):
                    model.apply(
                        lambda m: setattr(m, 'bits', bits))
                    if is_master():
                        meter = meters[str(bits)]
                    else:
                        meter = None
                    loss = forward_loss(
                        model, criterion, input, target, meter)
                    if eta is not None:
                        #if isinstance(bits, (list, tuple)):
                        #    bitw = bits[0]
                        #else:
                        #    bitw = bits
                        #loss *= eta(bitw)
                        loss *= eta(_pair(bits)[0])
                    if getattr(FLAGS, 'switch_lr', False):
                        #mprint(scale_dict[_pair(bits)[0]])
                        loss *= scale_dict[_pair(bits)[0]]
                    if epoch_dict is None:
                        loss.backward()
                    else:
                        epoch_valid = epoch_dict[_pair(bits)[0]]
                        if isinstance(epoch_valid, (list, tuple)):
                            epoch_start, epoch_end = epoch_valid
                        else:
                            epoch_start = epoch_valid
                            epoch_end = 1.0
                        epoch_start = int(FLAGS.num_epochs * epoch_start)
                        epoch_end = int(FLAGS.num_epochs * epoch_end)
                        if epoch_start <= epoch and epoch < epoch_end:
                            loss.backward()
                    if getattr(FLAGS, 'print_grad_std', False):
                        mprint(f'bits: {bits}')
                        layer_idx = 0
                        grad_std_list = []
                        for m in model.modules():
                            #if getattr(m, 'weight', None) is not None:
                            if isinstance(m, (QuantizableConv2d, QuantizableLinear)):
                                grad_std = torch.std(m.weight.grad)
                                mprint(f'layer_{layer_idx} grad: {grad_std}') #, module: {m}')
                                grad_std_list.append(grad_std)
                                layer_idx += 1
                        mprint(f'average grad std: {torch.mean(torch.tensor(grad_std_list))}')
            else:
                loss = forward_loss(
                    model, criterion, input, target, meters)
                loss.backward()
            #if getattr(FLAGS, 'distributed', False) and getattr(FLAGS, 'distributed_all_reduce', False):
            #    allreduce_grads(model)
            optimizer.step()
            # For PyTorch 1.0 or earlier, comment the following two lines
            if FLAGS.lr_scheduler in ['exp_decaying_iter', 'gaussian_iter', 'cos_annealing_iter', 'butterworth_iter', 'mixed_iter']:
                scheduler.step()
            if ema:
                ema.shadow_update(model)
                #for name, param in model.named_parameters():
                #    if param.requires_grad:
                #        ema.update(name, param.data)
                #bn_idx = 0
                #for m in model.modules():
                #    if isinstance(m, nn.BatchNorm2d):
                #        ema.update('bn{}_mean'.format(bn_idx), m.running_mean)
                #        ema.update('bn{}_var'.format(bn_idx), m.running_var)
                #        bn_idx += 1
        else: #not train
            if ema:
                ema.shadow_apply(model)
            if getattr(FLAGS, 'quantizable_training', False) and not single_sample:
                for bits_idx, bits in enumerate(FLAGS.bits_list):
                    model.apply(
                        lambda m: setattr(m, 'bits', bits))
                    #model.apply(
                    #    lambda m: setattr(m, 'threshold', FLAGS.schmitt_threshold * (0.0 * (epoch <= 30) + 0.01 * (30 < epoch <= 60) + 0.1 * (60 < epoch <= 90) + 1.0 * (90 < epoch))))
                    #model.apply(
                    #    lambda m: setattr(m, 'threshold', epoch * FLAGS.schmitt_threshold / FLAGS.num_epochs))
                    if is_master():
                        meter = meters[str(bits)]
                    else:
                        meter = None
                    forward_loss(
                        model, criterion, input, target, meter)
            else:
                forward_loss(model, criterion, input, target, meters)
            if ema:
                ema.weight_recover(model)
    ##opt_loss = float('inf')
    ##opt_results = None
    val_top1 = None
    if is_master():
        if getattr(FLAGS, 'quantizable_training', False) and not single_sample:
            #results_dict = {}
            val_top1_list = []
            for bits in FLAGS.bits_list:
                results = flush_scalar_meters(meters[str(bits)])
                mprint('{:.1f}s\t{}\t{} bits\t{}/{}: '.format(
                    time.time() - t_start, phase, bits, epoch,
                    FLAGS.num_epochs) + ', '.join('{}: {}'.format(k, v)
                                                  for k, v in results.items()))
                #results_dict[str(bits)] = results
                ##if results['loss'] < opt_loss:
                ##    opt_results = results
                ##    opt_loss = results['loss']
                val_top1_list.append(results['top1_error'])
            #results = results_dict
            val_top1 = np.mean(val_top1_list)
        else:
            results = flush_scalar_meters(meters)
            mprint('{:.1f}s\t{}\t{}/{}: '.format(
                time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
                  ', '.join('{}: {}'.format(k, v) for k, v in results.items()))
            ##if results['loss'] < opt_loss:
            ##    opt_results = results
            ##    opt_loss = results['loss']
            val_top1 = results['top1_error']
    ##return opt_results
    #return results
    return val_top1


#@profile
#@snoop(depth=2)
@timing
def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # init distributed
    if getattr(FLAGS, 'distributed', False):
        init_dist()
    # seed
    if getattr(FLAGS, 'use_diff_seed', False) and not getattr(FLAGS, 'stoch_valid', False):
        print('use diff seed is True')
        while not is_initialized():
            print('Waiting for initialization ...')
            time.sleep(5)
        print('Expected seed: {}'.format(getattr(FLAGS, 'random_seed', 0) + get_rank()))
        set_random_seed(getattr(FLAGS, 'random_seed', 0) + get_rank())
    else:
        set_random_seed()

    if getattr(FLAGS, 'adjust_lr', False):
        eta_dict = {32: 1.0, 16: 1.0, 8: 1.0, 7: 0.99, 6: 0.98, 5: 0.97, 4: 0.94, 3: 0.88, 2: 0.77, 1: 0.58}
        eta = lambda b: eta_dict[b]  # noqa: E731
    else:
        eta = None


    # experiment setting
    experiment_setting = get_experiment_setting()
    mprint('stoch_valid: {}, bn_calib_stoch_valid: {}'.format(getattr(FLAGS, 'stoch_valid', False), getattr(FLAGS, 'bn_calib_stoch_valid', False)))

    # model
    model, model_wrapper = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    if getattr(FLAGS, 'profiling_only', False):
        if 'gpu' in FLAGS.profiling:
            profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            profiling(model, use_cuda=False)
        return

    #
    ema_decay = getattr(FLAGS, 'ema_decay', None)
    if ema_decay:
        ema = EMA(ema_decay)
        ema.shadow_register(model_wrapper)
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        ema.register(name, param.data)
        #bn_idx = 0
        #for m in model.modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        ema.register('bn{}_mean'.format(bn_idx), m.running_mean)
        #        ema.register('bn{}_var'.format(bn_idx), m.running_var)
        #        bn_idx += 1
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

    # check pretrained
    if FLAGS.pretrained_file:
        pretrained_dir = FLAGS.pretrained_dir
        pretrained_dir = os.path.join(pretrained_dir, experiment_setting)
        pretrained_file = os.path.join(pretrained_dir, FLAGS.pretrained_file)
        checkpoint = torch.load(
            pretrained_file, map_location=lambda storage, loc: storage)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                mprint('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        mprint('Loaded model {}.'.format(pretrained_file))
    optimizer = get_optimizer(model_wrapper)
    cal_meters = get_meters('cal', single_sample=True)
    mprint('Start calibration.')
    run_one_epoch(
        -1, train_loader,
        model_wrapper, criterion, optimizer, cal_meters,
        phase='cal', ema=ema, single_sample=True)
    mprint('Start validation after calibration.')
    with torch.no_grad():
        run_one_epoch(
            -1, val_loader,
            model_wrapper, criterion, optimizer, cal_meters,
            phase='val', ema=ema, single_sample=True)
    return


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    """train and eval model"""
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()
