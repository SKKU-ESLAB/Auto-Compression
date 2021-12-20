import math
import os
import re
import sys
import time
import shutil
import pathlib
import glob

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def print_param(model):
    for name, param in model.named_parameters():
        if ('p_' in name):
            print(name,torch.exp(param.data[0]),param.data[0])
        elif ('gamma' in name):
            print(name,param.data[0])

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(model):
    '''Init layer parameters.'''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def count_memory(model):
    total_params = 0
    for name, param in model.named_parameters():
        if 'aux' in name:
            continue
        total_params += np.prod(param.size())
    return total_params / 1e6

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]


def create_checkpoint(model, model_ema, optimizer, is_best, is_ema_best,
        acc, best_acc, epoch, root, save_freq=10, prefix='train'):
    pathlib.Path(root).mkdir(parents=True, exist_ok=True) 

    filename = os.path.join(root, '{}_{}.pth'.format(prefix, epoch))
    bestname = os.path.join(root, '{}_best.pth'.format(prefix))
    #bestemaname = os.path.join(root, '{}_ema_best.pth'.format(prefix))
    #tempname = os.path.join(_temp_dir, '{}_tmp.pth'.format(prefix))

    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    if model_ema is not None: 
        if isinstance(model_ema, torch.nn.DataParallel):
            model_ema_state = model_ema.module.state_dict()
        else:
            model_ema_state = model_ema.state_dict()
    else:
        model_ema_state = None

    if is_best:
        torch.save(model_state, bestname)

    if is_ema_best:        
        torch.save(model_ema_state, bestname)
        
    if epoch > 0 and (epoch % save_freq) == 0:
        state = {
            'model': model_state,
            'model_ema': model_ema_state,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'best_acc': best_acc,
        }
        torch.save(state, filename)


def resume_checkpoint(model, model_ema, optimizer, scheduler, root, prefix='train'):
    files = glob.glob(os.path.join(root, "{}_*.pth".format(prefix)))

    max_idx = -1
    for file in files:
        num = re.search("{}_(\d+).pth".format(prefix), file)
        if num is not None:
            num = num.group(1)
            max_idx = max(max_idx, int(num))

    if max_idx != -1:
        print(f'Find last training info: epoch {max_idx}')
        checkpoint = torch.load(
            os.path.join(root, "{}_{}.pth".format(prefix, max_idx)))
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        if model_ema is not None:
            if isinstance(model_ema, torch.nn.DataParallel):
                model_ema.module.load_state_dict(checkpoint["model_ema"])
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])
        
        epoch = checkpoint["epoch"]
        
        try:
            best_acc = checkpoint["best_acc"]
        except KeyError:
            best_acc = checkpoint["acc"]
        
        ###################################
        if "optimizer" in checkpoint.keys():
            print(f"==> Resume epoch {epoch}.. ")
            optimizer.load_state_dict(checkpoint["optimizer"]) 
            optimizer.step()
            scheduler.step()
        else:
            for i in range(epoch):
                print(f'==> Restore epoch {epoch}..')
                print(f'[epoch {i+1}] lr = {optimizer.param_groups[3]["lr"]:.6f}  (restored)')
                scheduler.step()
        ###################################
        
        return (epoch, best_acc)
    else:
        print("==> Can't find checkpoint...training from initial stage")
        return (0, 0)


def get_tau(tau_init, tau_target, total_epoch, cur_epoch, scaling_type):
    if scaling_type == "exponential":
        beta = math.pow(tau_target/tau_init, (cur_epoch-1)/(total_epoch-1))
        tau = tau_init * beta
    elif scaling_type == "cosine":
        tau = tau_target + 1/2*(tau_init-tau_target)*(1 + math.cos(math.pi * (cur_epoch-1)/(total_epoch-1)))
    elif scaling_type == "linear":
        tau = tau_init - (cur_epoch-1) * (tau_init - tau_target) / (total_epoch-1)
    elif scaling_type == "exp_cyclic":
        raise ValueError
    string_to_log = f"tau   = {tau:.4f} (init = {tau_init:.2f} -> target = {tau_target:.2f})"

    return tau, string_to_log


def get_alpha(tau_init, tau_target, total_epoch, cur_epoch, scaling_type):
    if scaling_type == "exponential":
        beta = math.pow(tau_target/tau_init, (cur_epoch-1)/(total_epoch-1))
        tau = tau_init * beta
    elif scaling_type == "cosine":
        tau = tau_target + 1/2*(tau_init-tau_target)*(1 + math.cos(math.pi * (cur_epoch-1)/(total_epoch-1)))
    elif scaling_type == "linear":
        tau = tau_init - (cur_epoch-1) * (tau_init - tau_target) / (total_epoch-1)
    string_to_log = f"[Epoch {cur_epoch}] alpha = {tau:.4e} (init = {tau_init:.2e} -> target = {tau_target:.2e})"
    
    return tau, string_to_log


def get_exp_cyclic_annealing_tau(cycle_size_iter, temp_step, n, tau_init=1):
    """
    This function return the exp annealing function for the gumbel softmax.
    :param cycle_size_iter: integer that defies the cycle size
    :param temp_step: the step size coefficient
    :param n: a float scaling of the iteration index
    :return: a function which get an index and return a floating temperature value
    """
    def temp_func(i):
        if i < 0:
            return 1.0 #tau_init
        i = i % cycle_size_iter
        return np.maximum(0.5, 1 * np.exp(-temp_step * np.round(i / n))) #tau_init
    
    return temp_func


#def get_linear_annealing_tau(cycle_size_iter, temp_step, n, tau_init=1, tau_target=0.2):
#    def temp_func(i):
#        if i < 0:
#            return 1.0
#        i = 

def set_tau(QuantOps, model, tau):
    for module in model.modules():
        if isinstance(module, (QuantOps.Conv2d, QuantOps.ReLU, QuantOps.Sym, QuantOps.Linear, QuantOps.ReLU6)):
            module.tau = tau

def remove_gumbel(QuantOps, model):
    for module in model.modules():
        if isinstance(module, (QuantOps.Conv2d, QuantOps.ReLU, QuantOps.Sym, QuantOps.Linear, QuantOps.ReLU6)):
            module.gumbel_noise = False