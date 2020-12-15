import os
import sys
import time
import math
import json
import logging


import torch
import torch.nn as nn
import torch.nn.init as init

import numpy as np
from models import  AlignedGroupedMaskedMLP, AlignedGroupedMaskedConv2d, FilterMaskedConv2d

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def one_hot(ids, n_class):
    assert len(ids.shape) == 1, 'the ids should be 1-D'

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor

def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()


def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def print_layer_keep_ratio(model, logger):
    total = 0.
    keep = 0.
    for layer in model.modules():
        if isinstance(layer, AlignedGroupedMaskedMLP):
            weight_shape = layer.weight.shape
            threshold_shape = layer.threshold.shape
            threshold = layer.threshold.view(threshold_shape[0], -1)
            weight = layer.weight.view(threshold_shape[0], layer.group_shape[0], -1, layer.group_shape[1])
            if layer.grouped_rule == 'l1':
                grouped_weight = weight.abs().sum(dim=(1, 3))
            elif layer.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
            grouped_weight = grouped_weight - threshold
            grouped_mask = layer.step(grouped_weight)
            #ratio = torch.sum(grouped_mask) / grouped_mask.numel()
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))

        if isinstance(layer, AlignedGroupedMaskedConv2d):
            weight_shape = layer.weight.shape
            threshold_shape = layer.threshold.shape
            threshold = layer.threshold.view(threshold_shape[0], -1) # (out_c, 1)
            weight = layer.weight.view(threshold_shape[0], layer.group_shape[0], -1, layer.group_shape[1]) # (out_c, 1, -1, 4)
            if layer.grouped_rule == 'l1':
                grouped_weight = weight.abs().sum(dim=(1, 3))
            elif layer.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).sum(dim=(1, 3)).sqrt()
            grouped_weight = grouped_weight - threshold #(out_c, -1) ==(out_c, in_c/4)
            grouped_mask = layer.step(grouped_weight)
            #ratio = torch.sum(grouped_mask) / grouped_mask.numel()
            total += grouped_mask.numel()*layer.group_shape[1] # 4 
            keep += torch.sum(grouped_mask)*layer.group_shape[1]
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))

        if isinstance(layer, nn.Linear):
            """어차피 pruning안할거니까 sum으로 갯수세기"""
            weight_shape = layer.weight.shape
            weight = layer.weight.abs()
            mask = BinaryStep.apply(weight)
            #ratio = torch.sum(mask) /  mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))

        if isinstance(layer, nn.Conv2d):
            """어차피 pruningX"""
            weight_shape = layer.weight.shape
            weight = layer.weight.view(weight_shape[0], -1).abs()
            mask = BinaryStep.apply(weight)
            #ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))

        if isinstance(layer, FilterMaskedConv2d):
            weight_shape = layer.weight.shape
            threshold_shape = layer.threshold.shape
            threshold = layer.threshold.view(threshold_shape[0]) # (out, 1)
            weight = layer.weight.view(weight_shape[0],-1) #(out_c, in_c)
            if layer.grouped_rule == 'l1':
                grouped_weight = weight.abs().mean(dim=1)#weight.abs().mean(dim=(1))
            elif layer.grouped_rule == 'l2':
                grouped_weight = weight.pow(2).mean(dim=1).sqrt()
            grouped_weight = grouped_weight - threshold 
            grouped_mask = layer.step(grouped_weight) # (out_c, 1)
            #grouped_mask = grouped_mask.view(weight_shape[0],-1).repeat_interleave(weight.shape[1], dim=1)
            #ratio = torch.sum(grouped_mask) / (grouped_mask.numel()*weight.shape[1])
            total += grouped_mask.numel()*weight.shape[1]
            keep += torch.sum(grouped_mask)*weight.shape[1]
            #logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            logger.info("{}, keep ratio {:.4f}".format(layer, ratio))

    logger.info("Model keep ratio {:.4f}".format(keep/total))
    return keep / total

