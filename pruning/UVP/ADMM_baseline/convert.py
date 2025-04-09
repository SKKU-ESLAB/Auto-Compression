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
#import torchvision.models as models
import numpy as np

from models_old.mobilenet_v1 import MobileNetV1
import models
import math
from utils import get_admm_loss, initialize_perm_list, initialize_Z_and_U, update_X, update_Z, update_U, print_prune, apply_prune
from data import get_dataset
#from latency_predictor import Pixel1LatencyPredictor

#from get_model_from_gluoncv import get_pretrained_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
#parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                    help='number of total epochs to run')
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
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pretrained-model', default=None, type=str)
#parser.add_argument('--seed', default=None, type=int,
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
parser.add_argument('--vs', default='ved', choices=['ved', 'greedy', 'optimal'])
parser.add_argument('--width-mult', default=1.0, type=float)
parser.add_argument('--sparsity-method', default='uniform', choices=['gt', 'uniform'])
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

parser.add_argument('--baseline', dest='baseline', action='store_true')


best_acc1 = 0
best_acc5 = 0

def main():
    global best_acc1
    global best_acc5
    args = parser.parse_args()
    if 'mobilenet' in args.arch:
        width_mult_str = f"_{args.width_mult}"
    else:
        width_mult_str = ""
    groups = f"{args.arch}{width_mult_str}_ts{args.target_sparsity}_lr{args.lr}_admm{args.admm_epochs}_ft{args.ft_epochs}_v{args.vector_size}"
    if args.unaligned:
        groups = groups + f"_u_{args.vs}"
    else:
        groups = groups + "_a"
    if args.cp:
        groups = groups + "_cp"
    if args.repeat:
        groups = groups + "_repeat"
    if args.cp_ft:
        groups = groups + "_cpft"
    args.name = groups

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
        model = MobileNetV1(num_classes=1000, input_size=224, width_mult=1.0)
    else:
        model = models.__dict__[args.arch]()

    if args.pretrained_model is not None and False:
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained_model)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pretrained_model, map_location=loc)

        if args.arch == 'mobilenet_v2':
            fc_weight = checkpoint['classifier.1.weight']
            checkpoint['classifier.1.weight'] = fc_weight.view(
                    fc_weight.size(0), fc_weight.size(1), 1, 1)
            model.load_state_dict(checkpoint, strict=True)
        elif args.arch == 'mobilenet_v1':
            fc_weight = checkpoint['state_dict']['classifier.0.weight']
            checkpoint['state_dict']['classifier.0.weight'] = fc_weight.view(
                    fc_weight.size(0), fc_weight.size(1), 1, 1)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if checkpoint[name+'.weight'].dim() == 2:
                        weight = checkpoint[name+'.weight']
                        checkpoint[name+'.weight'] = weight.view(weight.size(0), weight.size(1), 1, 1)
            model.load_state_dict(checkpoint, strict=True)
        elif args.arch == 'resnet50':
            fc_weight = checkpoint['fc.weight']
            checkpoint['fc.weight'] = fc_weight.view(
                    fc_weight.size(0), fc_weight.size(1), 1, 1)
            model.load_state_dict(checkpoint, strict=True)

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

    if args.baseline:
        args.name = f"{args.arch}_baseline"

    save_dir = os.path.join('output', args.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    if not args.baseline:
        perm_list = None
        if perm_list is None:
            perm_list = initialize_perm_list(model, args)
        handles = apply_prune(model, args, perm_list)

        for handle in handles:
            handle.remove()
    handles = []


    batch_size = 1
    channels = 3
    height = 512
    width = 512
    model = model.eval()
    sample_input = torch.rand((batch_size, channels, height, width))

    # Convert Final Conv to FC
    if args.arch == 'resnet50':
        fc_weight = model.module.fc.weight
        out_features, in_features, _, _ = fc_weight.shape
        fc_weight = fc_weight.view(out_features, in_features)
        fc_bias = model.module.fc.bias
        model.module.fc = nn.Linear(in_features, out_features, bias=True, device=fc_weight.device)
        model.module.fc.weight.data = fc_weight
        model.module.fc.bias.data = fc_bias

    # Convert AvgPool2d to mean
    avg_pool2d_dict = {}

    def replace_global_avg_pool2d(child, input):
        input_size = input[0].size(-2), input[0].size(-1)
        kernel_size = child.kernel_size if isinstance(child.kernel_size, tuple) \
            else (child.kernel_size, child.kernel_size)

        # If input_size and kernel_size are equal, replace to global average pooling
        if  kernel_size == input_size:
            module, name = avg_pool2d_dict[child]
            setattr(module, name, nn.AdaptiveAvgPool2d((1, 1)))

    def check_global_avg_pool2d(module):
        for name, child in module.named_children():
            if isinstance(child, nn.AvgPool2d):
                avg_pool2d_dict[child] = (module, name)
                handle = child.register_forward_pre_hook(replace_global_avg_pool2d)
                handles.append(handle)
            else:
                check_global_avg_pool2d(child)

    #check_global_avg_pool2d(model)
    output = model(sample_input)

    for handle in handles:
        handle.remove()

    print(model)

    # PyTorch to ONNX
    onnx_model_path = os.path.join(save_dir, "model.onnx")
    torch.onnx.export(
        model.module.cpu(),
        sample_input,
        onnx_model_path,
        opset_version=12,
        input_names=['input'],
        output_names=['output']
    )

    # ONNX to TF
    import onnx2tf
    tf_model_path = os.path.join(save_dir, "model.tf")
    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=tf_model_path,
        #copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
        output_signaturedefs=True,
    )

    def representative_dataset():
        inputs = [np.ones([1, 224, 224, 3], dtype=np.float32) for _ in range(10)]
        outputs = [np.ones([1, 1000], dtype=np.float32) for _ in range(10)]

        for input, output, in zip(inputs, outputs):

            yield {
                "input": input,
                "output": output,
            }

    # TF to TFLite
    import tensorflow as tf
    tflite_model_path = os.path.join(save_dir, "model.tflite")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]

    tflite_model = converter.convert()
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()
