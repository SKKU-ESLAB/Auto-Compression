import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import random
import time
import logging
import locale

from models import *
from utils import *
from functions import *
from torch.autograd import Variable
from torch import Tensor
from models.MobileNetV2_quant import mobilenet_v2

parser = argparse.ArgumentParser(description='PyTorch - Learning Quantization')
parser.add_argument('--model', default='mobilenetv2', help='select model')
parser.add_argument('--dir', default='/data', help='data root')
parser.add_argument('--dataset', default='imagenet', help='select dataset')
#parser.add_argument('--load', default=0, type=int, help='0: no load, 1: resume, 2: load model, 3: load weight')
#parser.add_argument('--load_file', default='./checkpoint/trained.pth', help='select loading file')
parser.add_argument('--batchsize', default=128, type=int, help='set batch size')
#parser.add_argument('--epoch', default=160, type=int, help='number of epochs tp train for')
parser.add_argument('--warmup', default=5, type=int)
parser.add_argument('--ft_epoch', default=15, type=int)
parser.add_argument('--wd', default=1e-4, type=float, help='set weight decay value')
parser.add_argument('--seed', default=7, type=int, help='random seed')
#parser.add_argument('--strict_false', action='store_true', help='load_state_dict option "strict" False')
parser.add_argument('--m', default=0.9, type=float, help='set momentum value')
parser.add_argument('--lr_ms', nargs='+', default=[20, 40, 60], type=int, help='set milestones')
parser.add_argument('--lr_g', default=0.1, type=float, help='set gamma for multistep lr scheduler')
parser.add_argument('--cosine', action='store_true', help='set the lr scheduler to cosine annealing')
parser.add_argument('--workers', default=8, type=int, help='set number of workers')
parser.add_argument('--savefile', default='ckpt.pth', help='save file name')
parser.add_argument('--log_interval', default=50, type=int, help='logging interval')
parser.add_argument('--exp', default='test', type=str)
parser.add_argument("--quant_op")

#parser.add_argument('--lr1', default=0.01, type=float, help='set learning rate value1')
#parser.add_argument('--lr2', default=0.001, type=float, help='set learning rate value2')
#parser.add_argument('--lr3', default=0.0001, type=float, help='set learning rate value3')
parser.add_argument("--lr", default=0.04, type=float)

parser.add_argument('--comp_ratio', default=0.005, type=float, help='set target compression ratio of FLOPs loss')
parser.add_argument('--scaling', default=1e-6, type=float, help='set FLOPs loss scaling factor')
parser.add_argument('--w_bit', default=[8], type=int, nargs='+', help='set weight bits')
parser.add_argument('--a_bit', default=[8], type=int, nargs='+', help='set activation bits')

#parser.add_argument('--lq_mode', '-lq', action='store_true', help='learning quantization mode')
parser.add_argument('--lb_mode', '-lb', action='store_true', help='learn bitwidth (dnas approach)')
#parser.add_argument('--is_qt', '-q', action='store_true', help='quantization')
#parser.add_argument('--gamma', '-g', action='store_true', help='trainable gamma factor')
parser.add_argument('-eval', action='store_true', help='evaluation mode')
parser.add_argument('-initskip', action='store_true', help='skip initialization (for loading cw, dw? maybe..')
#parser.add_argument('-fwbw', action='store_true', help='use filter-wise bitwidth')
#parser.add_argument('-fwlq', action='store_true', help='use filter-wise quantization interval learning')
parser.add_argument('-sep_bitops', action='store_true', help='separate bitwidth calculation from forward()')
args = parser.parse_args()
args.save = f'logs/{args.dataset}/{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
create_exp_dir(args.save)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if len(args.w_bit)==1:
    print("Fixed bitwidth for weight")

if len(args.a_bit)==1:
    print("Fixed bitwidth for activation")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

best_acc = 0
start_epoch = 1
end_epoch = args.ft_epoch


# Dataloader
print('==> Preparing Data..')
train_loader, val_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)

print('==> Building Model..')
if args.lb_mode:
    print("Learning layer-wise bitwidth.")
else:
    print("Fixed bitwidth.")

# QuantOps
if args.quant_op == "duq":
    if args.sep_bitops:
        from functions.duq_sep_bitops import *
    else:
        from functions.duq import *
    print("==> differentiable and unified quantization method is selected..")
elif args.quant_op == "qil":
    torch.autograd.set_detect_anomaly(True)
    from functions.qil import * 
    print("==> quantization interval learning method is selected..")
elif args.quant_op == "lsq":
    from functions.lsq import *
    print("==> learning step size method is selected..")
elif args.quant_op == 'duq_wo_scale':
    from functions.duq_wo_scale import *
    print("==> differentiable and unified quantization without scale.. ")
elif args.quant_op == 'duq_w_offset':
    from functions.duq_w_offset import *
    print("==> differentiable and unified quantization with offset.. ")
elif args.quant_op == 'duq_init_change':
    from functions.duq_init_change import *
else:
    raise NotImplementedError

def calc_bitops(model):      
    a_bit_list = [32]
    w_bit_list = []
    compute_list = []

    for module in model.modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
            if isinstance(module.bits, int) :
                a_bit_list.append(module.bits)
            else:
                softmask = F.gumbel_softmax(module.theta, tau=1, hard=False)
                a_bit_list.append((softmask * module.bits).sum())

        elif isinstance(module, (Q_Conv2d, Q_Linear)):
            if isinstance(module.bits, int) :
                w_bit_list.append(module.bits)
            else:
                softmask = F.gumbel_softmax(module.theta, tau=1, hard=False)
                w_bit_list.append((softmask * module.bits).sum())
                
            compute_list.append(module.computation)

    cost = (Tensor(a_bit_list) * Tensor(w_bit_list) * Tensor(compute_list)).sum(dim=0, keepdim=True)
    #print(cost.shape)
    return cost


# bitops_total
def get_bitops_total():
    model_ = mobilenet_v2(QuantOps)
    model_ = model_.to(device)
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).cuda()
    else:
        input = torch.randn([1,3,224,224]).cuda()
    model_.train()

    if args.sep_bitops:
        print('==> separate bitops calculation')
        QuantOps.initialize(model_, train_loader, 32, weight=True)
        QuantOps.initialize(model_, train_loader, 32, act=True)
        bitops = calc_bitops(model_)
    else:
        out, bitops =  model_(input)

    return bitops

bitops_total = get_bitops_total()
print(f'bitops_total: {int(bitops_total):d}')
print(f'bitops_targt: {int(bitops_total * args.comp_ratio):d}')


# model
if args.model == "mobilenetv2":
    model = mobilenet_v2(QuantOps)
    model.load_state_dict(torch.load("./checkpoint/mobilenet_v2-b0353104.pth"), False)
else:
    raise NotImplementedError
model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f'==> DataParallel: device count = {torch.cuda.device_count()}')
    model = torch.nn.DataParallel(model) #, device_ids=range(torch.cuda.device_count()))


# optimizer
def get_optimizer(params, train_quant, train_weight, train_bnbias, lr_decay=1):
    (quant, skip, weight, bnbias) = params
    optimizer = optim.SGD([
        {'params': skip, 'weight_decay': 0, 'lr': 0},
        {'params': quant, 'weight_decay': 0., 'lr': args.lr * 1e-2 * lr_decay if train_quant else 0},
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr * lr_decay if train_bnbias else 0},
        {'params': weight, 'weight_decay': args.decay, 'lr': args.lr * lr_decay if train_weight else 0},
    ], momentum=0.9, nesterov=True)
    return optimizer

optimizer = optim.SGD(model.parameters(), lr=args.lr)


# scheduler
scheduler = CosineWithWarmup(optimizer, 
        warmup_len=args.warmup, warmup_start_multiplier=0.1,
        max_epochs=args.ft_epoch, eta_min=1e-3)

criterion = nn.CrossEntropyLoss()


# bitwidth Initilization
with torch.no_grad():
    
    print('==> weight bitwidth is set up..')
    QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
    print('==> activation bitwidth is set up..')
    QuantOps.initialize(model, train_loader, args.a_bit, act=True)


# Training
def train(epoch):
    print('train:')
    model.train()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    bitops_target = bitops_total * args.comp_ratio
    
    end = t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time()

        if args.sep_bitops:
            outputs = model(inputs)
            bitops = calc_bitops(model)
        else:
            outputs, bitops = model(inputs)

        if not isinstance(bitops, (float, int)):
            #print(bitops)
            bitops = bitops[0]

        loss = criterion(outputs, targets)
        eval_acc_loss.update(loss.item(), inputs.size(0))
        
        if args.lb_mode:
            loss_bitops = torch.abs((bitops-bitops_target)*args.scaling)
            if (batch_idx) % args.log_interval == 0:
                print(f'bitops-bitops_target: {bitops-bitops_target}')
            loss_bitops = loss_bitops.reshape(torch.Size([]))
            loss += loss_bitops 
            eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))

        acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_time = time.time()
        
        if (batch_idx) % args.log_interval == 0:
            logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                    'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ' +  \
                    'Data Time: %.3f s | Model Time: %.3f s',   # \t Memory %.03fMB',
                epoch, batch_idx * len(inputs),
                len(train_loader.dataset),
                eval_acc_loss.avg, eval_bitops_loss.avg, top1.avg, top5.avg,
                data_time - end, model_time - data_time)
        end = time.time()

    if args.is_qt:
        i=1
        str_to_log = '\n'
        str_to_print = f'Epoch {epoch} Bitwidth selection: \n'
        for _, m in enumerate(model.modules()):
            if isinstance(m, lq_conv2d_orig):
                i += 1
                if isinstance(args.w_bit, list):
                    prob_w = F.softmax(m.theta_w)
                    sel=torch.argmax(prob_w)
                    str_to_print += f'{args.w_bit[sel]}'
                    prob_w = [f'{i:.5f}' for i in prob_w.cpu().tolist()]
                    str_to_log += f'layer {i} theta_w: [{", ".join(prob_w)}]\n'
                else:
                    break
        logging.info(str_to_log)
        logging.info(str_to_print)
        
        i=1
        str_to_log = '\n'
        for _, m in enumerate(model.modules()):
            if isinstance(m, lq_conv2d_orig):
                i += 1
                if isinstance(args.a_bit, list):
                    prob_x = F.softmax(m.theta_x).cpu().tolist()
                    prob_x = [f'{i:.5f}' for i in prob_x]
                    str_to_log += f'layer {i} theta_x: [{", ".join(prob_x)}]\n'
                else:
                    break
                #m.dw.data = m.dw.clamp(1e-4, m.cw.data[0] - 1e-4)
                #m.dx.data = m.dx.clamp(1e-2, m.cx.data[0] - 1e-2)
        logging.info(str_to_log)
    t1 = time.time()
    print(f'epoch time: {t1-t0:.3f} s')


def eval(epoch):
    print('eval:')
    global best_acc
    model.eval()
    eval_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            if args.sep_bitops:
                outputs = model(inputs)
                bitops = calc_bitops(model)
            else:
                outputs, bitops = model(inputs)
            loss = criterion(outputs, targets)

            # TODO: bitops loss
            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            eval_loss.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

        logging.info('Loss: %.4f | top1.avg: %.3f%% | top5.avg: %.3f%%' % (eval_loss.avg, top1.avg, top5.avg))
        # Save checkpoint.
        if top1.avg > best_acc:
            best_acc = top1.avg
            if isinstance(model, torch.nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            state = {
                'model': model_state,
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, f'{args.save}/{args.exp}_best_{args.savefile}')

        if (epoch % 10) == 0:
            if isinstance(model, torch.nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            state = {
                'model': model_state,
                'acc': top1.avg,
                'epoch': epoch,
            }
            torch.save(state, f'{args.save}/{args.exp}_{args.savefile}')
    print(f'evalaution time bitops: {bitops[0]}')

if args.eval:
    eval(0)

else:
    for epoch in range(start_epoch, end_epoch+1):
        logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
        train(epoch)
        eval(epoch)
        scheduler.step()
        #print_param(model)
        if epoch == end_epoch:
            if args.is_qt:
                i=1
                str_to_log = 'Final bitwidth selection: \n'
                for _, m in enumerate(model.modules()):
                    if isinstance(m, lq_conv2d_orig):
                        i += 1
                        if isinstance(args.w_bit, list):
                            prob_w = F.softmax(m.theta_w)
                            sel=torch.argmax(prob_w)
                            str_to_log += f'{args.w_bit[sel]}'
                            #prob_w = [f'{i:.5f}' for i in prob_w.cpu().tolist()]
                            #str_to_log += f'layer {i} theta_w: [{", ".join(prob_w)}]\n'
                        else:
                            break
                logging.info(str_to_log)


logging.info('Best accuracy : {:.3f} %'.format(best_acc))
#print_param(model)
