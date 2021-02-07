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

parser.add_argument('--batchsize', default=64, type=int, help='set batch size')
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument('--warmup', default=3, type=int)
parser.add_argument('--ft_epoch', default=15, type=int)

parser.add_argument('--log_interval', default=50, type=int, help='logging interval')
parser.add_argument('--exp', default='test', type=str)
parser.add_argument('--seed', default=7, type=int, help='random seed')
parser.add_argument("--quant_op", required=True)

#parser.add_argument('--comp_ratio', default=1, type=float, help='set target compression ratio of Bitops loss')
parser.add_argument('--target_w', default=4, type=float, help='set target weight bitwidth')
parser.add_argument('--target_a', default=4, type=float, help='set target activation bitwidth')
parser.add_argument('--scaling', default=1e-6, type=float, help='set FLOPs loss scaling factor')
parser.add_argument('--w_bit', default=[32], type=int, nargs='+', help='set weight bits')
parser.add_argument('--a_bit', default=[32], type=int, nargs='+', help='set activation bits')

parser.add_argument('--eval', action='store_true', help='evaluation mode')
parser.add_argument('--lb_off', '-lboff', action='store_true', help='learn bitwidth (dnas approach)')
parser.add_argument('--cooltime', default=0, type=int, help='seconds for processor cooling (for sv8 and sv9')
parser.add_argument('--w_ep', default=1, type=int, help='')
parser.add_argument('--t_ep', default=1, type=int, help='')
parser.add_argument('--alternate', action="store_true")

args = parser.parse_args()
if args.exp == 'test':
    args.save = f'logs/{args.dataset}/{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
else:
    args.save = f'logs/{args.dataset}/{args.exp}' #-{time.strftime("%y%m%d-%H%M%S")}'

args.workers = 8
args.momentum = 0.9   # momentum value
args.decay = 1e-4 # weight decay value
args.lb_mode = False
args.comp_ratio = args.target_w / 32. * args.target_a / 32
if (len(args.w_bit) > 1 or len(args.a_bit) > 1) and not args.lb_off:
    args.lb_mode = True
    print("## Learning bitwidth selection")

create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Argument logging ################## 
string_to_log = '==> parsed arguments.. \n'
for key in vars(args):
    string_to_log += f'  {key} : {getattr(args, key)}\n'
logging.info(string_to_log)


if len(args.w_bit)==1:
    print("## Fixed bitwidth for weight")

if len(args.a_bit)==1:
    print("## Fixed bitwidth for activation")

if args.lb_mode:
    print("## Learning layer-wise bitwidth.")



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

best_acc = 0
last_epoch = 0
end_epoch = args.ft_epoch


# Dataloader
print('==> Preparing Data..')
train_loader, val_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)


print('==> Building Model..')
# QuantOps
if args.quant_op == "duq":
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

# calculate bitops (theta-weighted)
def calc_bitops(model, full=False):
    a_bit_list = [32]
    w_bit_list = []
    compute_list = []

    for module in model.modules():
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
            if isinstance(module.bits, int) :
                a_bit_list.append(module.bits)
            else:
                softmask = F.gumbel_softmax(module.theta, tau=1, hard=False, dim=0)
                a_bit_list.append((softmask * module.bits).sum())
                '''
                for i in range(len(softmask)):
                    softmask[i] *= module.bits[i]
                a_bit_list.append(sum(softmask))
                '''

        elif isinstance(module, (Q_Conv2d, Q_Linear)):
            if isinstance(module.bits, int) :
                w_bit_list.append(module.bits)
            else:
                softmask = F.gumbel_softmax(module.theta, tau=1, hard=False, dim=0)
                w_bit_list.append((softmask * module.bits).sum())
                '''
                for i in range(len(softmask)):
                    softmask[i] *= module.bits[i]
                w_bit_list.append(sum(softmask))
                '''
            compute_list.append(module.computation)
    cost = (Tensor(a_bit_list) * Tensor(w_bit_list) * Tensor(compute_list)).sum(dim=0, keepdim=True)
    return cost


# calculate bitops for full precision
def get_bitops_total():
    model_ = mobilenet_v2(QuantOps)
    model_ = model_.to(device)
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).cuda()
    else:
        input = torch.randn([1,3,224,224]).cuda()
    model_.eval()
    QuantOps.initialize(model_, train_loader, 32, weight=True)
    QuantOps.initialize(model_, train_loader, 32, act=True)
    _, bitops = model_(input)

    return bitops
 
print("==> Calculate target bitops..")
bitops_first_layer= 11098128384
bitops_total = get_bitops_total()
bitops_target = (bitops_total-bitops_first_layer) * (args.target_w/32.) * (args.target_a/32.) + (bitops_first_layer * (args.target_w/32.))
logging.info(f'bitops_total: {int(bitops_total*1e-9):d}')
logging.info(f'bitops_targt: {int(bitops_target*1e-9):d}')
logging.info(f'bitops_wrong: {int(bitops_total * (args.target_w/32.) * (args.target_a/32.)):d}')


# model
if args.model == "mobilenetv2":
    model = mobilenet_v2(QuantOps)
    if not os.path.isfile("./checkpoint/mobilenet_v2-b0353104.pth"):
        os.system("wget -P ./checkpoint https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
    model.load_state_dict(torch.load("./checkpoint/mobilenet_v2-b0353104.pth"), False)
else:
    raise NotImplementedError
model = model.to(device)

# optimizer -> for further coding (got from PROFIT)
def get_optimizer(params, train_weight, train_quant, train_bnbias, train_theta):
    (weight, quant, bnbias, theta, skip) = params
    optimizer = optim.SGD([
        {'params': weight, 'weight_decay': args.decay, 'lr': args.lr  if train_weight else 0},
        {'params': quant, 'weight_decay': 0., 'lr': args.lr if train_quant else 0},
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr if train_bnbias else 0},
        {'params': theta, 'weight_decay': 0., 'lr': args.lr if train_theta else 0},
        {'params': skip, 'weight_decay': 0, 'lr': 0},
    ], momentum=args.momentum, nesterov=True)
    return optimizer


def categorize_param(model):
    weight = []
    quant = []
    bnbias = []
    theta = []
    skip = []
    for name, param in model.named_parameters():
        if name.endswith(".a") or name.endswith(".b") \
            or name.endswith(".c") or name.endswith(".d"):
            quant.append(param)
        elif len(param.shape) == 1 and (name.endswith('weight') or  name.endswith(".bias")):
            bnbias.append(param)
        elif name.endswith(".theta"):
            theta.append(param)
        else:
            weight.append(param)

    return (weight, quant, bnbias, theta, skip,)


# bitwidth Initilization
with torch.no_grad():
    print('==> weight bitwidth is set up..')
    QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
    print('==> activation bitwidth is set up..')
    QuantOps.initialize(model, train_loader, args.a_bit, act=True)

if torch.cuda.device_count() > 1:
    print(f'==> DataParallel: device count = {torch.cuda.device_count()}')
    model = torch.nn.DataParallel(model) #, device_ids=range(torch.cuda.device_count()))


# optimizer & scheduler
params = categorize_param(model)
optimizer = get_optimizer(params, True, True, True, True)
current_lr = -1

scheduler = CosineWithWarmup(optimizer, 
        warmup_len=args.warmup, warmup_start_multiplier=0.1,
        max_epochs=args.ft_epoch, eta_min=1e-3)

criterion = nn.CrossEntropyLoss()
#scaler = torch.cuda.amp.GradScaler()
#model, optimizer = amp.initialize(model, optimizer, opt_level="01")

# Training
def train(epoch):
    print('train:')

    for i in range(len(optimizer.param_groups)):
        print(f'[epoch {epoch}] optimizer, lr{i} = {optimizer.param_groups[i]["lr"]:.6f}')

        
    model.train()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    current_lr = optimizer.param_groups[0]['lr']
    
    end = t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.lb_mode and args.alternate:
            if batch_idx % 1000 == 0: # learning weight
                optimizer.param_groups[0]['lr'] = current_lr
                #optimizer.param_groups[1]['lr'] = current_lr * 1e-2
                optimizer.param_groups[3]['lr'] = 0 # 0:weight  1:quant  2:bnbias  3:theta
                
            elif batch_idx % 1000 == 800: # learning theta
                optimizer.param_groups[0]['lr'] = 0
                #optimizer.param_groups[1]['lr'] = 0
                optimizer.param_groups[3]['lr'] = current_lr # 0:weight  1:quant  2:bnbias  3:theta
            
            #if batch_idx == 2:
            #    break
        inputs, targets = inputs.to(device), targets.to(device)
        data_time = time.time()
        with torch.cuda.amp.autocast():
            outputs, bitops = model(inputs)

            loss = criterion(outputs, targets)
            eval_acc_loss.update(loss.item(), inputs.size(0))
            
            if args.lb_mode and optimizer.param_groups[3]['lr'] != 0 :#(epoch-1) % (args.w_ep + args.t_ep) >= args.w_ep:
                if not isinstance(bitops, (float, int)):
                    bitops = bitops.mean()
                loss_bitops = bitops*args.scale
                loss_bitops = loss_bitops.reshape(torch.Size([]))
                loss += loss_bitops 
                eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))

            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        #scaler.step(optimizer)
        #scaler.update()

        model_time = time.time()
        if (batch_idx) % args.log_interval == 0:
            logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                    'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ' +  \
                    'Data Time: %.3f s | Model Time: %.3f s',   # \t Memory %.03fMB',
                epoch, batch_idx * len(inputs),
                len(train_loader.dataset),
                eval_acc_loss.avg, eval_bitops_loss.avg if optimizer.param_groups[3]['lr'] !=0 else 0, 
                top1.avg, top5.avg,
                data_time - end, model_time - data_time)
            if args.cooltime and epoch != end_epoch:
                print(f'> [sleep] {args.cooltime}s for cooling GPUs.. ', end='', flush=True)
                time.sleep(args.cooltime)
                print('done.')
        

        optimizer.param_groups[3]['lr']
        end = time.time()
    optimizer.param_groups[0]['lr'] = current_lr
    optimizer.param_groups[3]['lr'] = current_lr 

    if args.lb_mode:
        i=1
        str_to_log = '\n'
        str_to_print = f'Epoch {epoch}, weight bitwidth selection probability: \n'
        for _, m in enumerate(model.modules()):
            if isinstance(m, (Q_Conv2d, Q_Linear)):
                i += 1
                if len(m.bits) > 1:
                    prob_w = F.softmax(m.theta)
                    sel=torch.argmax(prob_w)
                    str_to_print += f'{args.w_bit[sel]}, '
                    prob_w = [f'{i:.5f}' for i in prob_w.cpu().tolist()]
                    str_to_log += f'layer {i} [{" ".join(prob_w)}]\n'
        logging.info(str_to_print)
        logging.info(str_to_log)
        
        i=1
        str_to_log = '\n'
        str_to_print = f'Epoch {epoch}, activation bitwidth selection probability: \n'
        for _, m in enumerate(model.modules()):
            if isinstance(m, (Q_ReLU, Q_Sym, Q_HSwish)):
                i += 1
                if len(m.bits) > 1:
                    prob_a = F.softmax(m.theta)
                    sel=torch.argmax(prob_a)
                    str_to_print += f'{args.a_bit[sel]} '
                    prob_a = [f'{i:.5f}' for i in prob_a.cpu().tolist()]
                    
                    # TODO: more readable print
                    str_to_log += f'layer {i} [{" ".join(prob_a)}]\n'
        logging.info(str_to_print)
        logging.info(str_to_log)

    t1 = time.time()
    print(f'epoch time: {t1-t0:.3f} s')


def eval(epoch):
    print('eval:')
    global best_acc
    model.eval()
    eval_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, bitops = model(inputs)
            loss = criterion(outputs, targets)
            if args.lb_mode:
                #bitops = calc_bitops(model)
                if not isinstance(bitops, (float, int)):
                    loss_bitops = bitops*args.scaling*1e-9
                if (batch_idx) % (args.log_interval*5) == 0:
                    logging.info(f'bitops_target:   {bitops_target*1e-9}')
                    logging.info(f'evalaution time bitops: {bitops*1e-9}')
                loss_bitops = loss_bitops.reshape(torch.Size([]))
                loss += loss_bitops 
                eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))

            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            eval_loss.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if (batch_idx) % args.log_interval == 0:
                logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                        'L_acc: %.3cf | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ',
                    epoch, batch_idx * len(inputs),
                    len(val_loader.dataset),
                    eval_loss.avg, eval_bitops_loss.avg, top1.avg, top5.avg)
                if args.cooltime and epoch != end_epoch:
                    print(f'> [sleep] {args.cooltime}s for cooling GPUs.. ', end='')
                    time.sleep(args.cooltime)
                    print('done.')

        logging.info('L_acc: %.4f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%%' \
                    % (eval_loss.avg, eval_bitops_loss.avg, top1.avg, top5.avg))
        
        # Save checkpoint.        
        is_best = False
        if top1.avg > best_acc:
            is_best = True
            best_acc = top1.avg
        
        create_checkpoint(model, None, optimizer, is_best, None, 
                          top1.avg, best_acc, epoch, args.save, 1, args.exp)
    


if args.eval:
    eval(0)

else:
    last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler, 
                                    args.save, args.exp)
    for epoch in range(last_epoch+1, end_epoch+1):
        logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
        train(epoch)
        eval(epoch)
        scheduler.step() 
        

logging.info('Best accuracy : {:.3f} %'.format(best_acc))
