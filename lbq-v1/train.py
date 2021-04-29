import argparse
import locale
import logging
import wandb
import os
import random
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable

from functions import *
from models import *
from models.MobileNetV2_quant import mobilenet_v2
from utils import *
from optimizer.radam import RAdam
import math
from torch.cuda import amp

### Common Arguments ----------------------------
parser = argparse.ArgumentParser(description='PyTorch - Learning Quantization')
parser.add_argument('--eval', action='store_true', help='evaluation mode')
parser.add_argument('--log_dir', type=str, default='/home/ubuntu/lkj/wandb_logs/',
                        help='Weights and Bias logging folder path')
parser.add_argument('--model', default='mobilenetv2', help='select model')
parser.add_argument('--dir', default='/data', help='data root')
parser.add_argument('--dataset', default='imagenet', help='select dataset')
parser.add_argument('--batchsize', default=64, type=int, help='set batch size')
parser.add_argument("--lr", default=0.025, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup', default=0, type=int)
parser.add_argument('--log_interval', default=100, type=int, help='logging interval')
parser.add_argument('--exp', default='test', type=str)
parser.add_argument('--seed', default=7, type=int, help='random seed')
# (note: discretize_epoch is added below.)
# TODO: Make gumbel process end-to-end
#       Now gumbel training process  is seperated as 'search' phase and 'retrain' phase.
#       Phases are discerned using existence of args.retrain_path.


### Quantization Arguments ----------------------
#parser.add_argument("--lr_quant", default=0.025, type=float)
#parser.add_argument("--lr_w_theta", default=0.025, type=float)
#parser.add_argument("--lr_a_theta", default=0.025, type=float)
#parser.add_argument("--lr_bn", default=0.025, type=float)
#parser.add_argument('--comp_ratio', default=1, type=float, help='set target compression ratio of Bitops loss')
parser.add_argument("--quant_op", required=True)
parser.add_argument('--w_target_bit', default=4, type=float, help='target weight bitwidth')
parser.add_argument('--a_target_bit', default=4, type=float, help='target activation bitwidth')
parser.add_argument('--w_bit', default=[4], type=int, nargs='+', help='candidate weight bits')
parser.add_argument('--a_bit', default=[4], type=int, nargs='+', help=' candidate activation bits')



### Gumbel Arguments -----------------------------
parser.add_argument('--lb_off', '-lboff', action='store_true', help='learn bitwidth (dnas approach)')
parser.add_argument('--w_ep', default=1, type=int, help='')
parser.add_argument('--t_ep', default=1, type=int, help='')
parser.add_argument('--alternate', action="store_true")
parser.add_argument('--retrain_path', default='', type=str, help='logged weight path to retrain')
parser.add_argument('--retrain_type', default=1, type=int, help='1: init weight using searched net\n'\
                                                            '2: init weight using pretrained\n'\
                                                            '3: init weight using searched net, reinit quantizer\n'\
                                                            '4: init weight using searched net, fixed bitwidth')
parser.add_argument('--fasttest', action='store_true')
parser.add_argument('--grad_scale', action='store_true')
#parser.add_argument('--lr_q_scale', default=1, type=float, help='lr_quant = args.lr * args.lr_q_scale')
parser.add_argument('--tau_init', default=1, type=float, help='softmax tau (initial)')
parser.add_argument('--tau_target', default=0.2, type=float, help='softmax tau (final)')
parser.add_argument('--tau_type', default='linear', type=str, help='softmax tau annealing type [linear, exponential, cosine, exp_cyclic]')
parser.add_argument('--alpha_init', default=1e+0, type=float, help='bitops scaling factor (initial)')
parser.add_argument('--alpha_target', default=1e+4, type=float, help='bitops scaling factor (final)')
parser.add_argument('--alpha_type', default='linear', type=str, help='bitops scaling factor annealing type [linear, exponential, cosine]')
parser.add_argument('--cycle_epoch', default=5, type=int, help='Training Cycle size until changing target compression')
parser.add_argument('--temp_step', default=1e-2, type=float, help='Exp factor of the temperature decay')
parser.add_argument('--n_gumbel', default=25, type=int, help='Number until rounding of single temperature step')
parser.add_argument('--optimizer', default='SGD', help='loss function optimizer [SGD, RAdam]')
parser.add_argument('--gumbel_end', action='store_true', help='Number of early epochs that use gumbel noise')
parser.add_argument('--tau_end', action='store_true', help='After gumbel_epochs, decrease tau linearly')
parser.add_argument('--amp', action='store_true', help='Use torch.cuda.amp')



### NEW ARGUMENTS 2021.04.27 ------------------------
parser.add_argument('--search_type', default='gumbel', type=str, choices=['gumbel', 'interpolation', 'direct'], 
                                                        help='one in [gumbel, interpolation, direct]')
parser.add_argument('--qparam', default=0, type=int, choices=[0,1,2,3,4],
                                        help='<direct learning> quantization parameter pairs.\n'\
                                             'an index for [(bw, stepsize), (bw, qmax), (nlvs, stepsize), (nlvs, qmax), (stepsize, qmax)]')
parser.add_argument('--interpolation_level', default=0, type=int, choices=[0,1,2],
                                        help='<interpolation learning> an index for [qresult, stepsize, nlvs]')
parser.add_argument('--window_size', default=2, type=int, choices=[2, 4, 6],
                                        help='<interpolation learning> window size ')
parser.add_argument('--discretize_epoch', default=25, type=int, help='Epoch to execute bitwidth discretization')
args = parser.parse_args()


#------------ For debuging and testing ------------

#args.stop_step = 101
#args.lr_a_theta = 0.01
#args.lr_w_theta = 0.01
#args.batchsize = 8

#--------------------------------------------------
args.lr_quant = args.lr
args.lr_w_theta = args.lr
args.lr_a_theta = args.lr
args.lr_bn = args.lr

PROJECT_NAME = 'LBQ'


if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
args.save = f'logs/{args.dataset}/{args.exp}' 

args.gumbel_epochs = (args.epochs - args.cycle_epoch) if args.gumbel_end else 10000


args.workers = 12
args.momentum = 0.9   # momentum value
args.decay = 5e-4 # weight decay value
args.lb_mode = False
args.comp_ratio = args.w_target_bit / 32. * args.a_target_bit / 32.


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

# Argument logging ####################
string_to_log = '==> parsed arguments.. \n'
for key in vars(args):
    string_to_log += f'  {key} : {getattr(args, key)}\n'
logging.info(string_to_log)


if len(args.w_bit)==1:
    print("## Fixed bitwidth for weight")

if len(args.a_bit)==1:
    print("## Fixed bitwidth for activation")

if args.lb_mode:
    logging.info("## Learning layer-wise bitwidth.")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

best_acc = 0
last_epoch = 0
end_epoch = args.epochs
tau = args.tau_init
alpha = args.alpha_init

# Dataloader
print('==> Preparing Data..')
train_loader, val_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)


print('==> Building Model..')
# QuantOps
if args.grad_scale:
    from functions.duq_gscale import *
    print("==> duq with grad_scale is selected..")
elif args.quant_op == "duq":
    from functions.duq import *
    print("==> differentiable and unified quantization method is selected..")
elif args.quant_op == "hwgq":
    from functions.hwgq import *
    print("==> Non-learnineg based HWGQ quantizer is selected..")
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
elif args.quant_op == 'duq_direct_b_qmax':
    from functions.duq_direct_b_qmax import *
else:
    raise NotImplementedError


# calculate bitops (theta-weighted)
def get_bitops(model, return_type='value'):
    """
    idx : If idx is -1, this func returns whole model's complexity.
          Else, it returns idx-th layer's complexity.
    """
    if args.search_type == 'interpolation':
        pass
    elif args.search_type == 'direct':
        a_bit_list = [8]
        w_bit_list = []
        computation_list = []

        for module in model.modules():
            if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
                a_bit_list.append(module.bits)
            elif isinstance(module, (Q_Conv2d, Q_Linear)):
                w_bit_list.append(module.bits)
                computation_list.append(module.computation)
        
        
    elif args.search_type == 'gumbel':
        a_bit_list = [8]
        w_bit_list = []
        computation_list = []
        
        for module in model.modules():
            if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)):
                if isinstance(module.bits, int) :
                    a_bit_list.append(module.bits)
                else:
                    softmask = F.softmax(module.theta/tau, dim=0)
                    a_bit_list.append((softmask * module.bits).sum())

            elif isinstance(module, (Q_Conv2d, Q_Linear)):
                if isinstance(module.bits, int) :
                    w_bit_list.append(module.bits)
                else:
                    softmask = F.softmax(module.theta/tau, dim=0)
                    w_bit_list.append((softmask * module.bits).sum())
                    
                computation_list.append(module.computation)
   

    if return_type == 'list':
        cost = []
        for i in range(len(a_bit_list)):
            cost.append(a_bit_list[i] * w_bit_list[i] * computation_list[i])
        
    else:
        cost = Tensor([0]).to(w_bit_list[0].device)
        for i in range(len(a_bit_list)):
            try:
                cost += a_bit_list[i] * w_bit_list[i] * computation_list[i]
            except Exception as e:
                print(computation_list[i].device())
                print(w_bit_list[i].device())
                print(a_bit_list[i].device())
                exit()
    return cost


# calculate bitops for full precision
def get_bitops_fullp():
    model_ = mobilenet_v2(QuantOps)
    model_ = model_.to(device)
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).cuda()
    else:
        input = torch.randn([1,3,224,224]).cuda()
    model_.train()
    QuantOps.initialize(model_, train_loader, 32, weight=True)
    QuantOps.initialize(model_, train_loader, 32, act=True)
    model_.eval()    
    return get_bitops(model_, 'list')


print("==> Calculate bitops..")
if args.fasttest:
    bitops_total = 298686218240 # 307992854528
    bitops_first = 2774532096 # 11098128384
    bitops_last = 327680000
else:
    bitops_list = get_bitops_fullp()
    bitops_total = sum(bitops_list)
    bitops_first = bitops_list[0]
    bitops_last = bitops_list[-1]

bitops_target = ((bitops_total - bitops_first - bitops_last) * (args.w_target_bit/32.) * (args.a_target_bit/32.) +\
              (bitops_first * (args.w_target_bit/32.)) +\
              (bitops_last * (args.a_target_bit/32.)))

logging.info(f'bitops_total : {int(bitops_total):d}')
logging.info(f'bitops_target: {int(bitops_target):d}')
#logging.info(f'bitops_wrong : {int(bitops_total * (args.w_target_bit/32.) * (args.a_target_bit/32.)):d}')
if type(bitops_target) != float:
    bitops_target = float(bitops_target)


# Make model
if args.model == "mobilenetv2":
    model = mobilenet_v2(QuantOps)
    if not os.path.isfile("./checkpoint/mobilenet_v2-b0353104.pth"):
        os.system("wget -P ./checkpoint https://download.pytorch.org/models/mobilenet_v2-b0353104.pth")
    model.load_state_dict(torch.load("./checkpoint/mobilenet_v2-b0353104.pth"), False)
    print("pretrained weight is loaded.")
else:
    raise NotImplementedError
model = model.to(device)


# optimizer -> for further coding (from PROFIT)
def get_optimizer(params, train_weight, train_quant, train_bnbias, train_w_theta, train_a_theta):
    #global lr_quant
    (weight, quant, bnbias, theta_w, theta_a, skip) = params
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD([
            {'params': weight, 'weight_decay': args.decay, 'lr': args.lr  if train_weight else 0},
            {'params': quant, 'weight_decay': 0., 'lr': args.lr_quant if train_quant else 0},
            {'params': bnbias, 'weight_decay': 0., 'lr': args.lr_bn if train_bnbias else 0},
            {'params': theta_w, 'weight_decay': 0., 'lr': args.lr_w_theta if train_w_theta else 0},
            {'params': theta_a, 'weight_decay': 0., 'lr': args.lr_a_theta if train_a_theta else 0},
            {'params': skip, 'weight_decay': 0, 'lr': 0},
        ], momentum=args.momentum, nesterov=True)
    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam([
            {'params': weight, 'weight_decay': args.decay, 'lr': args.lr  if train_weight else 0},
            {'params': quant, 'weight_decay': 0., 'lr': args.lr_quant if train_quant else 0},
            {'params': bnbias, 'weight_decay': 0., 'lr': args.lr_bn if train_bnbias else 0},
            {'params': theta_w, 'weight_decay': 0., 'lr': args.lr_w_theta if train_w_theta else 0},
            {'params': theta_a, 'weight_decay': 0., 'lr': args.lr_a_theta if train_a_theta else 0},
            {'params': skip, 'weight_decay': 0, 'lr': 0},
        ],)
    else:
        raise ValueError
    return optimizer


def categorize_param(model):
    weight = []
    quant = []
    bnbias = []
    theta_w = []
    theta_a = []
    skip = []
    for name, module in model.named_modules():
        if args.search_type == 'gumbel':
            if isinstance(module, (QuantOps.Conv2d, QuantOps.Linear)):
                theta_w.append(module.theta)
            elif isinstance(module, (QuantOps.ReLU, QuantOps.ReLU6, QuantOps.Sym)):
                theta_a.append(module.theta)
        
    for name, param in model.named_parameters():
        if name.endswith(".a") or name.endswith(".b") \
            or name.endswith(".c") or name.endswith(".d") \
            or name.endswith(".nlvs") or name.endswith(".bits"):
            quant.append(param)
        elif len(param.shape) == 1 and ((name.endswith('weight') or name.endswith(".bias"))):
            bnbias.append(param)
        elif name.endswith(".theta"):
            pass
        else:
            weight.append(param)
    return (weight, quant, bnbias, theta_w, theta_a, skip,)



# Bitwidth Initilization 
with torch.no_grad():
    # Retraining based on searched bitwidths
    if args.retrain_path:
        checkpoint = torch.load(args.retrain_path)
        if "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]
        
        for key in checkpoint:
            #print(key)
            if 'conv.0.0.bits' in key:
                num_w_bits = len(checkpoint[key])
                break
        for key in checkpoint:
            if 'conv.-0.bits' in key:
                num_a_bits = len(checkpoint[key])
                break
        dummy_w_bits = [i for i in range(3, 3+num_w_bits)]
        dummy_a_bits = [i for i in range(3, 3+num_a_bits)]

        if args.retrain_type == 1:
            logging.info('[Retraining type 1] sample weight and quantizer parameter from bitsearch result')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            
            print('==> load searched result..')
            model.load_state_dict(checkpoint)
            print('==> sample search result..')
            sample_search_result(model)
        
        elif args.retrain_type == 2:
            logging.info('[Retraining type 2] load weight from pretrained model, and just change bitwidths')
            print('=> Initialize for main model..')
            QuantOps.initialize(model, train_loader, dummy_w_bits[0], weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits[0], act=True)
            
            model2 = copy.deepcopy(model) 
            print('=> Initialize for auxilary model..')
            QuantOps.initialize(model2, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model2, train_loader, dummy_a_bits, act=True)
            model2.load_state_dict(checkpoint)
            
            print('==> Sample search result..')
            sample_search_result(model2)
            print('==> Transfer searched result to main model..')
            transfer_bitwidth(model2, model)
            del(model2)

        elif args.retrain_type == 3:
            logging.info('[Retraining type 3] sample weight from bitsearch result, and reinitialize a and c')
            print('=> Initialize for main model..')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            model.load_state_dict(checkpoint)

            print('==> Sample search result..')
            sample_search_result(model)
            model = model.to(device)
            QuantOps.initialize_quantizer(model, train_loader)

        elif args.retrain_type == 4:
            logging.info('[Retraining type 4] initialize weight using bitsearch result, training fixed bitwidth')
            QuantOps.initialize(model, train_loader, dummy_w_bits, weight=True)
            QuantOps.initialize(model, train_loader, dummy_a_bits, act=True)
            model.load_state_dict(checkpoint, strict=False)
            QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
            QuantOps.initialize(model, train_loader, args.a_bit, act=True)


        # ---- Variable interval (end) ---------------------------
        _, _, str_sel, _ = extract_bitwidth(model, weight_or_act="weight")
        logging.info(str_sel)
        _, _, str_sel, _ = extract_bitwidth(model, weight_or_act="act")
        logging.info(str_sel)

        model = model.to(device)
        logging.info(f"## sampled model bitops: {int(get_bitops(model).item())}")

    
    # Bitwidth searching or uniform QAT
    else:
        if args.search_type == 'direct':
            print('\n*********** direct ***********\n')
            print('==> weight bitwidth is set up..')
            QuantOps.initialize(model, train_loader, args.w_bit[0]+0.5, weight=True)
            print('==> activation bitwidth is set up..')
            QuantOps.initialize(model, train_loader, args.w_bit[0]+0.5, act=True)

        elif args.search_type == 'gumbel':
            print('==> weight bitwidth is set up..')
            QuantOps.initialize(model, train_loader, args.w_bit, weight=True)
            print('==> activation bitwidth is set up..')
            QuantOps.initialize(model, train_loader, args.a_bit, act=True)

        



model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f'==> DataParallel: device count = {torch.cuda.device_count()}')
    model = torch.nn.DataParallel(model) #, device_ids=range(torch.cuda.device_count()))



# optimizer & scheduler

params = categorize_param(model)
optimizer = get_optimizer(params, True, True, True, True, True)
current_lr = -1

scheduler = CosineWithWarmup(optimizer, 
        warmup_len=args.warmup, warmup_start_multiplier=0.1,
        max_epochs=args.epochs, eta_min=1e-3)

criterion = nn.CrossEntropyLoss()

w_module_nums = [4, 32, 36, 39, 123, 127, 130] if len(args.w_bit) > 1 else []
a_module_nums = [6, 30, 34, 38, 121, 125, 129] if len(args.a_bit) > 1 else []
temp_func = get_exp_cyclic_annealing_tau(args.cycle_epoch * len(train_loader),
                                         args.temp_step,
                                         np.round(len(train_loader) / args.n_gumbel))

_, _, _, str_prob = extract_bitwidth(model, weight_or_act=0)
print(str_prob)
_, _, _, str_prob = extract_bitwidth(model, weight_or_act=1)
print(str_prob)

scaler = amp.GradScaler()
wandb.init(project=PROJECT_NAME, dir=args.log_dir)
wandb.config.update(args)



# Training
def train(epoch, phase=None):
    global tau
    logging.info(f'[{phase}] train:')
    for i in range(len(optimizer.param_groups)):
        logging.info(f'[epoch {epoch}] optimizer, lr{i} = {optimizer.param_groups[i]["lr"]:.6f}')
    model.train()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        ###### DEBUG LINE START ######
        #if batch_idx == 2:
        #    break
        ###### DEBUG LINE END ######
        if (epoch > args.gumbel_epochs) and args.tau_end:
            pass
        else:
            tau_p = tau
            tau = temp_func((epoch - 1) * len(train_loader) + batch_idx)
            if tau_p != tau:
                set_tau(QuantOps, model, tau)
        if args.amp:
            with amp.autocast():
                inputs, targets = inputs.to(device), targets.to(device)
                data_time = time.time()
                outputs = model(inputs)
                bitops = get_bitops(model)
                loss_acc = criterion(outputs, targets)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
            data_time = time.time()
            outputs = model(inputs)
            bitops = get_bitops(model)
            loss_acc = criterion(outputs, targets)
        eval_acc_loss.update(loss_acc.item(), inputs.size(0))
        
        if args.lb_mode:
            loss_bitops = F.relu((bitops - bitops_target)/bitops_target ).reshape(torch.Size([]))
            loss = loss_acc + loss_bitops * alpha
            eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))
        else:
            loss = loss_acc

        acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        model_time = time.time()
        if (batch_idx) % args.log_interval == 0:
            tau = temp_func(batch_idx)
            logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                    'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ' +  \
                    'Data Time: %.3f s | Model Time: %.3f s',
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                eval_acc_loss.avg, eval_bitops_loss.avg if optimizer.param_groups[3]['lr'] !=0 else 0, 
                top1.avg, top5.avg,
                data_time - end, model_time - data_time)
            
            wandb.log({
                'train_acc1': acc1.item(),
                'train_acc5': acc5.item(),
                'train_loss_acc': loss_acc.item(), 
                'train_loss_bitops': loss_bitops.item() if args.lb_mode else 0,
                'tau': tau,
                }
            )
            ###  log theta of several selected modules
            i = 1
            fw_list = []
            fa_list = []
            for num in w_module_nums:
                fw_list.append(open(os.path.join(args.save, f'w_{num}_theta.txt'), mode='a'))
            for num in a_module_nums:
                fa_list.append(open(os.path.join(args.save, f'a_{num}_theta.txt'), mode='a'))
            for m in model.modules():
                if i in w_module_nums:
                    prob = F.softmax(m.theta / tau, dim=0)
                    prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
                    prob = ", ".join(prob) + "\n"
                    fw_list[w_module_nums.index(i)].write(prob)

                elif i in a_module_nums:
                    prob = F.softmax(m.theta / tau, dim=0)
                    prob = [f'{i:.5f}' for i in prob.cpu().tolist()]
                    prob = ", ".join(prob) + "\n"
                    fa_list[a_module_nums.index(i)].write(prob)

                i += 1
            for f_ in fw_list + fa_list:
                f_.close()
        end = time.time()

    if args.lb_mode:
        _, _, str_select, str_prob = extract_bitwidth(model, weight_or_act="weight", tau=tau)
        logging.info(f'Epoch {epoch}, weight bitwidth selection \n' + \
                      str_select + '\n'+ str_prob)
        
        _, _, str_select, str_prob = extract_bitwidth(model, weight_or_act="act", tau=tau)
        logging.info(f'Epoch {epoch}, activation bitwidth selection probability: \n' + \
                      str_select + '\n'+ str_prob)
        
    t1 = time.time()
    logging.info(f'epoch time: {t1-t0:.3f} s')
    


def eval(epoch):
    print('eval:')
    global best_acc
    model.eval()
    eval_acc_loss = AverageMeter()
    eval_bitops_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            ###### DEBUG LINE START ######
            #if batch_idx == 2:
            #    break
            ###### DEBUG LINE END ######
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss_acc = criterion(outputs, targets)
            bitops = get_bitops(model)
            eval_acc_loss.update(loss_acc.item(), inputs.size(0))

            if args.lb_mode:
                loss_bitops = F.relu((bitops - bitops_target)/bitops_target).reshape(torch.Size([]))
                loss = loss_acc + loss_bitops * alpha
                eval_bitops_loss.update(loss_bitops.item(), inputs.size(0))
                if (batch_idx) % (args.log_interval*5) == 0:
                    logging.info(f'bitops_target: {bitops_target}')
                    logging.info(f'evalaution time bitops: {bitops}')

            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            if (batch_idx) % args.log_interval == 0:
                logging.info('Train Epoch: %4d Process: %5d/%5d  ' + \
                        'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%% | ',
                    epoch, batch_idx * len(inputs), len(val_loader.dataset),
                    eval_acc_loss.avg, eval_bitops_loss.avg, top1.avg.item(), top5.avg.item())
            wandb.log({
                'eval_acc1': acc1.item(),
                'eval_acc5': acc5.item(),
                'eval_loss_acc': loss_acc.item(), 
                'eval_loss_bitops': loss_bitops.item() if args.lb_mode else 0,
                }
            )
        logging.info('L_acc: %.4f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%%' \
                    % (eval_acc_loss.avg, eval_bitops_loss.avg*1e-9, top1.avg, top5.avg))
        
        # Save checkpoint.        
        is_best = False
        if top1.avg > best_acc:
            is_best = True
            best_acc = top1.avg
        
        create_checkpoint(model, None, optimizer, is_best, None, 
                          top1.avg, best_acc, epoch, args.save, 1, args.exp)
    

if __name__ == '__main__':
    if args.search_type == 'direct':
        if args.eval:
            eval(0)
        else: # Search and/or finetuning
            args.lb_mode = True
            last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler,
                                            args.save, args.exp)
            for epoch in range(last_epoch+1, args.discretize_epoch+1):
                logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
                train(epoch, phase='Search')
                eval(epoch)
                scheduler.step()

            print("\n\n\ndiscretization()\n\n\n")
            sample_search_result(model)

            args.lb_mode = False
            for epoch in range(args.discretize_epoch+1, end_epoch):
                logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
                train(epoch, phase='Retrain')
                eval(epoch)
                scheduler.step()


    elif args.search_type == 'interpolation':
        pass


    elif args.search_type == 'gumbel':
        if args.eval:
            eval(0)
        elif args.retrain_path:
            last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler, 
                                            args.save, args.exp)
            for epoch in range(last_epoch+1, end_epoch+1):
                logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
                train(epoch, phase='Retrain')
                eval(epoch)
                scheduler.step()
        else:
            last_epoch, best_acc = resume_checkpoint(model, None, optimizer, scheduler, 
                                            args.save, args.exp)
            
            for epoch in range(last_epoch+1, end_epoch+1):
                if args.gumbel_epochs+1 == epoch:
                    remove_gumbel(QuantOps, model)

                alpha, string = get_alpha(args.alpha_init, args.alpha_target, args.epochs, epoch, args.alpha_type)
                wandb.log({'alpha': alpha})
                logging.info(string)

                if (args.gumbel_epochs < epoch) and args.tau_end:
                    tau, string = get_tau(args.tau_init, args.tau_target, args.epochs-args.gumbel_epochs, epoch-args.gumbel_epochs, 'linear')
                    logging.info(string)
                    set_tau(QuantOps, model, tau)

                logging.info('Epoch: %d/%d Best_Acc: %.3f' %(epoch, end_epoch, best_acc))
                train(epoch, phase='Search')
                eval(epoch)
                scheduler.step()

    logging.info('Best accuracy : {:.3f} %'.format(best_acc))
