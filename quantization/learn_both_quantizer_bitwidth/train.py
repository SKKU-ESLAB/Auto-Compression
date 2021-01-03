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

from models import *
from utils import *
from functions import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch - Learning Quantization')
parser.add_argument('--model', default='resnet32', help='select model')
parser.add_argument('--dir', default='/data', help='data root')
parser.add_argument('--dataset', default='cifar100', help='select dataset')
parser.add_argument('--load', default=0, type=int, help='0: no load, 1: resume, 2: load model, 3: load weight')
parser.add_argument('--load_file', default='./checkpoint/trained.pth', help='select loading file')
parser.add_argument('--batchsize', default=128, type=int, help='set batch size')
parser.add_argument('--epoch', default=160, type=int, help='number of epochs tp train for')
parser.add_argument('--wd', default=1e-4, type=float, help='set weight decay value')
parser.add_argument('--seed', default=7, type=int, help='random seed')
parser.add_argument('--strict_false', action='store_true', help='load_state_dict option "strict" False')
parser.add_argument('--m', default=0.9, type=float, help='set momentum value')
parser.add_argument('--lr_ms', nargs='+', default=[20, 40, 60], type=int, help='set milestones')
parser.add_argument('--lr_g', default=0.1, type=float, help='set gamma for multistep lr scheduler')
parser.add_argument('--cosine', action='store_true', help='set the lr scheduler to cosine annealing')
parser.add_argument('--workers', default=4, type=int, help='set number of workers')
parser.add_argument('--savefile', default='ckpt.pth', help='save file name')
parser.add_argument('--log_interval', default=50, type=int, help='logging interval')
parser.add_argument('--exp', type=str)

parser.add_argument('--lr1', default=0.01, type=float, help='set learning rate value1')
parser.add_argument('--lr2', default=0.001, type=float, help='set learning rate value2')
parser.add_argument('--lr3', default=0.0001, type=float, help='set learning rate value3')
parser.add_argument('--lq_mode', '-lq', action='store_true', help='learning quantization mode')
parser.add_argument('--lb_mode', '-lb', action='store_true', help='learn bitwidth (dnas approach)')
parser.add_argument('--is_qt', '-q', action='store_true', help='quantization')
parser.add_argument('--gamma', '-g', action='store_true', help='trainable gamma factor')
parser.add_argument('--comp_ratio', default=0.005, type=float, help='set target compression ratio of FLOPs loss')
parser.add_argument('--scaling', default=1e-6, type=float, help='set FLOPs loss scaling factor')
parser.add_argument('--w_bit', default=[6], type=int, nargs='+', help='set weight bits')
parser.add_argument('--x_bit', default=[8], type=int, nargs='+', help='set activation bits')
parser.add_argument('-eval', action='store_true', help='evaluation mode')
parser.add_argument('-initskip', action='store_true', help='skip initialization (for loading cw, dw? maybe..')

parser.add_argument('-fwbw', action='store_true', help='use filter-wise bitwidth')
parser.add_argument('-fwlq', action='store_true', help='use filter-wise quantization interval learning')
args = parser.parse_args()
args.save = f'logs/{args.dataset}/{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
create_exp_dir(args.save)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)





#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

best_acc = 0
start_epoch = 1
end_epoch = args.epoch

if len(args.w_bit)==1:
    print("Fixed bitwidth for weight")
    args.w_bit = args.w_bit[0]
if len(args.x_bit)==1:
    print("Fixed bitwidth for activation")
    args.x_bit = args.x_bit[0]

# Data
print('==> Preparing Data..')
train_loader, val_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
torch.autograd.set_detect_anomaly(True)

print('==> Building Model..')
if args.fwbw and not args.lb_mode:
    index = np.load(f'./index_{args.w_bit}bit.npy', allow_pickle=True)
    print("Filter-wise bitwidth!")
elif args.fwbw and args.lb_mode:
    index = []
    print("Learning filter-wise bitwidth!") ### TODO
    raise NotImplementedError
elif args.lb_mode:
    index = []
    print("Learning layer-wise bitwidth")
else:
    index = []
    print("Fixed bitwidth!")

def get_bitops_total():
    model_ = model_builder(args.model, args.dataset)
    model_ = model_.to(device)
    if args.dataset in ["cifar100", "cifar10"]:
        input = torch.randn([1,3,32,32]).cuda()
    else:
        input = torch.randn([1,3,224,224]).cuda()
    model_.train()
    out, _ =  model_(input)
    bitops = 0
    for m in model_.modules():
        if isinstance(m, lq_conv2d_orig):
            bitops += m.bitops_count()
    return bitops

bitops_total = get_bitops_total()
print(f'bitops_total: {int(bitops_total[0]):d}')

model = model_builder(args.model, args.dataset, args.is_qt, args.lq_mode, index, args.fwlq)
model = model.to(device)

#for key in model.state_dict().keys():
#    print(key)
if args.initskip and device == torch.device('cuda'):
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
elif args.load == 1:
    # Load saved file.
    print(f'==> Resuming from saved file..: {args.load_file}')
    checkpoint = torch.load('%s' %args.load_file)
    if 'model' in  checkpoint.keys():
        checkpoint = checkpoint['model']
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    #for i in checkpoint.keys():
    #    #print(i)
    #    if i in model.state_dict().keys():
    #        print(i)

    if args.model=='mobilenetv2':
        model.load_state_dict(checkpoint, strict=False)
        best_acc=0
        start_epoch=0
    else:
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 2
        print('%.3f' %best_acc)

elif args.load == 2:
    print('==> Loading model')
    checkpoint = torch.load('%s' %args.load_file)
    #model.load_state_dict(checkpoint)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['model'], strict=not(args.strict_false))

elif args.load == 3:
    print('==> Loading pretrained weight')
    checkpoint = torch.load('./%s' %args.load_file)
    pretrained = model_builder(args.model, args.dataset, False, index)
    pretrained.load_state_dict(checkpoint['model'])

else :
    model = torch.nn.DataParallel(model)
    pass
if args.is_qt:
    for m in model.modules():
        if isinstance(m, lq_conv2d_orig):
            m.is_qt = args.is_qt
            m.tr_gamma = args.gamma
            m.set_bit_width(args.w_bit, args.x_bit, args.eval or args.initskip)
model = model.to(device)
#print(m)
#a = checkpoint['model'];
#b = a['module.layer3.0.conv1.weight']
#c = torch.flatten(b,1)

#np.savetxt('weight_.csv', c.cpu(), delimiter=',')


criterion = nn.CrossEntropyLoss()

lr1_param = []
lr2_param = []
lr3_param = []
lr1_param_name = []
lr2_param_name = []
lr3_param_name = []
for name, param in model.named_parameters():
    if 'gamma' in name:
        lr3_param.append(param)
        lr3_param_name.append(name)
    elif ('dw' in name) or ('cw' in name) or ('dx' in name) or ('cx' in name)  :
        lr2_param.append(param)
        lr2_param_name.append(name)
    else:
        lr1_param.append(param)
        lr1_param_name.append(name)

optimizer = optim.SGD([{'params': lr1_param, 'lr': args.lr1, 'weight_decay': args.wd},
                       {'params': lr2_param, 'lr': args.lr2},
                       {'params': lr3_param, 'lr': args.lr3}], momentum=args.m)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_ms, gamma=args.lr_g)
if args.cosine:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr1*(args.lr_g ** 3))

logging.info("args = %s", args)


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
        outputs, bitops= model(inputs)
        if len(bitops)>1:
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
        '''
        if args.lb_mode:
            progress_bar(batch_idx, len(train_loader), 'L_acc: %.3f | L_bitops: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%%'
                % (eval_acc_loss.avg, eval_bitops_loss.avg, top1.avg, top5.avg))
        else :
            progress_bar(batch_idx, len(train_loader), 'L_acc: %.3f | top1.avg: %.3f%% | top5.avg: %.3f%%'
                % (eval_acc_loss.avg, top1.avg, top5.avg))
        '''
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
                if isinstance(args.x_bit, list):
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
            outputs, bitops = model(inputs)
            loss = criterion(outputs, targets)

            # TODO: bitops loss
            acc1, acc5 = accuracy(outputs.data, targets.data, top_k=(1,5))
            eval_loss.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            #progress_bar(batch_idx, len(val_loader), 'Loss: %.4f | top1.avg: %.3f%% | top5.avg: %.3f%%'
            #% (eval_loss.avg , top1.avg, top5.avg))
        logging.info('Loss: %.4f | top1.avg: %.3f%% | top5.avg: %.3f%%' % (eval_loss.avg, top1.avg, top5.avg))
        # Save checkpoint.
        if top1.avg > best_acc:
            best_acc = top1.avg
            state = {
                'model': model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, f'{args.save}/{args.exp}_best_{args.savefile}')

        if (epoch % 10) == 0:
            state = {
                'model': model.state_dict(),
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
                            prob_w = F.softmax(m.theta_w).cpu().tolist()
                            sel=torch.argmax(prob_w)
                            str_to_log += f'{args.w_bit[sel]}'
                            #prob_w = [f'{i:.5f}' for i in prob_w]
                            #str_to_log += f'layer {i} theta_w: [{", ".join(prob_w)}]\n'
                        else:
                            break
                logging.info(str_to_log)


logging.info('Best accuracy : {:.3f} %'.format(best_acc))
#print_param(model)
