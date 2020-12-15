import os
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.nn.utils.prune as prune
from train_argument import parser, print_args
from ofa.model_zoo import ofa_specialized

from time import time
from utils import *
from trainer import *

# models
from models.resnet import * 
from geffnet import * # https://github.com/rwightman/gen-efficientnet-pytorch
#from efficientnet_pytorch import EfficientNet # https://github.com/lukemelas/EfficientNet-PyTorch


def main(args):
    save_folder = args.save_folder

    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(model_folder)

    setattr(args, 'model_folder', model_folder)

    logger = create_logger(model_folder, 'train', 'info')
    print_args(args, logger)
    
    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    # ResNet
    if "ResNet" in args.model :
        depth_ = args.model.split('-')[1]
        
        # when it is dst, resnet block is special
        if args.prune_method!='dst' : p_type=None

        res_dict = {'18': resnet18(pretrained=args.pretrained, progress=True, prune_type = p_type),
                    '34': resnet34(pretrained=args.pretrained, progress=True, prune_type = p_type),
                    '50': resnet50(pretrained=args.pretrained, progress=True, prune_type = p_type),
                    '101': resnet101(pretrained=args.pretrained, progress=True, prune_type = p_type)
                    }
            
        net = res_dict[depth_]

    #elif 'efficientnet' in args.model:
    #    net = EfficientNet.from_pretrained(args.model)
    elif args.model == 'efficientnet_b0':
        print('efficientnet-b0 load...')
        net = efficientnet_b0(pretrained=args.pretrained)

    # MobileNet
    elif args.model == "mobilenetv3-large-1.0":
        print('mobilenetv3-large-1.0')
        net = mobilenetv3_large_100(pretrained=args.pretrained)

    elif args.model == 'once-mobilenetv3-large-1.0':
        print('once-mobilenetv3-large-1.0')
        net, image_size = ofa_specialized('note8_lat@65ms_top1@76.1_finetune@25', pretrained=args.pretrained)

    elif args.model == 'mobilenetv2-120d':
        print('mobilenetv2-120d load...')
        net = mobilenetv2_120d(pretrained=args.pretrained)
        
    # conv1 trainable
    if args.conv1_not_train :
        print('conv1 weight not train')
        if args.model == "mobilenetv3-large-1.0":
            for param in net.conv_stem.parameters():
                param.requires_grad = False
        elif "ResNet" in args.model :
            for param in net.conv1.parameters():
                param.requires_grad = False

        else:
            assert(False, 'not ready')
    
    # custom pretrain path
    if args.pretrain_path:
        print('load custom pretrain weight...')
        net.load_state_dict(torch.load(args.pretrain_path))

    net2 = copy.deepcopy(net) # for save removed_models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(net)
    net.to(device)

    # KD
    if args.KD:
        print('knowledge distillation model load!')
        teacher_net, image_size = ofa_specialized('flops@482M_top1@79.6_finetune@75', pretrained=True) # 79.6%
        teacher_net = nn.DataParallel(teacher_net)
        teacher_net.to(device)

    # set trainer
    if args.KD:
        trainer = Trainer_KD(args, logger)
    else:
        trainer = Trainer(args, logger)

    # loss
    loss = nn.CrossEntropyLoss()

    # dataloader
    if args.model != 'once-mobilenetv3-large-1' : image_size = 224
    if args.dataset=='imagenet':
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('/data/imagenet/', split='train',download=False, transform=transforms.Compose([
                            transforms.RandomSizedCrop(image_size),
                            transforms.RandomHorizontalFlip(),#ImageNetPolicy(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
        #
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('/data/imagenet/', split='val',download=False, transform=transforms.Compose([
                            transforms.Resize(int(image_size/0.875)),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker ,pin_memory=True)

    # optimizer & scheduler
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= eval(args.multi_step_epoch), gamma=args.multi_step_gamma)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, verbose=True, factor=0.3, threshold=1e-4, min_lr=1e-6)

    
    
    
    # pruning
    if args.prune_method=='global':
        if args.prune_type=='group_filter':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size() # (out, ch, h, w)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1) # (out, inp)
                    #append_size = 4 - tmp_pruned.shape[1] % 4
                    #tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, args.block_size) # out, -1, 4
                    tmp_pruned = tmp_pruned.abs().mean(2, keepdim=True).expand(tmp_pruned.shape) # out, -1, 4
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.2
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    #append_size = 4 - tmp_pruned.shape[1] % 4
                    #tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, args.block_size) # out, -1, 4
                    tmp_pruned = tmp_pruned.abs().mean(2, keepdim=True).expand(tmp_pruned.shape) # out,-1, 4
                    tmp_pruned = tmp_pruned.ge(threshold)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1) # out, inp
                    #tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size) # out, ch, h, w

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)
        
        elif args.prune_type=='group_channel':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size() # (out, ch, h, w)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1) # (out, inp)
                    #append_size = 4 - tmp_pruned.shape[1] % 4
                    #tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(-1, args.block_size, tmp_pruned.shape[1]) # out, -1, 4
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape) # out, -1, 4
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.2
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    #append_size = 4 - tmp_pruned.shape[1] % 4
                    #tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(-1, args.block_size, tmp_pruned.shape[1]) # out, -1, 4
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape) # out,-1, 4
                    tmp_pruned = tmp_pruned.ge(threshold)
                    #tmp_pruned = tmp_pruned.view(original_size[0], -1) # out, inp
                    #tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size) # out, ch, h, w

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)       
        
        elif args.prune_type =='filter':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.5
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(1, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmp_pruned = tmp_pruned.ge(threshold)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size)

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)        
        
        elif args.prune_type =='channel':
            tmps = []
            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(0, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmps.append(tmp)

            tmps = torch.cat(tmps)
            num = tmps.shape[0]*(1 - args.sparsity)#sparsity 0.5
            top_k = torch.topk(tmps, int(num), sorted=True)
            threshold = top_k.values[-1]

            for n,conv in enumerate(net.modules()):
                if isinstance(conv, nn.Conv2d):
                    if conv.weight.shape[1]<=3:
                        continue
                    tmp_pruned = conv.weight.data.clone()
                    original_size = tmp_pruned.size()
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned.abs().mean(0, keepdim=True).expand(tmp_pruned.shape)
                    tmp = tmp_pruned.flatten()
                    tmp_pruned = tmp_pruned.ge(threshold)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0: conv.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.contiguous().view(original_size)

                    prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)   
        print(f'model pruned!!(sparsity : {args.sparsity : .2f}, prune_method : {args.prune_method}, prune_type : {args.prune_type}-level pruning')
    
    elif args.prune_method=='uniform':
        assert False, 'uniform code is not ready'

    elif args.prune_method =='dst':
        print(f'model pruned!!(prune_method : {args.prune_method}, prune_type : {args.prune_type}-level pruning')

    elif args.prune_method == None:
        print('Not pruned model training started!')
    

    
    # Training
    if args.KD:
        trainer.train(net, teacher_net,  loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)
    else:
        trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)

    # save removed models
    filename = os.path.join(args.model_folder, 'pruned_models.pth')
    temp = torch.load(filename) ; temp_dict = OrderedDict()
    for i in temp:
        if ('orig' in i):
            value = temp[i]*temp[i.split('_orig')[0]+'_mask']
            temp_dict[i.split('module.')[1].split('_orig')[0]]= value
        elif 'mask' not in i:
            temp_dict[i.split('module.')[1]] = temp[i]
    net2.load_state_dict(temp_dict)
    save_model(net2, os.path.join(args.model_folder, 'removed_models.pth'))
    print('saved removed models')

if __name__ == '__main__':
    args = parser()
    main(args)
    if args.model == "mobilenetv3-large-1.0":
        os.system(f"python onnx_export.py --checkpoint=./checkpoint/{args.model_folder}/removed_moldels.pth --model mobilenetv3_large_100 ./removed_models.onnx")
