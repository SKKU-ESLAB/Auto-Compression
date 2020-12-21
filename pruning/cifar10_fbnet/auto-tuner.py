import torch
import torchvision.models as models
import argparse
import general_functions.dataloaders as dataloaders
from torchprofile import profile_macs as get_flops
from general_functions import AverageMeter, accuracy
from supernet_main_file import train_supernet


parser = argparse.ArgumentParser()
parser.add_argument('--target_modelnum', type=int, default=5)
parser.add_argument('--accuracy_gap_min', type=float, default=1)
parser.add_argument('--accuracy_gap_max', type=float, default=3)
parser.add_argument('--flops_gap_min', type=float, default=1)
parser.add_argument('--flops_gap_max', type=float, default=5)
parser.add_argument('--accuracy_lower_bound', type=float, default=10)
parser.add_argument('--flops_lower_bound', type=float, default=10)
parser.add_argument('--init_alpha', type=float, default=1)
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--dataset_path', type=str, default='/var/imagenet')
args = parser.parse_args()
args.train_or_sample = 'train'
args.architecture_name = args.arch
args.prune = 'channel'

def get_accuracy(model, test_dataloader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for step, (X, target) in enumerate(test_dataloader)
        outs = model(X)
        acc1 = accuracy(output, target, topk=(1))
        top1.update(acc1[0], X.size(0))
    return top1.avg

def stop_condtion(acc, flops, acc_lower, flops_lower):
    if len(acc) == 1:
        return False
    elif acc[-1] < acc_lower or flops < flops_lower:
        return True
    else:
        return False

def get_alpha(acc1, flops, alpha, args):
    if len(acc) == 1 :
        pass
    else:
        acc_gap = acc1[-2]-acc1[-1]
        flops_gap = (flops[-2]-flops[-1]) / flops[0]
        if acc_gap < args.accuracy_gap_min:
            alpha_gap = alpha[-1] * (accuracy_gap_min / acc_gap)
        elif acc_gap > args.accuracy_gap_max:
            alpha_gap = alpha[-1] * (accuracy_gap_max / acc_gap)
        elif flops_gap < args.flops_gap_min:
            alpha_gap = alpha[-1] * (flops_gap_min / flops_gap)
        elif flops_gap > args.flops_gap_max:
            alpha_gap = alpha[-1] * (flops_gap_max / flops_gap)
        else:
            alpha_gap = alpha[-1] * (accuracy_gap_max / accuracy_gap_min)
        alpha.append(alpha[-1] + alpha_gap)[]
    return alpha[-1]

if __name__ == "__main__":
    model = models.__dict__[args.arch](pretrained=True)
    if args.dataset == "imagenet":
        image_size = (224, 224, 3)
    else:
        image_size = (32, 32, 3)
    test_dataloader = dataloaders.get_test_loader(128, args.dataset_path)

    acc_baseline = get_accuracy(model, test_dataloader)
    flops_baseline = get_flops(model, image_size)
    acc_lower = acc_baseline - args.accuracy_lower_bound
    flops_lower = flops_baseline - args.flops_lower_bound

    alpha = [args.alpha]
    acc1 = [acc_baseline]
    flops = [flops_baseline]

    while not stop_condition(acc1, flops, acc_lower, flops_lower):
        alpha.append(get_alpha(acc1, flops, alpha))
        args.alpha = alpha[-1]
        a, f = train_supernet(args)
        acc1.append(a)
        flops.append(b)


    args.prune = 'group'
    alpha = [alpha[0]]
    acc1 = [acc_baseline]
    flops = [flops_baseline]

    while not stop_condition(acc1, flops, acc_lower, flops_lower):
        alpha.append(get_alpha(acc1, flops, alpha))
        args.alpha = alpha[-1]
        a, f = train_supernet(args)
        acc1.append(a)
        flops.append(b)

