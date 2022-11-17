import logging

from .resnet import *
from .resnet_cifar import *
from .mobilenet import *
from .mobilenet_cifar10 import *
def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pre_trained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pre_trained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pre_trained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pre_trained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pre_trained)

    if args.arch == 'resnet20':
        model = resnet20(pretrained=args.pre_trained)
    elif args.arch == 'resnet32':
        model = resnet32(pretrained=args.pre_trained)
    elif args.arch == 'resnet44':
        model = resnet44(pretrained=args.pre_trained)
    elif args.arch == 'resnet56':
        model = resnet56(pretrained=args.pre_trained)
    elif args.arch == 'resnet110':
        model = resnet152(pretrained=args.pre_trained)
    elif args.arch == 'resnet1202':
        model = resnet1202(pretrained=args.pre_trained)

    if args.arch == "MobileNetv2" and args.dataloader.dataset == "cifar10":
        print("hello")
        model = mobilenetv2_cifar10(pretrained = args.pre_trained)
    elif args.arch == 'MobileNetv2' or args.arch == 'mobilenetv2':
        model = mobilenetv2_100(pretrained = args.pre_trained)
    elif args.arch == 'mobilenetv2_0.1':
        model = mobilenetv2_01(pretrained=args.pre_trained)
    elif args.arch == 'mobilenetv2_0.25':
        model = mobilenetv2_25(pretrained=args.pre_trained)
    elif args.arch == 'mobilenetv2_0.35':
        model = mobilenetv2_35(pretrained=args.pre_trained)
    elif args.arch == 'mobilenetv2_0.5':
        model = mobilenetv2_50(pretrained=args.pre_trained)
    elif args.arch == 'mobilenetv2_0.75':
        model = mobilenetv2_75(pretrained=args.pre_trained)
    elif args.arch == 'mobilenetv2_1.0':
        model = mobilenetv2_100(pretrained=args.pre_trained)
    

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model
