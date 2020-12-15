import argparse

def parser():
    parser = argparse.ArgumentParser(description='imagenet')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset (imagenet [default])')
    parser.add_argument('--model',  default='ResNet-18', help='Which model to use')
    parser.add_argument('--data_root', default='data', help='the directory to save the dataset')

    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--save_folder', default='natural_train', help='the affix for the save folder')


    # Training realted
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', '-e', type=int, default=160, help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="SGD weight decay(defalt: 1e-4)")
    parser.add_argument('--num_worker', '-nw', type=int, default=16, help='dataloader num_worker')
    parser.add_argument('--pretrained', '-pt', action='store_true' , help='pretrain models/ True or False')
    parser.add_argument('--pretrain_path', default=None, type=str, help='custum pretrain weight /')
    parser.add_argument('--conv1_not_train', '-cnt', action='store_true' , help='first conv1 trainable/ True or False')
    
    # KD
    parser.add_argument('--KD', '-kd', action='store_true' , help='knowledge distillation / True for False')
    parser.add_argument('--KD_ratio','-kdr', type=float, default=1.0, help="kd ratio")

    ## scheduler
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler(multistep, plateau)')
    parser.add_argument('--multi_step_epoch', default='[30, 60]', type=str, help='multi-step decay epoch')
    parser.add_argument('--multi_step_gamma', default=0.1, type=float, help='multi-step decay gamma. e.g)gamma=0.1, lr:0.1 --> 0.01')

    parser.add_argument('--n_eval_step', type=int, default=100, help='number of iteration per one evaluation')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')

    # pruning related
    #parser.add_argument('--group-shape', type=int, nargs='+', default=[1, 4], help='group shape')
    #parser.add_argument('--grouped-rule', type=str, default='l1', help='grouped rule (l1 or l2)')
    parser.add_argument('--prune_method',  choices=['dst', 'global', 'uniform'], default = None, help='dst / global threshold / uniform sparsity(layer-wise uniform sparsity)')
    parser.add_argument('--prune_type', default = None, help='filter-level or group-level(4-level) pruning')
    parser.add_argument('--sparsity', '-s', type=float, default=0, help='sparsity(zero parameters/ all paramters')
    parser.add_argument('--alpha', type=float, default=1e-6, help="penalty coefficient/ In the dst method, this controls the sparsity")
    parser.add_argument('--block_size', type=int, default=4, help='block size') 

    #  
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
