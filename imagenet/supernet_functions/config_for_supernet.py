import numpy as np

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [0]
    },
    'lookup_table' : {
        'create_from_scratch' : False,
        'path_to_lookup_table' : './supernet_functions/lookup_table.txt',
        'path_to_lookup_table_high' : './supernet_functions/lookup_table_high.txt',
        'number_of_runs' : 50 # each operation run number_of_runs times and then we will take average
    },
    'logging' : {
        'path_to_log_file' : './supernet_functions/logs/logger/',
        'path_to_tensorboard_logs' : './supernet_functions/logs/tb'
    },
    'dataloading' : {
        'batch_size' : 40,
        'w_share_in_train' : 0.8,
        'path_to_save_data' : '/home/khs/data/sup_logs/imagenet'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.0005,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-5, #1e-4
        # Adam parameters for thetas
        'thetas_lr' : 0.01,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 0.2,
        'beta' : 0.6
    },
    'train_settings' : {
        'cnt_epochs' : 10, # 90
        'train_thetas_from_the_epoch' : 2,
        'print_freq' : 50,
        'path_to_save_model' : '/home/khs/data/sup_logs/imagenet/best_model.pth',
        'path_to_save_model_high' : '/home/khs/data/sup_logs/imagenet/best_model_high.pth',
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045)
    }
}
