import torch
import argparse
import ml_collections
import os

from tt_mixer import TTMixer

def get_mixer_b16_tt_config(args):
    """Returns TTMixer-B/16 configuration"""
    config = ml_collections.ConfigDict()
    config = ml_collections.ConfigDict()
    config.name = 'Mixer-B_16'
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_dim = 768
    config.hidden_shape = args.hidden_tt_shape
    config.num_blocks = 12
    config.tokens_mlp_dim = 384
    config.channels_mlp_dim = 3072
    config.channels_mlp_shape = args.channels_tt_shape
    config.tt_ranks = args.tt_ranks
    return config

def set_configs(args):
    
    args.save_path = "saved_models/B_16_cifar_10.pt"    
    args.img_size = 224
    args.num_classes = 10
    args.tt_ranks = [int(i) for i in args.tt_ranks.split(',')]
    args.hidden_tt_shape = [int(i) for i in args.hidden_tt_shape.split(',')]
    args.channels_tt_shape = [int(i) for i in args.channels_tt_shape.split(',')]
    args.target_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    return args

def main():
    
    parser = argparse.ArgumentParser()
    # TT-format Configuration

    # ranks config [64, 64], [32, 32], [16, 16], [8, 8]
    parser.add_argument("--tt-ranks", default="32, 32",
                        type=str,
                        help="Ranks for TT-Format")
    # 768 factorize (e.g. 768 = 8 x 8 x 12)
    parser.add_argument("--hidden-tt-shape", default="8, 8, 12",
                        type=str,
                        help="Factorized hidden dimension for TT-format")
    # 3072 factorize (e.g. 3072 = 12 x 12 x 16)
    parser.add_argument("--channels-tt-shape", default="12, 16, 16",
                        type=str,
                        help="Factorized channel dimension for TT-format")
    args = parser.parse_args()
    
    args = set_configs(args)
    
    config = get_mixer_b16_tt_config(args)
    model = TTMixer(config,
                    args.img_size,
                    num_classes=args.num_classes,
                    patch_size=16,
                    zero_head=False,
                    target_layer=args.target_layer)

    torch.save(model.state_dict(), args.save_path)

if __name__=="__main__":
    main()
