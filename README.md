# DNAS-Compression

Model compression techniques with differentiable neural architecture search.

Currently, pruning and quantization are supported.

* Pruning : Channel-level / Group-level
  * Sparsity and group size can be set
* Quantization : Uniform quantization
  * Bitwidth can be set 



## References

This project is implemented based on FBNet reproduced version.

* FBNet [https://github.com/AnnaAraslanova/FBNet.git][FBNet]



## Usage

* Requirement:
  * pytorch 1.6.0
  * tensorboardX 1.8
  * scipy 1.4.0
  * numpy 1.19.4

1. Choose what type of compression would you run

   * Pruning ( channel / group )
   * Quantization
   * Both pruning and quantization

2. Edit hyperparmeters in supernet_functions/config_for_supernet.py

   * Usual hyperparameters 	
     * batch size
     * learning rate
     * epochs
   * Special hyperparameters (pay attention to it!)
     * alpha, beta for adjustment of flops loss
     * w_share_in_train
     * thetas_lr
     * train_thetas_from_the_epoch

3. Run supernet_main_file.py

   Quick start command: 

   python3 supernet_main_file.py --train_or_sample train
