# **The repository consists of 2 Neural Net Models:**

The repository consists of 2 Neural Net Models:

**(1)** FBNet Searched Architectures. All tools in the *architecture_functions* folder

**(2)** Stochastic SuperNet to search for new architectures. All tools in the *supernet_functions* folder

> They use different functions and architectures specification. Functions used by both Nets are in the folders: *general_functions* (utilities) and *fbnet_building_blocks* (modified code of facebookresearch team)

> If you want to use your own dataset, you should rewrite functions in *general_functions/dataloaders.py*

-----

#### (1) FBNet Searched Architectures

> CONFIG FILE : *architecture_functions/config_for_arch.py*

Available FBNet architectures: **fbnet_a**, **fbnet_b**, **fbnet_c**, **fbnet_96_035_1** (following the paper, for input size 96 and channel scaling 0.35), **fbnet_samsung_s8** (for Samsung Galaxy S8), **fbnet_iphonex** (for iPhoneX), **fbnet_cpu_sample1**, **fbnet_cpu_sample2** (FBNets searched with my CPU latency)

You can choose any architectures and train it (takes 1.5 hours with 1 GPU):

`python architecture_main_file.py --architecture_name fbnet_cpu_sample1`

All logs will be printed and also saved to *architecture_functions/logs/logger*, tensorboard logs will be in *architecture_functions/logs/tb/*, best model will be saved in *architecture_functions/logs/best_model.pth*

#### (2) Stochastic SuperNet to search for new architectures:

> CONFIG FILE : *supernet_functions/config_for_supernet.py*

You can recalculate latecy for your CPU, just change the config : *supernet_functions/config_for_supernet.py* -> *'lookup_table'* -> 'create_from_scratch' = True

Next, train it (takes 4 hours with 1 GPU):

`python supernet_main_file.py --train_or_sample train`

All logs will be printed and also saved to *supernet_functions/logs/logger*, tensorboard logs will be in *supernet_functions/logs/tb/*, best model will be saved in *supernet_functions/logs/best_model.pth*

> QuickNote: Thetas will converge well if you would want to train until the overfitting point. But I advise stop earlier because if thetas will truly converge, the softmax becomes the hardmax and you could sample only one architecture (btw - the best for your device)

After training Sample arhitecture from the resulted model (pick a unique name and replace `my_unique_name_for_architecture` in the following query):

`python supernet_main_file.py --train_or_sample sample --architecture_name my_unique_name_for_architecture --hardsampling_bool_value True`

The sampled architecture will be written into *fbnet_building_blocks/fbnet_modeldef.py*

Now, you can train your architecture `my_unique_name_for_architecture` from scratch with the instructions **(1)** above
