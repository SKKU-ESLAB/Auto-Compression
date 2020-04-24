import torch
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

class MixedOperation(nn.Module):
    
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        #self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters[idx])
         #                         for idx, op_name in enumerate(ops_names)])
        self.latency = [latency[op_name] for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
        latency_to_accumulate = latency_to_accumulate + latency
        return output, latency_to_accumulate

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table, cnt_classes=1000):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        # self.first identical to 'add_first' in the fbnet_building_blocks/fbnet_builder.py
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=2,
                                pad=3 // 2, no_bias=1, use_relu="relu", bn_type="bn")
        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])
        self.last_stages = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(lookup_table.layers_parameters[-1][1], 1504, kernel_size = 1)),
            #("avg_pool_k7", nn.AvgPool2d(kernel_size=7)),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1504, out_features=cnt_classes)),
        ]))
    
    def forward(self, x, temperature, latency_to_accumulate):       
        y = self.first(x)
        for mixed_op in self.stages_to_search:
            y, latency_to_accumulate = mixed_op(y, temperature, latency_to_accumulate)
        y = self.last_stages(y)
        return y, latency_to_accumulate
    
class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.weight_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outs, targets, latency, losses_ce, losses_lat, N):
        
        ce = self.weight_criterion(outs, targets)
        lat = torch.log(latency ** self.beta)
        
        losses_ce.update(ce.item(), N)
        losses_lat.update(lat.item(), N)
        
        loss = self.alpha * ce * lat
        return loss #.unsqueeze(0)

