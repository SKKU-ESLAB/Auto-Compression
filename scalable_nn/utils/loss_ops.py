import torch

from utils.config import FLAGS


## for soft-target
class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    ## target := [Batch, num_classes], i.e. label (softmax), prediction of supernet
    ## output_log_prob (olp) := [Batch, num_classes], i.e. prediction of subnet
    ## target_unsqz = [Batch, 1, num_classes]
    ## olp_unsqz = [Batch, num_classes, 1]
    ## CE_loss = -1 X [Batch, 1, num_classes] X [Batch, num_classes, 1] -> [Batch, 1, 1], i.e. sum(-label * log Predcition)
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


## for CE Loss 
class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    """ label smooth """
    ## target := [Batch]
    ## one_hot := [Batch, num_classes], i.e. zeros_like(*).scatter :sparse to dense operation
    ## output_log_prob (olp) := [Batch, num_classes], i.e. prediction of subnet
    ## target_unsqz := [Batch, 1, num_classes]
    ## olp_unsqz := [Batch, num_classes, 1]
    ## CE_loss = -1 X [Batch, 1, num_classes] X [Batch, num_classes, 1] -> [Batch, 1, 1], i.e. sum(-label * log Predcition)
    def forward(self, output, target):
        eps = FLAGS.label_smoothing
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss
