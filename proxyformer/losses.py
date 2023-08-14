# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torch.nn as nn 
import math
class CustomDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, IQ, IK):
        out = IQ @ IK.transpose(-1, -2)
        ctx.save_for_backward(IQ, IK, out)
    @staticmethod
    def backward(ctx, grad_output):
        IQ, IK, out = ctx.saved_tensors
        grad_output = grad_output.type(IQ.dtype)
        
        IK_proj = out / (IQ ** 2).sum(dim = -1, keepdim = True)
        IK_ortho = IK.unsqueeze(-3) - IK_proj.unsqueeze(-1) * IQ.unsqueeze(-2)
        grad_IQ = (grad_output.unsqueeze(-2) @ IK_ortho).squeeze()
        
        IQ_proj = out.transpose(-1, -2) / (IK ** 2).sum(dim = -1, keepdim = True)
        IQ_ortho = IQ.unsqueeze(-3) - IQ_proj.unsqueeze(-1) * IK.unsqueeze(-2)
        grad_IK = (grad_output.transpose(-1, -2).unsqueeze(-2) @ IQ_ortho).squeeze()
        return grad_IQ, grad_IK

class NpairLoss(nn.Module):
    def __init__(self):
        super(NpairLoss, self).__init__()
        self.scale = 64 ** -0.25
        self.sim = CustomDot.apply

    def forward(self, I):
        I = I[0]
        B, H, N, C = I.shape
        I = I * self.scale
        
        Inorm = I.norm(dim = -1, keepdim = True)

        mask = torch.eye(N, dtype = I.dtype, device = I.device)
        dot = self.sim(I, I)

        with torch.no_grad():
            implicit_redundancy = Inorm @ Inorm.transpose(-1, -2)
            implicit_redundancy = implicit_redundancy * mask - implicit_redundancy * (1. - mask)
            implicit_redundancy = implicit_redundancy - (implicit_redundancy * mask).sum(dim = -1, keepdim = True)
        dot = dot - (dot * mask).sum(dim = -1, keepdim = True)
        loss = (dot.exp().sum(dim = -1).log() - implicit_redundancy.log()).mean() / N
        return loss

class ContrastiveLoss(nn.Module):
    """
        Contrastive Loss
    """
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = 64 ** -0.25

    def forward(self, I):
        IQ = I[0]
        IK = I[1]
        
        B, H, N, C = IQ.shape

        IQ = IQ * self.temperature
        normIQ = IQ.norm(dim = -1, keepdim = True)
        dirIQ = IQ / (normIQ + 1e-12)
        IQ = normIQ.detach() * dirIQ

        IK = IK * self.temperature
        normIK = IK.norm(dim = -1, keepdim = True)
        dirIK = IK / (normIK + 1e-12)
        IK = normIK.detach() * dirIK

        
        mask = torch.eye(IQ.shape[-2], dtype = IQ.dtype, device = IQ.device)
        dot = (IQ @ IK.transpose(-1, -2))
        sparse_labels = torch.arange(0, N, dtype = torch.long, device = IQ.device).expand(B * H, -1)
        loss = torch.nn.functional.cross_entropy(dot.reshape(B * H, N, -1), sparse_labels)  

        #target_logits = normIQ @ normIK.transpose(-1, -2)
        #target_logits = (target_logits * mask - target_logits * (1.-mask))
        #target_softmax = target_logits.softmax(dim = -1)

        #loss = torch.nn.functional.kl_div(dot.log_softmax(dim = -1), target_softmax.detach(), reduction = "none") * mask
        #loss = -(target_softmax.detach() * dot.log_softmax(dim = -1)) * mask
        #loss = loss.sum(dim = -1).mean()

        #entropy = -torch.sum(target_softmax * torch.log(target_softmax + 1e-9), dim=-1).mean()
        
        #loss = loss + entropy
        return loss


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
