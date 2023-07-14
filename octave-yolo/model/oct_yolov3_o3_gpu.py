import torch
import torch.nn as nn
import torch.utils.tensorboard
import numpy as np
import pdb

import utils.utils
from octconv2 import *
from octconv import *

class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda:1' if x.is_cuda else 'cpu')

        num_batches = x.size(0)
        grid_size = x.size(2)

        # 출력값 형태 변환
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Get outputs
        cx = torch.sigmoid(prediction[..., 0])  # Center x
        cy = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # Calculate offsets for each grid
        stride = self.image_size / grid_size
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
        pred_boxes[..., 0] = cx + grid_x
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        pred = (pred_boxes.view(num_batches, -1, 4) * stride,
                pred_conf.view(num_batches, -1, 1),
                pred_cls.view(num_batches, -1, self.num_classes))
        output = torch.cat(pred, -1)

        if targets is None:
            return output, 0

        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.utils.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=scaled_anchors,
            ignore_thres=self.ignore_thres,
            device=device
        )

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss_layer = loss_bbox + loss_conf + loss_cls

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_cls": loss_cls.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "cls_acc": cls_acc.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }

        return output, loss_layer


class OCTYOLOv3_O3_GPU(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super(OCTYOLOv3_O3_GPU, self).__init__()
        anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
                   'scale2': [(30, 61), (62, 45), (59, 119)],
                   'scale3': [(116, 90), (156, 198), (373, 326)]}
        final_out_channel = 3 * (4 + 1 + num_classes)

        self.darknet53 = self.make_darknet53()
        self.conv_block3 = self.make_octconv_block1(1024, 512)
        self.conv_final3 = self.make_conv_final(512, final_out_channel)
        self.yolo_layer3 = YOLODetection(anchors['scale3'], image_size, num_classes)

        self.upsample2 = self.make_upsample(512, 256, scale_factor=2)
        self.conv_block2 = self.make_octconv_block2(1024, 256)
        self.conv_final2 = self.make_conv_final(256, final_out_channel)
        self.yolo_layer2 = YOLODetection(anchors['scale2'], image_size, num_classes)

        self.upsample1 = self.make_upsample(256, 120, scale_factor=2)
        self.conv_block1 = self.make_octconv_block3(1024, 120)
        self.conv_final1 = self.make_conv_final(120, final_out_channel)
        self.yolo_layer1 = YOLODetection(anchors['scale1'], image_size, num_classes)

        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]

    def forward(self, x, targets=None):
        loss = 0
        residual_output = {}

        # Darknet-53 forward
        with torch.no_grad():
            for key, module in self.darknet53.items():
                module_type = key.split('_')[0]

                if module_type == 'conv':
                    x = module(x)
                elif module_type == 'residual':
                    out = module(x)
                    x += out
                    if key == 'residual_3_8' or key == 'residual_4_8' or key == 'residual_5_4':
                        residual_output[key] = x

        # Yolov3 layer forward
        
        conv_block3, _conv_block3 = self.conv_block3(residual_output['residual_5_4'])
        scale3 = self.conv_final3(conv_block3)
        yolo_output3, layer_loss = self.yolo_layer3(scale3, targets)
        loss += layer_loss
        
        conv_block2, _conv_block2 = self.conv_block2((residual_output['residual_4_8'],conv_block3))
        scale2 = self.conv_final2(conv_block2)
        yolo_output2, layer_loss = self.yolo_layer2(scale2, targets)
        loss += layer_loss
        
        conv_block1, _conv_block1, __conv_block1 = self.conv_block1((residual_output['residual_3_8'],conv_block2,conv_block3))
        scale1 = self.conv_final1(conv_block1)
        yolo_output1, layer_loss = self.yolo_layer1(scale1, targets)
        loss += layer_loss

        yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def make_darknet53(self):
        modules = nn.ModuleDict()

        modules['conv_1'] = self.make_conv(3, 32, kernel_size=3, requires_grad=False)
        modules['conv_2'] = self.make_conv(32, 64, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_1_1'] = self.make_residual_block(in_channels=64)
        modules['conv_3'] = self.make_conv(64, 128, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_2_1'] = self.make_residual_block(in_channels=128)
        modules['residual_2_2'] = self.make_residual_block(in_channels=128)
        modules['conv_4'] = self.make_conv(128, 256, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_3_1'] = self.make_residual_block(in_channels=256)
        modules['residual_3_2'] = self.make_residual_block(in_channels=256)
        modules['residual_3_3'] = self.make_residual_block(in_channels=256)
        modules['residual_3_4'] = self.make_residual_block(in_channels=256)
        modules['residual_3_5'] = self.make_residual_block(in_channels=256)
        modules['residual_3_6'] = self.make_residual_block(in_channels=256)
        modules['residual_3_7'] = self.make_residual_block(in_channels=256)
        modules['residual_3_8'] = self.make_residual_block(in_channels=256)
        modules['conv_5'] = self.make_conv(256, 512, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_4_1'] = self.make_residual_block(in_channels=512)
        modules['residual_4_2'] = self.make_residual_block(in_channels=512)
        modules['residual_4_3'] = self.make_residual_block(in_channels=512)
        modules['residual_4_4'] = self.make_residual_block(in_channels=512)
        modules['residual_4_5'] = self.make_residual_block(in_channels=512)
        modules['residual_4_6'] = self.make_residual_block(in_channels=512)
        modules['residual_4_7'] = self.make_residual_block(in_channels=512)
        modules['residual_4_8'] = self.make_residual_block(in_channels=512)
        modules['conv_6'] = self.make_conv(512, 1024, kernel_size=3, stride=2, requires_grad=False)
        modules['residual_5_1'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_2'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_3'] = self.make_residual_block(in_channels=1024)
        modules['residual_5_4'] = self.make_residual_block(in_channels=1024)
        return modules

    def make_conv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True):
        module1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        module2 = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        if not requires_grad:
            for param in module1.parameters():
                param.requires_grad_(False)
            for param in module2.parameters():
                param.requires_grad_(False)

        modules = nn.Sequential(module1, module2, nn.LeakyReLU(negative_slope=0.1))
        return modules
    
    def make_octconv(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True, alpha_in=0.5, alpha_out=0.5):
        module = Conv_BN_ACT(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, alpha_in=alpha_in, alpha_out=alpha_out,activation_layer=nn.LeakyReLU(negative_slope=0.1))
        if not requires_grad:
            for param in module.parameters():
                param.requires_grad_(False)

        return module
    
    def make_octconv2(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=1, requires_grad=True, alpha_in=0.25, alpha_out=0.25):
        module = Conv_BN_ACT2(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, alpha_in=alpha_in, alpha_out=alpha_out,activation_layer=nn.LeakyReLU(negative_slope=0.1))
        if not requires_grad:
            for param in module.parameters():
                param.requires_grad_(False)

        return module

    def make_conv_block(self, in_channels: int, out_channels: int):
        double_channels = out_channels * 2
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            self.make_conv(out_channels, double_channels, kernel_size=3),
            self.make_conv(double_channels, out_channels, kernel_size=1, padding=0),
            self.make_conv(out_channels, double_channels, kernel_size=3),
            self.make_conv(double_channels, out_channels, kernel_size=1, padding=0)
        )
        return modules
    
    def make_octconv_block1(self, in_channels: int, out_channels: int):
        double_channels = out_channels * 2
        modules = nn.Sequential(
            self.make_octconv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0, alpha_out=0.25),
            self.make_octconv(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0)
        )
        return modules
    
    def make_octconv_block2(self, in_channels: int, out_channels: int):
        double_channels = out_channels * 2
        modules = nn.Sequential(
            self.make_octconv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.5, alpha_out=0.25),
            self.make_octconv(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0)
        )
        return modules
    
    def make_octconv_block3(self, in_channels: int, out_channels: int):
        double_channels = out_channels * 2
        modules = nn.Sequential(
            self.make_octconv2(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv2(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv2(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv2(in_channels=out_channels, out_channels=double_channels, kernel_size=3, alpha_in=0.25, alpha_out=0.25),
            self.make_octconv2(in_channels=double_channels, out_channels=out_channels, kernel_size=1, padding=0, alpha_in=0.25, alpha_out=0)
        )
        return modules

    def make_conv_final(self, in_channels: int, out_channels: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, in_channels * 2, kernel_size=3),
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return modules

    def make_residual_block(self, in_channels: int):
        half_channels = in_channels // 2
        block = nn.Sequential(
            self.make_conv(in_channels, half_channels, kernel_size=1, padding=0, requires_grad=False),
            self.make_conv(half_channels, in_channels, kernel_size=3, requires_grad=False)
        )
        return block

    def make_upsample(self, in_channels: int, out_channels: int, scale_factor: int):
        modules = nn.Sequential(
            self.make_conv(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return modules

    # Load original weights file
    def load_darknet_weights(self, weights_path: str):
        # Open the weights file
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values (0~2: version, 3~4: seen)
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # Load Darknet-53 weights
        for key, module in self.darknet53.items():
            module_type = key.split('_')[0]

            if module_type == 'conv':
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            elif module_type == 'residual':
                for i in range(2):
                    ptr = self.load_bn_weights(module[i][1], weights, ptr)
                    ptr = self.load_conv_weights(module[i][0], weights, ptr)

        # Load YOLOv3 weights
        if weights_path.find('yolov3.weights') != -1:
            for module in self.conv_block3:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final3[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final3[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample2[0][0], weights, ptr)

            for module in self.conv_block2:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final2[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample1[0][0], weights, ptr)

            for module in self.conv_block1:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final1[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[1], weights, ptr)

    # Load BN bias, weights, running mean and running variance
    def load_bn_weights(self, bn_layer, weights, ptr: int):
        num_bn_biases = bn_layer.bias.numel()

        # Bias
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_biases)
        ptr += num_bn_biases
        # Weight
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_weights)
        ptr += num_bn_biases
        # Running Mean
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_running_mean)
        ptr += num_bn_biases
        # Running Var
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_running_var)
        ptr += num_bn_biases

        return ptr

    # Load convolution bias
    def load_conv_bias(self, conv_layer, weights, ptr: int):
        num_biases = conv_layer.bias.numel()

        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]).view_as(conv_layer.bias)
        conv_layer.bias.data.copy_(conv_biases)
        ptr += num_biases

        return ptr

    # Load convolution weights
    def load_conv_weights(self, conv_layer, weights, ptr: int):
        num_weights = conv_layer.weight.numel()

        conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
        conv_weights = conv_weights.view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_weights)
        ptr += num_weights

        return ptr


if __name__ == '__main__':
    model = OCTYOLOv3_O3_GPU(image_size=416, num_classes=80)
    model.load_darknet_weights('../weights/yolov3.weights')
    print(model)

    test = torch.rand([1, 3, 416, 416])
    y = model(test)

    writer = torch.utils.tensorboard.SummaryWriter('../logs')
    writer.add_graph(model, test)
    writer.close()
