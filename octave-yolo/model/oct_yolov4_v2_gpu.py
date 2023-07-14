import torch
import torch.nn as nn
import torch.utils.tensorboard
import numpy as np
from .backbone.CSPDarknet53 import _BuildCSPDarknet53
import utils.utils
from octconv import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        device = torch.device('cuda:1')
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma
        self.BCELoss = nn.BCELoss()

    def forward(self, inputs, targets):
        BCE_loss = self.BCELoss(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
            Conv(feature_channels[-1]//2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
        )

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x]+features, dim=1)

        return features

    def __initialize_weights(self):
        # print("**" * 10, "Initing head_conv weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = Conv(in_channels, out_channels, 1),

    def forward(self, x):
        return self.upsample(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.downsample(x)

class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0]//2, 1)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1]//2, 1)
        
        self.resample5_4 = Upsample(feature_channels[2]//2, feature_channels[1]//2)
        self.resample4_3 = Upsample(feature_channels[1]//2, feature_channels[0]//2)
        self.resample3_4 = Downsample(feature_channels[0]//2, feature_channels[1]//2)
        self.resample4_5 = Downsample(feature_channels[1]//2, feature_channels[2]//2)

        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2]*2, feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )
        self.__initialize_weights()

    def forward(self, features):
        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]

        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
        downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))

        upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
        upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))

        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        print("**" * 10, "Initing PANet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))

class OctPANet(nn.Module):
    def __init__(self, feature_channels, alpha=0.5):
        super(OctPANet, self).__init__()

        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0]//2, 1)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1]//2, 1)

        self.resample5_4 = Conv(feature_channels[2]//2, feature_channels[1]//2, 1)
        self.resample4_3 = Conv(feature_channels[1]//2, feature_channels[0]//2, 1)
        self.resample3_4 = Conv(feature_channels[0]//2, feature_channels[1]//2, 1)
        self.resample4_5 = Conv(feature_channels[1]//2, feature_channels[2]//2, 1)

        self.downstream_conv5 = nn.Sequential(
            Conv_BN_ACT0a(in_channels=feature_channels[2]*2, out_channels=feature_channels[2]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.0, alpha_out=0.25, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[2]//2, out_channels=feature_channels[2], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.25, alpha_out=0.25, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[2], out_channels=feature_channels[2], kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.25, alpha_out=0.5, padding=0)
        )
        self.downstream_conv4 = nn.Sequential(
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.5, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[1]//2, out_channels=feature_channels[1], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[1]//2, out_channels=feature_channels[1], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1], kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=0.5 ,padding=0),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv_BN_ACT(in_channels=feature_channels[0], out_channels=feature_channels[0]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.5, alpha_out=0.75, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[0]//2, out_channels=feature_channels[0], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.75, alpha_out=0.75, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[0], out_channels=feature_channels[0]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.75, alpha_out=0.75, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[0]//2, out_channels=feature_channels[0], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.75, alpha_out=0.75, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[0], out_channels=feature_channels[0], kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.75, alpha_out=0.5, padding=0),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.5, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[1]//2, out_channels=feature_channels[1], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[1]//2, out_channels=feature_channels[1], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[1], out_channels=feature_channels[1], kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=0.5, padding=0),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv_BN_ACT(in_channels=feature_channels[2], out_channels=feature_channels[2]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=0.5, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[2]//2, out_channels=feature_channels[2], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACT(in_channels=feature_channels[2], out_channels=feature_channels[2]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=0),
            Conv_BN_ACT(in_channels=feature_channels[2]//2, out_channels=feature_channels[2], kernel_size=3, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=alpha, padding=1),
            Conv_BN_ACTa0(in_channels=feature_channels[2], out_channels=feature_channels[2]//2, kernel_size=1, bias=False, activation_layer=nn.LeakyReLU(negative_slope=0.1), alpha_in=alpha, alpha_out=0.0, padding=0),
        )
        self.__initialize_weights()

    def forward(self, features):
        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]

        downstream_feature5_h, downstream_feature5_l = self.downstream_conv5(features[2])
        downstream_feature4_h, downstream_feature4_l = self.downstream_conv4((features[1], self.resample5_4(downstream_feature5_h)))
        downstream_feature3_h, downstream_feature3_l = self.downstream_conv3((features[0], self.resample4_3(downstream_feature4_h)))

        upstream_feature4_h, upstream_feature4_l = self.upstream_conv4((self.resample3_4(downstream_feature3_l), downstream_feature4_l))
        upstream_feature5, upstream_feature5_l = self.upstream_conv5((self.resample4_5(upstream_feature4_l), downstream_feature5_l))

        return [downstream_feature3_h, upstream_feature4_h, upstream_feature5]

    def __initialize_weights(self):
        print("**" * 10, "Initing PANet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))

class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):
        super(PredictNet, self).__init__()

        self.predict_conv = nn.ModuleList([
            nn.Sequential(
                Conv(feature_channels[i]//2, feature_channels[i], 3),
                nn.Conv2d(feature_channels[i], target_channels, 1)
            ) for i in range(len(feature_channels))
        ])
        self.__initialize_weights()

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]

        return predicts

    def __initialize_weights(self):
        # print("**" * 10, "Initing PredictNet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))

class YOLODetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(YOLODetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2)
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

class YOLOv4(nn.Module):
    def __init__(self, image_size: int, num_classes: int, alpha = 0.5):
        super(YOLOv4, self).__init__()
        anchors = {'scale1': [(12, 16), (19, 36), (40, 28)],
                   'scale2': [(36,75),(76,55),(72,146)],
                   'scale3': [(142,110),(192,243),(459,401)]}
        final_out_channel = 3 * (4 + 1 + num_classes)

        self.CSPdarknet53, feature_channels = _BuildCSPDarknet53(weight_path="./weights/yolov4.weights", resume=False)

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(feature_channels)

        # Path Aggregation Net
        self.panet = OctPANet(feature_channels,alpha)

        # predict
        self.predict_net = PredictNet(feature_channels, final_out_channel)
        
        self.conv_block3 = self.make_conv_block(1024, 512)
        self.conv_final3 = self.make_conv_final(512, final_out_channel)
        self.yolo_layer3 = YOLODetection(anchors['scale3'], image_size, num_classes)

        self.upsample2 = self.make_upsample(512, 256, scale_factor=2)
        self.conv_block2 = self.make_conv_block(768, 256)
        self.conv_final2 = self.make_conv_final(256, final_out_channel)
        self.yolo_layer2 = YOLODetection(anchors['scale2'], image_size, num_classes)

        self.upsample1 = self.make_upsample(256, 128, scale_factor=2)
        self.conv_block1 = self.make_conv_block(384, 128)
        self.conv_final1 = self.make_conv_final(128, final_out_channel)
        self.yolo_layer1 = YOLODetection(anchors['scale1'], image_size, num_classes)

        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]

    def forward(self, x, targets=None):
        loss = 0

        # CSPDarknet-53 forward
        with torch.no_grad():
            features =self.CSPdarknet53(x)

        #YOLOv4 layer forward
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        predicts = self.predict_net(features)

        yolo_output3, layer_loss = self.yolo_layer3(predicts[0], targets)
        loss += layer_loss

        yolo_output2, layer_loss = self.yolo_layer2(predicts[1], targets)
        loss += layer_loss

        yolo_output1, layer_loss = self.yolo_layer1(predicts[2], targets)
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
    model = YOLOv4(image_size=416, num_classes=80)
    model.load_darknet_weights('../weights/yolov3.weights')
    print(model)

    test = torch.rand([1, 3, 416, 416])
    y = model(test)

    writer = torch.utils.tensorboard.SummaryWriter('../logs')
    writer.add_graph(model, test)
    writer.close()
