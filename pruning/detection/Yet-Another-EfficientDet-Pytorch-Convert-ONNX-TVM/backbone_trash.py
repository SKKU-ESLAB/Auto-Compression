# Author: Zylo117

import math

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet.utils import Anchors
from torchvision.ops import nms
#from utils.utils import preprocess, invert_affine, postprocess

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    
    out = []
    for i in range(x.shape[0]):

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)
        #
        scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
        boxes_ = transformed_anchors_per[anchors_nms_idx, :]

        out.append({
            'rois': boxes_,
            'class_ids': classes_,
            'scores': scores_,
        })


    return out
class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False, onnx_export=onnx_export,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef], onnx_export=onnx_export)

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)

        # if you just want to convert to onnx, you can cancel the two lines of comments
        # or, if you want convert to tvm, just return regression and classification
        
        anchors = self.anchors(inputs, inputs.dtype)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        #features, regression, classification, anchors
        nms_threshold=0.4
        threshold = 0.4
        preds = postprocess(inputs,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        return preds




        
        #return regression, classification

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
