import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

import model.oct_yolov4_v2
import utils.datasets
import utils.utils

import onnx
import onnxruntime as ort

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device):
    # 모델을 evaluation mode로 설정
    model.eval()
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    model_path = "/home/sangjun/octyolo/yolov3-pytorch/"+str(args.alpha)+".onnx"
    ort_session = ort.InferenceSession(model_path,providers=['CUDAExecutionProvider'])

    # 데이터셋, 데이터로더 설정
    dataset = utils.datasets.ListDataset(path, image_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=dataset.collate_fn)

    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for _, images, targets in tqdm.tqdm(dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = utils.utils.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
            outputs = ort_session.run(None,  ort_inputs)
            entire_time += time.time() - start_time


    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return inference_time, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                        help="path to pretrained weights file")
    parser.add_argument("--image_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha")
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정값을 가져오기
    data_config = utils.utils.parse_data_config(args.data_config)
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.utils.load_classes(data_config['names'])

    # 모델 준비하기
    model = model.oct_yolov4_v2.YOLOv4(args.image_size, num_classes, args.alpha).to(device)
    if args.pretrained_weights.endswith('.pth'):
        model.load_state_dict(torch.load(args.pretrained_weights, map_location=device),strict=False)
    else:
        model.load_darknet_weights(args.pretrained_weights)

    # 입력 데이터 샘플 정의 (예시)
    dummy_input = torch.randn(1, 3, 448, 448).to("cuda")
    # ONNX로 변환
    torch.onnx.export(model, dummy_input, str(args.alpha)+'.onnx')
    

    # 검증 데이터셋으로 모델을 평가
    inference_time, fps = evaluate(model,
                                                                        path=valid_path,
                                                                        iou_thres=args.iou_thres,
                                                                        conf_thres=args.conf_thres,
                                                                        nms_thres=args.nms_thres,
                                                                        image_size=args.image_size,
                                                                        batch_size=args.batch_size,
                                                                        num_workers=args.num_workers,
                                                                        device=device)

    # AP, mAP, inference_time 출력
    print('Inference_time (ms): {:.02f}'.format(inference_time))
    print('FPS: {:.02f}'.format(fps))


    