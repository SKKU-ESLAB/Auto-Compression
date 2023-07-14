import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm

import model.yolov3
import utils.datasets
import utils.utils


def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_workers, device):
    # 모델을 evaluation mode로 설정
    model.eval()

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
            outputs = model(images)
            entire_time += time.time() - start_time
            outputs = utils.utils.non_max_suppression(outputs, conf_thres, nms_thres)
        

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(utils.utils.get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = utils.utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / dataset.__len__()
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    return precision, recall, AP, f1, ap_class, inference_time, fps


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
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 설정값을 가져오기
    data_config = utils.utils.parse_data_config(args.data_config)
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.utils.load_classes(data_config['names'])

    # 모델 준비하기
    model = model.yolov3.YOLOv3(args.image_size, num_classes).to(device)
    if args.pretrained_weights.endswith('.pth'):
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        model.load_darknet_weights(args.pretrained_weights)

    # 검증 데이터셋으로 모델을 평가
    precision, recall, AP, f1, ap_class, inference_time, fps = evaluate(model,
                                                                        path=valid_path,
                                                                        iou_thres=args.iou_thres,
                                                                        conf_thres=args.conf_thres,
                                                                        nms_thres=args.nms_thres,
                                                                        image_size=args.image_size,
                                                                        batch_size=args.batch_size,
                                                                        num_workers=args.num_workers,
                                                                        device=device)

    # AP, mAP, inference_time 출력
    print('Average Precisions:')
    for i, class_num in enumerate(ap_class):
        print('\tClass {} ({}) - AP: {:.02f}'.format(class_num, class_names[class_num], AP[i] * 100))
    print('mAP: {:.02f}'.format(AP.mean() * 100))
    print('Inference_time (ms): {:.02f}'.format(inference_time))
    print('FPS: {:.02f}'.format(fps))

    # AP, mAP, inference_time을 csv 파일로 저장
    os.makedirs('csv', exist_ok=True)
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    with open('csv/test{}.csv'.format(now), mode='w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        writer.writerow(['Class Number', 'Class Name', 'AP'])
        for i, class_num in enumerate(ap_class):
            writer.writerow([class_num, class_names[class_num], AP[i] * 100])
        writer.writerow(['mAP', AP.mean() * 100, ' '])
        writer.writerow(['Inference_time (ms)', inference_time, ' '])
        writer.writerow(['FPS', fps, ' '])
    print('Saved result csv file.')

    dummy_input = torch.randn(1,3,448,448, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Latency : ",mean_syn)
