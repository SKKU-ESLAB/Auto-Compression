import argparse
import os

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tqdm

import model.yolov3
import utils.datasets
import utils.utils

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="./data/voc_test", help="path to image folder")
parser.add_argument("--save_folder", type=str, default='./demo', help='path to saving result folder')
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--data_config", type=str, default="config/voc.data", help="path to data config file")
parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                    help="path to pretrained weights file")
parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 설정값을 가져오기
data_config = utils.utils.parse_data_config(args.data_config)
num_classes = int(data_config['classes'])
class_names = utils.utils.load_classes(data_config['names'])

# 모델 준비하기
model = model.yolov3.YOLOv3(args.image_size, num_classes).to(device)
if args.pretrained_weights.endswith('.pth'):
    model.load_state_dict(torch.load(args.pretrained_weights))
else:
    model.load_darknet_weights(args.pretrained_weights)

# 데이터셋, 데이터로더 설정
dataset = utils.datasets.ImageFolder(args.image_folder, args.image_size)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers)

# 객체를 검출하는 코드
model.eval()  # 모델을 evaluation mode로 설정
img_predictions = []  # 각 이미지의 예측 결과 저장
img_paths = []  # 각 이미지의 경로 저장
for paths, images in tqdm.tqdm(dataloader, desc='Batch'):
    with torch.no_grad():
        images = images.to(device)
        prediction = model(images)
        prediction = utils.utils.non_max_suppression(prediction, args.conf_thres, args.nms_thres)

    # 예측 결과 저장
    img_predictions.extend(prediction)
    img_paths.extend(paths)

# bounding box colormap 설정
cmap = np.array(plt.cm.get_cmap('Paired').colors)
cmap_rgb: list = np.multiply(cmap, 255).astype(np.int32).tolist()

# 결과 이미지를 저장하는 코드
os.makedirs(args.save_folder, exist_ok=True)
for path, prediction in tqdm.tqdm(zip(img_paths, img_predictions), desc='Save images', total=dataset.__len__()):
    # 원본 이미지 열기
    path = path.replace('\\', '/')
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)

    if prediction is not None:
        # 원본 이미지로 bounding box를 rescale한다.
        prediction = utils.utils.rescale_boxes_original(prediction, args.image_size, image.size)

        for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in prediction:
            # bounding box color 설정
            color = tuple(cmap_rgb[int(cls_pred) % len(cmap_rgb)])

            # bounding box 그리기
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)

            # label 그리기
            text = '{} {:.1f}'.format(class_names[int(cls_pred)], obj_conf.item() * 100)
            font = ImageFont.load_default()
            text_width, text_height = font.getsize(text)
            draw.rectangle(((x1, y1), (x1 + text_width, y1 + text_height)), fill=color)
            draw.text((x1, y1), text, fill=(0, 0, 0), font=font)

    # 결과 이미지 저장
    filename = path.split('/')[-1]
    image.save(os.path.join(args.save_folder, filename))
    image.close()
