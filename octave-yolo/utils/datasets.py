import os
import glob
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms
import numpy as np
from PIL import Image


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad


def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, image_size):
        self.image_files = sorted(glob.glob("{}/*.*".format(folder_path)))
        self.image_size = image_size

    def __getitem__(self, index):
        image_path = self.image_files[index]

        # Extract image as PyTorch tensor
        image = torchvision.transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        # Pad to square resolution
        image, _ = pad_to_square(image)

        # Resize
        image = resize(image, self.image_size)
        return image_path, image

    def __len__(self):
        return len(self.image_files)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, image_size: int, augment: bool, multiscale: bool, normalized_labels=True):
        with open(list_path, 'r') as file:
            self.image_files = file.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                .replace('JPEGImages', 'labels') for path in self.image_files]
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.image_files[index].rstrip()

        # Apply augmentations
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # Extract image as PyTorch tensor
        image = transforms(Image.open(image_path).convert('RGB'))

        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        image, pad = pad_to_square(image)
        _, padded_h, padded_w = image.shape

        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(320, 608 + 1, 64))

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets
