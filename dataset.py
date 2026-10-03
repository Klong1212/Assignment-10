"""Dataset utilities for lane segmentation from Ultralytics YOLO-seg polygon labels.

Expected on-disk layout (Ultralytics convention):

    <data_root>/images/train/*.jpg
    <data_root>/images/val/*.jpg
    <data_root>/labels/train/*.txt
    <data_root>/labels/val/*.txt

Each label file has one line per polygon instance:

    class_id x1 y1 x2 y2 x3 y3 ... xn yn

with all coordinates normalized to [0, 1] relative to the *original* image
size. Since source images vary in resolution, masks are rasterized at the
original resolution first, then image and mask are resized together to the
network's fixed 48x48 input/output size.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for_image(image_path, labels_dir):
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"


def load_yolo_seg_polygons(label_path, img_w, img_h, lane_class_id=0):
    """Parse a YOLO-seg label file into a list of int32 pixel-coordinate polygons for the target lane class."""
    label_path = Path(label_path)
    polygons = []
    if not label_path.exists():
        return polygons
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 (x, y) points
                continue
            if lane_class_id is not None and int(parts[0]) != lane_class_id:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polygons.append(coords.astype(np.int32))
    return polygons


def polygons_to_mask(polygons, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask


def augment_image(image):
    """Slight photometric augmentation to mimic outdoor sunlight/sensor noise.

    Applies, each with independent probability: a small per-channel white
    balance gain shift, a small brightness shift, and a light Gaussian blur.
    """
    img = image.astype(np.float32)

    if random.random() < 0.5:
        gains = np.random.uniform(0.92, 1.08, size=3).astype(np.float32)
        img = img * gains[np.newaxis, np.newaxis, :]

    if random.random() < 0.5:
        delta = np.random.uniform(-25.0, 25.0)
        img = img + delta

    img = np.clip(img, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


class LaneSegDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=48, augment=False, lane_class_id=0):
        self.images = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.lane_class_id = lane_class_id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        label_path = label_path_for_image(image_path, self.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h, lane_class_id=self.lane_class_id)
        mask = polygons_to_mask(polygons, w, h)

        if self.augment:
            image = augment_image(image)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return image_t, mask_t
