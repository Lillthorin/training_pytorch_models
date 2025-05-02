# dataset.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import torch


def get_train_transform(imgsz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.7,
            border_mode=0
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.4),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.Resize(imgsz, imgsz, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3,
        clip=True,
        filter_invalid_bboxes=True
    ))


def get_weak_train_transform(imgsz):
    return A.Compose([
        A.RandomBrightnessContrast(p=0.1),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        A.Resize(imgsz, imgsz, p=1),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3,
        clip=True,
        filter_invalid_bboxes=True
    ))


def get_val_transform(imgsz):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        A.Resize(imgsz, imgsz, p=1),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.3,
        clip=True,
        filter_invalid_bboxes=True
    ))


class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = np.array(Image.open(os.path.join(
            self.root, path)).convert("RGB"))

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            masks.append(coco.annToMask(ann).astype(np.uint8))

        # Hoppa över bilder utan giltig data
        if len(boxes) == 0 or len(masks) == 0:
            return self.__getitem__((index + 1) % len(self))

        # Transformera
        if self.transforms:
            transformed = self.transforms(
                image=img,
                masks=masks,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            masks = transformed['masks']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            transformed = ToTensorV2()(image=img)
            img = transformed['image']

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.ids)
