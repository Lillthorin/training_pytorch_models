# dataset.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import torch
import cv2
import random


class CocoDataset(Dataset):
    def __init__(self, root, annFile, image_size=640, transforms=None, mosaic_prob=0.25, verbose=True):
        self.root = root
        self.coco = COCO(annFile)
        self.transforms = transforms
        self.image_size = int(image_size/2)
        self.mosaic_prob = mosaic_prob

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.valid_ids = self._validate_dataset(verbose=verbose)

        self.cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(
            self.coco.dataset['categories'])}

    def _validate_dataset(self, verbose=True):
        valid_ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, img_info['file_name'])

            if not os.path.isfile(img_path):
                continue

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            has_mask = any(
                'segmentation' in ann and ann['segmentation'] for ann in anns)
            if has_mask:
                valid_ids.append(img_id)

        if verbose:
            print(
                f"[COCO Dataset Check] Valid images with masks: {len(valid_ids)} / {len(self.ids)}")
        return valid_ids

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, index):
        use_mosaic = random.random() < self.mosaic_prob and len(self.valid_ids) >= 4
        return self._load_mosaic(index) if use_mosaic else self._load_regular(index)

    def _load_regular(self, index):
        img_id = self.valid_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        img = np.array(Image.open(os.path.join(
            self.root, path)).convert("RGB"))
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for ann in anns:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_index[ann['category_id']])
            masks.append(self.coco.annToMask(ann).astype(np.uint8))

        if len(boxes) == 0 or len(masks) == 0:
            return self.__getitem__((index + 1) % len(self))

        if self.transforms:
            transformed = self.transforms(
                image=img,
                masks=masks,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            masks = transformed['masks']
        else:
            img = ToTensorV2()(image=img)['image']
            masks = np.array(masks)

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

    def _load_mosaic(self, index):
        indices = [index] + random.choices(range(len(self.valid_ids)), k=3)
        ids = [self.valid_ids[i] for i in indices]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        mosaic_img = np.full(
            (self.image_size * 2, self.image_size * 2, 3), 114, dtype=np.uint8)
        mosaic_mask = []
        final_boxes, final_labels = [], []

        for i, img_id in enumerate(ids):
            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img = np.array(Image.open(os.path.join(
                self.root, path)).convert("RGB"))

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (self.image_size, self.image_size))
            scale_x, scale_y = self.image_size / w, self.image_size / h

            offset_x = positions[i][1] * self.image_size
            offset_y = positions[i][0] * self.image_size

            mosaic_img[offset_y:offset_y + self.image_size,
                       offset_x:offset_x + self.image_size] = img_resized

            for ann in anns:
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                x, y, bw, bh = ann['bbox']
                x1 = x * scale_x + offset_x
                y1 = y * scale_y + offset_y
                x2 = (x + bw) * scale_x + offset_x
                y2 = (y + bh) * scale_y + offset_y
                final_boxes.append([x1, y1, x2, y2])
                final_labels.append(self.cat_id_to_index[ann['category_id']])

                # Skala och flytta mask
                mask = self.coco.annToMask(ann).astype(np.uint8)
                mask_resized = cv2.resize(
                    mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                full_mask = np.zeros(
                    (self.image_size * 2, self.image_size * 2), dtype=np.uint8)
                full_mask[offset_y:offset_y + self.image_size,
                          offset_x:offset_x + self.image_size] = mask_resized
                mosaic_mask.append(full_mask)

        if len(final_boxes) == 0 or len(mosaic_mask) == 0:
            return self._load_regular(index)

        if self.transforms:
            transformed = self.transforms(
                image=mosaic_img,
                masks=mosaic_mask,
                bboxes=final_boxes,
                class_labels=final_labels
            )
            mosaic_img = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(
                transformed['class_labels'], dtype=torch.int64)
            masks = torch.as_tensor(
                np.array(transformed['masks']), dtype=torch.uint8)
        else:
            mosaic_img = ToTensorV2()(image=mosaic_img)['image']
            boxes = torch.as_tensor(final_boxes, dtype=torch.float32)
            labels = torch.as_tensor(final_labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(mosaic_mask), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([ids[0]])
        }
        return mosaic_img, target
