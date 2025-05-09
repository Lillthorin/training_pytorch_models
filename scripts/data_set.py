

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from PIL import Image
import torch
import os
import cv2
import random


import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2


class CocoDataset(Dataset):
    def __init__(self, root, annFile, image_size, transforms=None, verbose=True, mosaic_prob=0.25):
        self.root = root
        self.coco = COCO(annFile)
        self.transforms = transforms
        self.mosaic_prob = mosaic_prob
        self.image_size = int(image_size/2)

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.valid_ids = self._validate_dataset(verbose=verbose)

        self.cat_id_to_index = {
            cat['id']: idx for idx, cat in enumerate(self.coco.dataset['categories'])
        }

    def _validate_dataset(self, verbose=True):
        valid_ids, missing_files, empty_anns = [], 0, 0
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, img_info['file_name'])

            if not os.path.isfile(img_path):
                missing_files += 1
                continue

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_bboxes = [
                ann for ann in anns if 'bbox' in ann and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]

            if not valid_bboxes:
                empty_anns += 1
                continue

            valid_ids.append(img_id)

        if verbose:
            print(f"[COCO Dataset Check]")
            print(f"Total images in annotationfile: {len(self.ids)}")
            print(f"Approved images with bboxes: {len(valid_ids)}")
            print(f"Missing images: {missing_files}")
            print(f"Pictures without annotations: {empty_anns}")

        return valid_ids

    def __getitem__(self, index):
        use_mosaic = random.random() < self.mosaic_prob and len(self.valid_ids) >= 4
        return self._load_mosaic(index) if use_mosaic else self._load_regular(index)

    def _load_regular(self, index):
        img_info = self.coco.loadImgs(self.valid_ids[index])[0]
        img_id = img_info["id"]
        path = img_info["file_name"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        try:
            img = np.array(Image.open(os.path.join(
                self.root, path)).convert("RGB"))
        except FileNotFoundError:
            return self.__getitem__((index + 1) % len(self))

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_index[ann['category_id']])

        if not boxes:
            return self.__getitem__((index + 1) % len(self))

        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(
                transformed['class_labels'], dtype=torch.int64)

            if boxes.ndim != 2 or boxes.shape[0] == 0:

                return self.__getitem__((index + 1) % len(self))
            _, h, w = img.shape
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                return self.__getitem__((index + 1) % len(self))
        else:
            img = ToTensorV2()(image=img)['image']
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }
        return img, target

    def _load_mosaic(self, index):
        indices = [index] + random.choices(range(len(self.valid_ids)), k=3)
        ids = [self.valid_ids[i] for i in indices]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        mosaic_img = np.full(
            (self.image_size * 2, self.image_size * 2, 3), 114, dtype=np.uint8)
        final_boxes, final_labels = [], []

        for i, img_id in enumerate(ids):
            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            try:
                img = np.array(Image.open(os.path.join(
                    self.root, path)).convert("RGB"))
            except FileNotFoundError:
                continue

            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (self.image_size, self.image_size))
            scale_x, scale_y = self.image_size / w, self.image_size / h

            offset_x = positions[i][1] * self.image_size
            offset_y = positions[i][0] * self.image_size

            mosaic_img[offset_y:offset_y + self.image_size,
                       offset_x:offset_x + self.image_size] = img_resized

            for ann in anns:
                x, y, bw, bh = ann['bbox']
                cls = self.cat_id_to_index[ann['category_id']]
                x1 = x * scale_x + offset_x
                y1 = y * scale_y + offset_y
                x2 = (x + bw) * scale_x + offset_x
                y2 = (y + bh) * scale_y + offset_y
                final_boxes.append([x1, y1, x2, y2])
                final_labels.append(cls)

        if self.transforms:
            transformed = self.transforms(
                image=mosaic_img,
                bboxes=final_boxes,
                class_labels=final_labels
            )
            if len(transformed['bboxes']) == 0:
                return self.__getitem__((index + 1) % len(self))

            mosaic_img = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(
                transformed['class_labels'], dtype=torch.int64)

            if boxes.ndim != 2 or boxes.shape[0] == 0:
                return self.__getitem__((index + 1) % len(self))

            _, h, w = mosaic_img.shape
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                return self._load_regular(index)
        else:
            mosaic_img = ToTensorV2()(image=mosaic_img)['image']
            boxes = torch.tensor(final_boxes, dtype=torch.float32)
            labels = torch.tensor(final_labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([ids[0]])
        }
        return mosaic_img, target

    def __len__(self):
        return len(self.valid_ids)
