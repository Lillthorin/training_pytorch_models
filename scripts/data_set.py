import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os

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
        min_visibility=0.3,
        label_fields=['class_labels'],
        clip=True,
        filter_invalid_bboxes=True
    ),
    is_check_shapes=True,
    p=1.0
    )




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
        min_visibility=0.3,
        label_fields=['class_labels'],
        clip=True,
        filter_invalid_bboxes=True
    ),
    is_check_shapes=True,
    p=1.0
    )


def get_val_transform(imgsz):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        A.Resize(imgsz, imgsz, p=1),
        ToTensorV2()
    ],
     bbox_params=A.BboxParams(
        format='pascal_voc',
        min_visibility=0.3,
        label_fields=['class_labels'],
        clip=True,
        filter_invalid_bboxes=True
    ),
    is_check_shapes=True,
    p=1.0
    )

class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, verbose=True):
        self.root = root
        self.coco = COCO(annFile)
        self.transforms = transforms

        self.ids = list(sorted(self.coco.imgs.keys()))
        self.valid_ids = self._validate_dataset(verbose=verbose)

    def _validate_dataset(self, verbose=True):
        valid_ids = []
        missing_files = 0
        empty_anns = 0

        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root, img_info['file_name'])

            # 1. Kontrollera om bildfilen finns
            if not os.path.isfile(img_path):
                missing_files += 1
                continue

            # 2. Kontrollera att det finns minst en giltig bbox
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_bboxes = [ann for ann in anns if 'bbox' in ann and ann['bbox'] and ann['bbox'][2] > 0 and ann['bbox'][3] > 0]
            if len(valid_bboxes) == 0:
                empty_anns += 1
                continue

            valid_ids.append(img_id)

        if verbose:
            print(f"[COCO Dataset Check]")
            print(f"Total images in annotationfile: {len(self.ids)}")
            print(f"Approved images with bboxes: {len(valid_ids)}")
            print(f"Missing images: {missing_files}")
            print(f"Picures without annotations: {empty_anns}")

        return valid_ids

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.valid_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        try:
            img = np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))
        except FileNotFoundError:
            return self.__getitem__((index + 1) % len(self))

        boxes = []
        labels = []
        for ann in anns:
            if 'bbox' not in ann or not ann['bbox']:
                continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self))

        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor([int(x) for x in transformed['class_labels']], dtype=torch.int64)

            _, h, w = img.shape
            if boxes.numel() == 0:
                return self.__getitem__((index + 1) % len(self))
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)

            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                return self.__getitem__((index + 1) % len(self))
        else:
            img = ToTensorV2()(image=img)['image']
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.valid_ids)
