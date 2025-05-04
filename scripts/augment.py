import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transform(imgsz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-20, 20),
            shear=(-10, 10),
            scale=(0.85, 1.15),
            translate_percent=(0.05, 0.1),
            p=0.3,
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        # A.Resize(imgsz, imgsz, p=1),
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
        # A.Resize(imgsz, imgsz, p=1),
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
        ToTensorV2()
    ]
    )
