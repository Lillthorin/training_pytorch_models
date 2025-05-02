import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(imgsz):
       return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5,
            border_mode=0
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
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
