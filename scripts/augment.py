import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_strong_transform(imgsz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            rotate=(-20, 20),
            shear=(-10, 10),
            scale=(0.85, 1.15),
            translate_percent=(0.05, 0.1),
            p=0.3,
        ),
        A.ElasticTransform(alpha=1, sigma=50, p=0.1),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.ChannelShuffle()
        ], p=0.6),

        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),  shadow_dimension=5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0.0, 1.0))
        ], p=0.2),

        A.CoarseDropout(num_holes_range=(3, 10), hole_height_range=(
            0.01, 0.05), hole_width_range=(0.01, 0.05), p=0.2),

        A.OneOf([
            A.GaussNoise(),
            A.MotionBlur(blur_limit=3)
        ], p=0.3),
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


def get_middle_train_transform(imgsz):
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.Affine(
            rotate=(-10, 10),
            shear=(-5, 5),
            scale=(0.95, 1.05),
            translate_percent=(0.02, 0.05),
            p=0.2,
        ),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.HueSaturationValue(),
            A.RGBShift()
        ], p=0.4),

        A.OneOf([
            A.GaussNoise(),
            A.MotionBlur(blur_limit=3)
        ], p=0.2),

        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),  shadow_dimension=5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0.0, 0.5))
        ], p=0.1),

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


        A.OneOf([
            A.GaussNoise(),
            A.MotionBlur(blur_limit=3)
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.ColorJitter(),
            A.RGBShift()
        ], p=0.2),




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


def get_val_transform(imgsz):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )
