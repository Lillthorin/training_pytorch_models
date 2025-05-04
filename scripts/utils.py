import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn_v2, MaskRCNN,
    ssd300_vgg16, ssdlite320_mobilenet_v3_large
)
from torchvision.models.detection.faster_rcnn import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, StepLR


def create_model(model_name='fasterrcnn',
                 num_classes=2,
                 pretrained_backbone=True):
    """
    Skapar detectionmodell baserat på val.
    Stöd för: fasterrcnn, retinanet, maskrcnn, ssd, ssd_mobilenetv3
    """
    model_name = model_name.lower()

    if model_name == 'ssd':
        model = ssd300_vgg16(
            weights='DEFAULT' if pretrained_backbone else None,
            num_classes=num_classes
        )
    elif model_name == 'ssd_mobilenetv3':
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone='DEFAULT' if pretrained_backbone else None,
            num_classes=num_classes
        )
    elif model_name == 'fasterrcnn':
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=ResNet50_Weights, num_classes=num_classes)
        print(f"antal klasser: {num_classes}")
    elif model_name == 'retinanet':
        model = retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def get_optimizer(optimizer_name, model, learning_rate=1e-3, weight_decay=0.0, momentum=0.9):
    """
    Returnerar en optimizer baserat på användarens val.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, EPOCHS, scheduler_name='steplr', step_size=10, gamma=0.1, patience=5,
                  max_lr=0.01, total_steps=100, warmup_steps=10,):
    """
    Returnerar en scheduler baserat på användarens val.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'steplr':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            final_div_factor=1e4,
        )
    elif scheduler_name == 'reduceonplateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=patience,
            threshold=0.001,
            min_lr=1e-6
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
