# Updated train_maskrcnn_aug.py aligned with train.py structure
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.dataset import CocoDataset
from scripts.augment import get_train_transform, get_val_transform, get_weak_train_transform
from scripts.utils_masked import EarlyStopping, visualize_predictions, visualize_labels, plot_metrics, validate
from scripts.utils import get_optimizer, get_scheduler


def get_model(num_classes, pretrained_backbone=True):
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=weights_backbone, num_classes=num_classes)
    return model


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', param_group['lr'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"🔄 Loaded checkpoint from epoch {start_epoch}")
    return model, optimizer, start_epoch, checkpoint.get('scheduler_state_dict', None)


def train(
        DATA_DIR,
        NUM_CLASSES,
        BATCH_SIZE=4,
        EPOCHS=50,
        LR=0.001,
        MODEL_NAME='maskrcnn_resnet50',
        OPTIMIZER_NAME='sgd',
        SCHEDULER_NAME='reduceonplateau',
        PRETRAINED_BACKBONE=True,
        PATIENCE=10,
        RESUME_TRAINING=False,
        CHECKPOINT_PATH='',
        WARMUP_EPOCHS=3,
        IMGSZ = 640, AUGMENT=True):
    MAIN_FOLDER = os.path.join('runs', MODEL_NAME)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MAIN_FOLDER, exist_ok=True)
    existing_folders = [f for f in os.listdir(MAIN_FOLDER) if f.isdigit()]
    SUB_FOLDER = str(max(map(int, existing_folders)) +
                     1) if existing_folders else '1'
    SAVE_PATH = os.path.join(MAIN_FOLDER, SUB_FOLDER)
    os.makedirs(SAVE_PATH)
    BEST = os.path.join(SAVE_PATH, 'best.pt')
    LAST = os.path.join(SAVE_PATH, 'last.pt')

    BEST_MASK = os.path.join(SAVE_PATH, 'best_mask.pt')
    BEST_BBOX = os.path.join(SAVE_PATH, 'best_bbox.pt')
    LAST = os.path.join(SAVE_PATH, 'last.pt')
    CHECKPOINT_LAST = os.path.join(SAVE_PATH, 'checkpoint_last.pth')
    CHECKPOINT_BEST_MASK = os.path.join(SAVE_PATH, 'checkpoint_best_mask.pth')
    CHECKPOINT_BEST_BBOX = os.path.join(SAVE_PATH, 'checkpoint_best_bbox.pth')

    train_dataset = CocoDataset(
        root=f"{DATA_DIR}/train", annFile=f"{DATA_DIR}/annotations/train.json", transforms=get_train_transform(imgsz=IMGSZ) if AUGMENT else get_val_transform(imgsz=IMGSZ)
        )
    val_dataset = CocoDataset(
        root=f"{DATA_DIR}/valid", annFile=f"{DATA_DIR}/annotations/valid.json", transforms= get_val_transform(imgsz=IMGSZ))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(NUM_CLASSES, PRETRAINED_BACKBONE).to(DEVICE)
    optimizer = get_optimizer(
        optimizer_name=OPTIMIZER_NAME, model=model, learning_rate=LR)
    scheduler = get_scheduler(
        optimizer=optimizer, scheduler_name=SCHEDULER_NAME, EPOCHS=EPOCHS)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)

    start_epoch = 0
    if RESUME_TRAINING:
        model, optimizer, start_epoch, scheduler_state = load_checkpoint(
            model, optimizer, CHECKPOINT_PATH, DEVICE)
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)

    prev_mask_map = None
    prev_bbox_map = None
    history = []
    reduced_augmentation = False
    no_augmentation = False

    warmup_start_lr = 1e-6
    base_lr = LR
    warmup_iters = WARMUP_EPOCHS * len(train_loader)
    try:
        for epoch in range(start_epoch, EPOCHS + start_epoch):
            total_loss = 0.0
            model.train()
            print(f"\n📦 Epoch {epoch + 1}/{EPOCHS + start_epoch}")

            if epoch == start_epoch or epoch == EPOCHS + start_epoch:
                visualize_labels(train_loader, epoch, SAVE_PATH)

            if epoch + 1 == int((EPOCHS + start_epoch) * 0.5):
                if AUGMENT:
                    print("🔄 Switching to weak augmentation")
                    train_dataset.transforms = get_weak_train_transform(imgsz=IMGSZ)
            elif epoch + 1 == int((EPOCHS + start_epoch) * 0.8):
                if AUGMENT:
                    print("🔄 Switching to no augmentation")

                    train_dataset.transforms = get_val_transform(imgsz=IMGSZ)

            for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()}
                           for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                global_step = epoch * len(train_loader) + batch_idx
                if global_step < warmup_iters:
                    warmup_lr = warmup_start_lr + \
                        (base_lr - warmup_start_lr) * \
                        global_step / warmup_iters
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr

                total_loss += losses.item()

            loss = total_loss / len(train_loader)
            val_mAP_mask, val_mAP_bbox = validate(
                model, val_loader, epoch, SAVE_PATH, device=DEVICE, NUM_CLASSES=NUM_CLASSES)
            visualize_predictions(model, val_loader, epoch,
                                  SAVE_PATH, device=DEVICE)

            if SCHEDULER_NAME == 'reduceonplateau':
                scheduler.step(val_mAP_mask)
            else:
                scheduler.step()

            torch.save(model.state_dict(), LAST)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, CHECKPOINT_LAST)

            if prev_mask_map is None or val_mAP_mask > prev_mask_map:
                prev_mask_map = val_mAP_mask
                torch.save(model.state_dict(), BEST_MASK)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, CHECKPOINT_BEST_MASK)
                visualize_predictions(model, val_loader, epoch,
                                      SAVE_PATH, device=DEVICE, name='best_mask')
                print("💾 Saved BEST MASK model")

            if prev_bbox_map is None or val_mAP_bbox > prev_bbox_map:
                prev_bbox_map = val_mAP_bbox
                torch.save(model.state_dict(), BEST_BBOX)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, CHECKPOINT_BEST_BBOX)
                visualize_predictions(model, val_loader, epoch,
                                      SAVE_PATH, device=DEVICE, name='best_bbox')
                print("💾 Saved BEST BBOX model")

            history.append({
                'epoch': epoch + 1,
                'train_loss': loss,
                'val_mAP_mask': val_mAP_mask,
                'val_mAP_bbox': val_mAP_bbox,
                'lr': optimizer.param_groups[0]['lr']
            })

            if early_stopping.counter == int(early_stopping.patience * 0.5) and not reduced_augmentation and not AUGMENT:
                    print("🧪 Early stopping trigger reducing augmentation!")
                    train_dataset.transforms = get_weak_train_transform(
                        imgsz=IMGSZ)
                    reduced_augmentation = True
                    visualize_labels(train_loader, epoch, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)
                    visualize_labels(train_loader, epoch+1, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)

            elif early_stopping.counter == int(early_stopping.patience * 0.8) and not no_augmentation and not AUGMENT:
                    print(
                        "🧪 Early stopping is about to trigger! Turning off Augmentation for finetune.")
                    train_dataset.transforms = get_val_transform(imgsz=IMGSZ)
                    no_augmentation = True
                    visualize_labels(train_loader, epoch, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)
                    visualize_labels(train_loader, epoch+1, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)

            if early_stopping.early_stop:
                print(f"⏹️ Early stopp, mAP has not increased in {PATIENCE} epochs.")
                break
    except KeyboardInterrupt:
        print("Training aborted by user")

    torch.cuda.empty_cache()
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(SAVE_PATH, "loss_history.csv"), index=False)
    plot_metrics(df, SAVE_PATH)


if __name__ == "__main__":
    train(
        DATA_DIR="new_dataset",
        NUM_CLASSES=5,
        BATCH_SIZE=4,
        EPOCHS=30,
        LR=0.0005,
        PRETRAINED_BACKBONE=True,
        CONTINUE_TRAINING=False,
        CHECKPOINT_PATH=""
    )
