import os
import torch
import pandas as pd
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)
from torchvision.models.detection.faster_rcnn import ResNet50_Weights, MobileNet_V3_Large_Weights, FasterRCNN_ResNet50_FPN_Weights

from torchvision.transforms import functional as F
from scripts.data_set import CocoDataset
from scripts.augment import get_val_transform, get_train_transform, get_weak_train_transform
from scripts.visuallise import validate, visualize_labels, visualize_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scripts.utils import get_optimizer, get_scheduler


def train(DATA_DIR, MODEL_NAME,  EPOCHS, NUM_CLASSES, BATCH_SIZE=4, IMGSZ=640, LR=0.001, AUGMENT=True, OPTIMIZER_NAME='adamw', PRETRAINED_BACKBONE=True, SCHEDULER_NAME='reduceonplateau', PATIENCE=10, RESUME_TRAINING=False, CHECKPOINT_PATH='', EXPORT_TORCHSCRIPT=False, WARMUP_EPOCHS=3):
    # === Konfig ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = DATA_DIR
    BATCH_SIZE = BATCH_SIZE
    EPOCHS = EPOCHS
    LR = LR
    NUM_CLASSES = NUM_CLASSES

    MAIN_FOLDER = os.path.join('runs', MODEL_NAME)

    IMGSZ = IMGSZ

    os.makedirs(MAIN_FOLDER, exist_ok=True)
    existing_folders = [f for f in os.listdir(MAIN_FOLDER) if f.isdigit()]
    SUB_FOLDER = str(max(map(int, existing_folders)) +
                     1) if existing_folders else '1'
    SAVE_PATH = os.path.join(MAIN_FOLDER, SUB_FOLDER)
    os.makedirs(SAVE_PATH)
    BEST = os.path.join(SAVE_PATH, 'best.pt')
    LAST = os.path.join(SAVE_PATH, 'last.pt')

    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def __call__(self, current_score):
            if self.best_score is None:
                self.best_score = current_score
            elif current_score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0

    def load_checkpoint(model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Sätt initial_lr om det saknas
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])

        start_epoch = checkpoint['epoch'] + 1  # Fortsätt från nästa epoch
        print(f"🔄 Loaded checkpoint  {start_epoch}")
        return model, optimizer, start_epoch, checkpoint.get('scheduler_state_dict', None)

    def get_model():
        if MODEL_NAME.lower() == 'fasterrcnn_resnet50':

            if PRETRAINED_BACKBONE:
                print('Loading fasterrcnn_resnet50_fpn with pretrained backbone...')
                model = fasterrcnn_resnet50_fpn(
                    weights=None,
                    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                    num_classes=NUM_CLASSES
                )
            else:
                print('Loading fasterrcnn_resnet50_fpn without pretrained weights...')
                model = fasterrcnn_resnet50_fpn(
                    weights=None,
                    weights_backbone=None,
                    num_classes=NUM_CLASSES
                )
        elif MODEL_NAME.lower() == 'fasterrcnn_mobile':

            if PRETRAINED_BACKBONE:
                print(
                    'Loading fasterrcnn_mobilenet_v3_large_fpn with pretrained backbone...')
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=None,
                    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                    num_classes=NUM_CLASSES
                )
            else:
                print(
                    'Loading fasterrcnn_mobilenet_v3_large_fpn without pretrained weights...')
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=None,
                    weights_backbone=None,
                    num_classes=NUM_CLASSES
                )
        elif MODEL_NAME.lower() == 'fasterrcnn_mobile_320':

            if PRETRAINED_BACKBONE:
                print(
                    'Loading fasterrcnn_mobilenet_v3_large_320_fpn with pretrained backbone...')
                model = fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=None,
                    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                    num_classes=NUM_CLASSES
                )
            else:
                print(
                    'Loading fasterrcnn_mobilenet_v3_large_320_fpn without pretrained weights...')
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=None,
                    weights_backbone=None,
                    num_classes=NUM_CLASSES
                )

        elif MODEL_NAME.lower() == 'retinanet_resnet50_fpn_v2':

            if PRETRAINED_BACKBONE:
                print('Loading RetinaNet with pretrained backbone...')
                model = retinanet_resnet50_fpn_v2(
                    weights=None,
                    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                    num_classes=NUM_CLASSES
                )
            else:
                print('Loading RetinaNet without pretrained weights...')
                model = retinanet_resnet50_fpn_v2(
                    weights=None,
                    weights_backbone=None,
                    num_classes=NUM_CLASSES
                )

        elif MODEL_NAME.lower() == 'ssd_mobilenetv3':
            if PRETRAINED_BACKBONE:
                print('Loading SSD with pretrained backbone...')
                model = ssdlite320_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                    num_classes=NUM_CLASSES
                )
            else:
                print('Loading SSD without pretrained weights...')
                model = ssdlite320_mobilenet_v3_large(
                    weights=None,
                    weights_backbone=None,
                    num_classes=NUM_CLASSES
                )
        else:
            raise ValueError(f"Unsupported model name: {MODEL_NAME}")

        return model

    def training():
        try:
            prev_loss = None
            start_epoch = 0
            print('Loading and checking training dataset...')
            train_dataset = CocoDataset(
                root=f"{DATA_DIR}/train",
                annFile=f"{DATA_DIR}/annotations/train.json",
                image_size=IMGSZ,
                transforms=(get_train_transform(
                    imgsz=IMGSZ) if AUGMENT else get_val_transform(imgsz=IMGSZ))
            )
            print('Loading and checking valid dataset...')
            val_dataset = CocoDataset(
                root=f"{DATA_DIR}/valid",
                annFile=f"{DATA_DIR}/annotations/valid.json",
                image_size=IMGSZ,
                transforms=get_val_transform(imgsz=IMGSZ)
            )

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=lambda x: tuple(zip(*x)), drop_last=True)

            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

            model = get_model().to(DEVICE)

            optimizer = get_optimizer(
                optimizer_name=OPTIMIZER_NAME, model=model, learning_rate=LR)

            early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
            scheduler = get_scheduler(
                optimizer=optimizer, scheduler_name=SCHEDULER_NAME, EPOCHS=start_epoch-1)
            if RESUME_TRAINING:
                model, optimizer, start_epoch, scheduler_state = load_checkpoint(
                    model, optimizer, CHECKPOINT_PATH)
                scheduler = get_scheduler(
                    optimizer=optimizer,
                    scheduler_name=SCHEDULER_NAME,
                    EPOCHS=start_epoch  # 💥 viktigt!
                )
                if scheduler_state:
                    scheduler.load_state_dict(scheduler_state)

            history = []
            reduced_augmentation = False
            no_augmentation = False

            base_lr = LR
            warmup_start_lr = 1e-6
            warmup_epochs = WARMUP_EPOCHS

            visualize_labels(train_loader, epoch=1, num_images=4,
                             save_path="train_labels", SAVE_PATH=SAVE_PATH)
            visualize_labels(train_loader, epoch=2, num_images=4,
                             save_path="train_labels", SAVE_PATH=SAVE_PATH)
            warmup_iters = warmup_epochs * len(train_loader)

            for epoch in range(start_epoch, EPOCHS+start_epoch):
                total_loss = 0.0
                model.train()
                print(f"\n📦 Epoch {epoch + 1}/{EPOCHS + start_epoch}")

                # === Stäng av augmentering i slutet ===
                if epoch + 1 == int(EPOCHS+start_epoch * 0.5):
                    if AUGMENT:
                        print("🔄 Changing augmentation state to low")
                        train_dataset.transforms = get_weak_train_transform(
                            imgsz=IMGSZ)
                if epoch + 1 == int(EPOCHS+start_epoch * 0.8):
                    if AUGMENT:
                        print("🔄 Chaning to no augmentation")
                        train_dataset.transforms = get_val_transform(
                            imgsz=IMGSZ)

                for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
                    images = [img.to(DEVICE) for img in images]
                    targets = [{k: v.to(DEVICE) for k, v in t.items()}
                               for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    total_loss += losses.item()
                    global_step = epoch * len(train_loader) + batch_idx

                    if global_step < warmup_iters:
                        warmup_lr = warmup_start_lr + \
                            (base_lr - warmup_start_lr) * \
                            global_step / warmup_iters
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = warmup_lr

                loss = total_loss / len(train_loader)

                mAP, AP50, AP75, AP_small, AP_medium, AP_large = validate(
                    model, val_loader, epoch, DEVICE=DEVICE, SAVE_PATH=SAVE_PATH, prev_loss=prev_loss, NUM_CLASSES=NUM_CLASSES, start_epoch=start_epoch, EPOCHS=EPOCHS)

                print(
                    f"🧪 Train Loss: {loss:.4f} | mAP: {mAP:.4f} | AP50: {AP50:.4f} | AP75: {AP75:.4f}")
                if SCHEDULER_NAME == 'reduceonplateau' and global_step >= warmup_iters:
                    scheduler.step(mAP)
                    early_stopping(mAP)
                elif global_step >= warmup_iters:
                    scheduler.step()
                    early_stopping(mAP)

                if (epoch+1) == EPOCHS+start_epoch:
                    visualize_predictions(
                        model, val_loader, epoch=epoch+1, SAVE_PATH=SAVE_PATH, DEVICE=DEVICE)

                CHECKPOINT_PATH_LAST = os.path.join(
                    SAVE_PATH, 'checkpoint_last.pth')
                CHECKPOINT_PATH_BEST = os.path.join(
                    SAVE_PATH, 'checkpoint_best.pth')

                model_path = LAST
                torch.save(model.state_dict(), model_path)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, CHECKPOINT_PATH_LAST)
                print(f"💾 Model saved to: {model_path}")
                visualize_predictions(
                    model, val_loader, epoch=epoch+1, name='last', SAVE_PATH=SAVE_PATH, DEVICE=DEVICE)
                if prev_loss == None or mAP > prev_loss:
                    prev_loss = mAP

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, CHECKPOINT_PATH_BEST)
                    torch.save(model.state_dict(), BEST)
                    print(f"💾 Model saved to: {BEST}")
                    visualize_predictions(
                        model, val_loader, epoch=epoch+1, name='best', SAVE_PATH=SAVE_PATH, DEVICE=DEVICE)

                history.append({
                    "epoch": epoch + 1,
                    "train_loss": loss,
                    "val_mAP": mAP,
                    "val_AP50": AP50,
                    "val_AP75": AP75,
                    "val_AP_small": AP_small,
                    "val_AP_medium": AP_medium,
                    "val_AP_large": AP_large,
                    "lr": optimizer.param_groups[0]['lr']
                })
                # Kolla EarlyStopping på val mAP

                # === Kolla EarlyStopping och hantera adaptiv augmentering ===
                if early_stopping.counter == int(early_stopping.patience * 0.5) and not reduced_augmentation:
                    if AUGMENT:
                        print("🧪 Early stopping trigger reducing augmentation!")
                        train_dataset.transforms = get_weak_train_transform(
                            imgsz=IMGSZ)
                        reduced_augmentation = True
                        visualize_labels(train_loader, epoch, num_images=4,
                                         save_path="train_labels", SAVE_PATH=SAVE_PATH)
                        visualize_labels(train_loader, epoch+1, num_images=4,
                                         save_path="train_labels", SAVE_PATH=SAVE_PATH)

                elif early_stopping.counter == early_stopping.patience - 5 and not no_augmentation and AUGMENT:

                    print(
                        "🧪 Early stopping is about to trigger! Turning off Augmentation for finetune.")
                    train_dataset.transforms = get_val_transform(imgsz=IMGSZ)
                    no_augmentation = True
                    visualize_labels(train_loader, epoch, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)
                    visualize_labels(train_loader, epoch+1, num_images=4,
                                     save_path="train_labels", SAVE_PATH=SAVE_PATH)

                if early_stopping.early_stop:
                    print(
                        f"⏹️ Early stopp, mAP has not increased in {PATIENCE} epochs.")
                    break
        except KeyboardInterrupt:
            print("Training aborted by user")

        torch.cuda.empty_cache()

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        axs = axs.flatten()
        # === Sparar history ===
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(SAVE_PATH, "loss_history.csv"), index=False)
        # Plot 1: Loss
        axs[0].plot(df["epoch"], df["train_loss"], marker='o', color='blue')
        axs[0].set_title("Train Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].grid(True)

        # Plot 2: mAP
        axs[1].plot(df["epoch"], df["val_mAP"], marker='s', color='green')
        axs[1].set_title("mAP")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("mAP")
        axs[1].grid(True)

        # Plot 3: AP50
        axs[2].plot(df["epoch"], df["val_AP50"], marker='^', color='orange')
        axs[2].set_title("AP50")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("AP50")
        axs[2].grid(True)

        # Plot 4: AP75
        axs[3].plot(df["epoch"], df["val_AP75"], marker='d', color='red')
        axs[3].set_title("AP75")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("AP75")
        axs[3].grid(True)

        # Plot 5: LR
        axs[4].plot(df["epoch"], df["lr"], marker='x', color='purple')
        axs[4].set_title("Learning Rate")
        axs[4].set_xlabel("Epoch")
        axs[4].set_ylabel("LR")
        axs[4].grid(True)

        # Plot 6: AP_small / AP_medium / AP_large
        axs[5].plot(df["epoch"], df["val_AP_small"],
                    label="AP Small", marker='P')
        axs[5].plot(df["epoch"], df["val_AP_medium"],
                    label="AP Medium", marker='*')
        axs[5].plot(df["epoch"], df["val_AP_large"],
                    label="AP Large", marker='h')
        axs[5].set_title("AP Small / Medium / Large")
        axs[5].set_xlabel("Epoch")
        axs[5].set_ylabel("AP")
        axs[5].grid(True)
        axs[5].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, "all_metrics_grid.png"))
        plt.close()

    training()
