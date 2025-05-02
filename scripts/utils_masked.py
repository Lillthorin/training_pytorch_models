# utils.py
from matplotlib.patches import Rectangle
from torchvision.utils import draw_segmentation_masks
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import contextlib
import io
from pycocotools.cocoeval import COCOeval
import json
import random
from torchvision.transforms import functional as F
import math
from tqdm import tqdm
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import torchvision


def match_predictions_to_targets(targets, outputs, iou_threshold=0.5):
    matched_targets = []
    matched_preds = []

    for target, output in zip(targets, outputs):
        t_boxes = target["boxes"].cpu()
        t_labels = target["labels"].cpu()

        p_boxes = output["boxes"].cpu()
        p_labels = output["labels"].cpu()
        p_scores = output["scores"].cpu()

        if len(t_boxes) == 0 or len(p_boxes) == 0:
            continue

        # Filtera bort prediktioner med låg score
        mask = p_scores > 0.5
        p_boxes = p_boxes[mask]
        p_labels = p_labels[mask]

        if len(p_boxes) == 0:
            continue

        ious = torchvision.ops.box_iou(
            t_boxes, p_boxes)  # [num_targets, num_preds]

        for t_idx in range(len(t_boxes)):
            iou_row = ious[t_idx]
            best_iou, best_p_idx = iou_row.max(0)

            if best_iou >= iou_threshold:
                matched_targets.append(t_labels[t_idx].item())
                matched_preds.append(p_labels[best_p_idx].item())

    return matched_targets, matched_preds


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


def create_confusion_matrix(targets, preds, class_names, filename, save_dir='confusion_matrices', save_path=""):
    save_dir = os.path.join(save_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


# Enkel färgpalett med fasta färger för klass-ID


def visualize_labels(train_loader, epoch, SAVE_PATH, num_images=4, save_path="train_labels"):
    save_dir = os.path.join(SAVE_PATH, save_path)
    os.makedirs(save_dir, exist_ok=True)

    all_batches = list(train_loader)
    batch_imgs, batch_targets = random.choice(all_batches)

    batch_size = len(batch_imgs)
    num_images = min(num_images, batch_size)
    selected_indices = random.sample(range(batch_size), num_images)

    imgs = [batch_imgs[i] for i in selected_indices]
    targets = [batch_targets[i] for i in selected_indices]

    cols = 2
    rows = math.ceil(num_images / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i in range(num_images):
        img = imgs[i].cpu()
        target = targets[i]

        # Denormalisera + omvandla till numpy
        if img.shape[0] == 1:
            img_np = img.squeeze(0).numpy()
            axs[i].imshow(img_np, cmap='gray')
        else:
            img = denormalize(img).clamp(0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            axs[i].imshow(img_np)

        axs[i].set_title(f"Train Labels {i+1}")
        axs[i].axis("off")

        # Rita segmenteringsmasker i rött
        if "masks" in target:
            masks = target["masks"]
            for mask in masks:
                mask = mask.cpu().numpy()
                red_mask = np.zeros((*mask.shape, 3))
                red_mask[..., 0] = mask  # Röd kanal
                axs[i].imshow(red_mask, alpha=0.2)

        # Rita bboxar + klass-ID
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            axs[i].add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='blue', facecolor='none', linewidth=2
            ))
            axs[i].text(x1, y1 - 5, f"ID {int(label)}",
                        color="blue", fontsize=10)

    # Töm överflödiga rutor
    for j in range(num_images, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{epoch}_train_labels.png"))
    plt.close()


def visualize_predictions(model, val_loader, epoch, SAVE_PATH, device, num_images=4, name='last'):
    import math
    save_dir = os.path.join(SAVE_PATH, 'val')
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_batches = list(val_loader)
    batch_imgs, batch_targets = random.choice(all_batches)

    batch_size = len(batch_imgs)
    num_images = min(num_images, batch_size)
    selected_indices = random.sample(range(batch_size), num_images)

    imgs = [batch_imgs[i].to(device) for i in selected_indices]
    targets = [batch_targets[i] for i in selected_indices]

    with torch.no_grad():
        outputs = model(imgs)

    def denormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=img_tensor.device).view(3, 1, 1)
        return img_tensor * std + mean

    def make_canvas(title_prefix, boxes_key, color, filename_prefix):
        cols = 2
        rows = math.ceil(num_images / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for i in range(num_images):
            img = imgs[i].cpu()
            img = denormalize(img).clamp(0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            axs[i].imshow(img_np)

            axs[i].set_title(f"{title_prefix} {i+1}")
            axs[i].axis("off")

            box_source = targets[i] if boxes_key == "boxes" else outputs[i]

            # Rita boxar
            for j, box in enumerate(box_source["boxes"]):
                if boxes_key == "scores" and box_source["scores"][j].item() <= 0.5:
                    continue
                x1, y1, x2, y2 = box.cpu().numpy()
                axs[i].add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor=color, facecolor='none', linewidth=2
                ))

                # Score-text vid predictions
                if boxes_key == "scores":
                    score = box_source["scores"][j].item()
                    axs[i].text(
                        x1, y1 - 5, f"{score:.2f}",
                        fontsize=12, color=color,
                        bbox=dict(facecolor='white', alpha=0.5,
                                  edgecolor='none', boxstyle='round,pad=0.2'))

            # Rita maskerna om de finns
            if "masks" in box_source:
                masks = box_source["masks"].cpu()
                for m in range(masks.shape[0]):
                    if boxes_key == "scores" and box_source["scores"][m].item() <= 0.5:
                        continue
                    mask = masks[m][0] if masks[m].dim() == 3 else masks[m]
                    mask = mask.numpy()
                    colored_mask = np.zeros((*mask.shape, 4))
                    if title_prefix == "Labels":
                        colored_mask[mask > 0.5] = (0, 1, 0, 0.4)  # grön
                    else:
                        colored_mask[mask > 0.5] = (1, 0, 0, 0.4)  # röd
                    axs[i].imshow(colored_mask)

        for j in range(num_images, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_{filename_prefix}.png"))
        plt.close()

    # Kör båda varianterna
    make_canvas("Labels", "boxes", "green", "val_labels")
    make_canvas("Prediction", "scores", "red", "val_pred")
    torch.cuda.empty_cache()


previous_bbox_score = 0


def validate(model, val_loader, epoch, SAVE_PATH, device, NUM_CLASSES):
    global previous_bbox_score
    model.eval()
    coco_gt = val_loader.dataset.coco
    results = []
    image_ids = []
    all_targets, all_preds = [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", ncols=120)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            outputs = model(images)

            matched_targets, matched_preds = match_predictions_to_targets(
                targets, outputs)
            all_targets.extend(matched_targets)
            all_preds.extend(matched_preds)

            for output, target in zip(outputs, targets):
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels_pred = output["labels"].cpu()
                image_id = int(target["image_id"].item())
                image_ids.append(image_id)

                for box, score, label in zip(boxes, scores, labels_pred):
                    x_min, y_min, x_max, y_max = box.tolist()
                    width = x_max - x_min
                    height = y_max - y_min
                    result = {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    }
                    results.append(result)

    if len(results) == 0:
        return 0.0, 0.0

    result_path = os.path.join(
        SAVE_PATH, f"val_results_epoch_{epoch:03d}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(result_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if coco_eval.stats[1] > previous_bbox_score:
            previous_bbox_score = coco_eval.stats[1]
            class_names = [str(i) for i in range(NUM_CLASSES)]
            create_confusion_matrix(all_targets, all_preds, class_names,
                                    filename=f"confusion_matrix_best.png", save_path=SAVE_PATH)

    os.remove(result_path)
    torch.cuda.empty_cache()
    return coco_eval.stats[0], coco_eval.stats[1]


def plot_metrics(df, SAVE_PATH):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.flatten()
    axs[0].plot(df["epoch"], df["train_loss"], marker='o')
    axs[0].set_title("Train Loss")
    axs[1].plot(df["epoch"], df["val_mAP_mask"], marker='s', label="Mask mAP")
    axs[1].plot(df["epoch"], df["val_mAP_bbox"], marker='^', label="BBox mAP")
    axs[1].legend()
    axs[1].set_title("Validation mAP")
    axs[2].plot(df["epoch"], df["lr"], marker='d')
    axs[2].set_title("Learning Rate")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "all_metrics_grid.png"))
    plt.close()
