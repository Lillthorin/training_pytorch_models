import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from pycocotools.cocoeval import COCOeval
import json
import contextlib
import io
import random
from matplotlib.patches import Rectangle


def create_confusion_matrix(targets, preds, class_names, SAVE_PATH, filename="confusion_matrix.png", title="Confusion Matrix", ):
    save_dir = os.path.join(SAVE_PATH, 'confusion_matrices')
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # förhindra division med 0
    cm_normalized = cm.astype('float') / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def match_predictions_to_targets(targets, outputs, iou_threshold=0.5, id_map=None):
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

        ious = box_iou(t_boxes, p_boxes)  # [num_targets, num_preds]

        for t_idx in range(len(t_boxes)):
            iou_row = ious[t_idx]
            best_iou, best_p_idx = iou_row.max(0)

            if best_iou >= iou_threshold:
                gt_label = t_labels[t_idx].item()
                pred_label = p_labels[best_p_idx].item()
                if id_map:
                    pred_label = id_map[pred_label]
                matched_targets.append(gt_label)
                matched_preds.append(pred_label)

    return matched_targets, matched_preds


all_targets = []
all_preds = []


def validate(model, val_loader, epoch, DEVICE, SAVE_PATH, prev_loss, start_epoch, EPOCHS, NUM_CLASSES):
    global all_targets, all_preds
    all_targets = []
    all_preds = []
    model.eval()
    coco_gt = val_loader.dataset.coco
    results = []
    image_ids = []

    id_map = {i: c['id'] for i, c in enumerate(coco_gt.dataset['categories'])}

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validering"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

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
                        "category_id": int(id_map[int(label)]),  # 🔁 FIX HÄR
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score)
                    }

                    results.append(result)

                # === 🧠 Samla alla ground truths och predictions === #
                matched_targets, matched_preds = match_predictions_to_targets(
                    targets, outputs, id_map=id_map)
                all_targets.extend(matched_targets)
                all_preds.extend(matched_preds)

    if len(results) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    result_path = os.path.join(
        SAVE_PATH, f"val_results_epoch_{epoch:03d}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(result_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        print("\n📊 COCOeval Resultat:")
        coco_eval.summarize()
        if (epoch + 1) == EPOCHS:
            class_names = [str(i) for i in range(NUM_CLASSES)]
            create_confusion_matrix(all_targets, all_preds, class_names,
                                    filename=f"confusion_matrix_last.png", title=f"Confusion Matrix Epoch {epoch+1}", SAVE_PATH=SAVE_PATH)

    mAP = coco_eval.stats[0]
    AP50 = coco_eval.stats[1]
    AP75 = coco_eval.stats[2]
    AP_small = coco_eval.stats[3]
    AP_medium = coco_eval.stats[4]
    AP_large = coco_eval.stats[5]
    if prev_loss == None or mAP > prev_loss:
        class_names = [str(i) for i in range(NUM_CLASSES)]
        create_confusion_matrix(all_targets, all_preds, class_names,
                                filename=f"confusion_matrix_best.png", title=f"Confusion Matrix Epoch {epoch+1}", SAVE_PATH=SAVE_PATH)

    os.remove(result_path)

    return mAP, AP50, AP75, AP_small, AP_medium, AP_large


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


def visualize_labels(train_loader, epoch, SAVE_PATH, num_images=4, save_path="train_labels"):
    save_dir = os.path.join(SAVE_PATH, save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Skapa en lista av alla batches
    all_batches = list(train_loader)
    # Slumpa en batch
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

        # Hantera grayscale eller RGB
        if img.shape[0] == 1:
            img_np = img.squeeze(0).numpy()
            axs[i].imshow(img_np, cmap='gray')
        else:
            img = denormalize(img).clamp(0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            axs[i].imshow(img_np)

        axs[i].set_title(f"Train Labels {i+1}")
        axs[i].axis("off")

        # Rita bboxar och klass-ID
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            axs[i].add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='blue', facecolor='none', linewidth=2
            ))
            axs[i].text(x1, y1 - 5, f"ID {int(label)}",
                        color="blue", fontsize=10)

    # Stäng av överflödiga rutor
    for j in range(num_images, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{epoch}_train_labels.png"))
    plt.close()


def visualize_predictions(model, val_loader, epoch, SAVE_PATH, DEVICE, num_images=4, name='last'):
    save_dir = os.path.join(SAVE_PATH, 'val')
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    # Skapa en lista av alla batches
    all_batches = list(val_loader)
    # Slumpa en batch
    batch_imgs, batch_targets = random.choice(all_batches)

    batch_size = len(batch_imgs)
    num_images = min(num_images, batch_size)
    selected_indices = random.sample(range(batch_size), num_images)

    # Välj slumpade bilder
    imgs = [batch_imgs[i].to(DEVICE) for i in selected_indices]
    targets = [batch_targets[i] for i in selected_indices]

    with torch.no_grad():
        outputs = model(imgs)

    def make_canvas(title_prefix, boxes_key, color, filename_prefix):
        cols = 2
        rows = math.ceil(num_images / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        for i in range(num_images):
            img = imgs[i].cpu()

            # Hantera grayscale eller RGB
            if img.shape[0] == 1:
                img_np = img.squeeze(0).numpy()
                axs[i].imshow(img_np, cmap='gray')
            else:
                img = denormalize(img).clamp(0, 1)  # 💥 Denormalisera här
                img_np = img.permute(1, 2, 0).numpy()
                axs[i].imshow(img_np)

            axs[i].set_title(f"{title_prefix} {i+1}")
            axs[i].axis("off")

            # Rita boxar
            box_source = targets[i] if boxes_key == "boxes" else outputs[i]
            for j, box in enumerate(box_source["boxes"]):
                if boxes_key == "scores" and box_source["scores"][j].item() <= 0.5:
                    continue
                x1, y1, x2, y2 = box.cpu().numpy()
                axs[i].add_patch(plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    edgecolor=color, facecolor='none', linewidth=2
                ))

                # ✍️ Lägg till score-text vid boxen om det är prediction (dvs om "scores" används)
                if boxes_key == "scores":
                    score = box_source["scores"][j].item()
                    axs[i].text(
                        x1, y1 - 5, f"{score:.2f}",
                        fontsize=12, color=color,
                        bbox=dict(facecolor='white', alpha=0.5,
                                  edgecolor='none', boxstyle='round,pad=0.2'))

        # Stäng av överflödiga rutor
        for j in range(num_images, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, f"{name}_{filename_prefix}.png"))
        plt.close()
        torch.cuda.empty_cache()

    # Spara GT och PRED canvasar
    make_canvas("Labels", "boxes", "green", "val_labels")
    make_canvas("Prediction", "scores", "red", "val_pred")
