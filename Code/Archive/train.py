"""
Training script for Gland vs Solid patch classifier.

Usage:
    python train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from dataset import PatchDataset, get_train_transforms, get_val_transforms
from model import create_model, freeze_early_layers, unfreeze_all


def make_sampler(dataset):
    """WeightedRandomSampler to handle class imbalance."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    weights_per_class = 1.0 / class_counts
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "acc": correct / total,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    total = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_labels).sum() / total
    f1 = f1_score(all_labels, all_preds, average="binary")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        "loss": running_loss / total,
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


def save_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    config = Config()
    config.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    train_dataset = PatchDataset(
        Path(config.output_dir) / "train",
        transform=get_train_transforms(config.input_size),
    )
    val_dataset = PatchDataset(
        Path(config.output_dir) / "val",
        transform=get_val_transforms(config.input_size),
    )

    print(f"Train: {len(train_dataset)} patches, {train_dataset.get_class_counts()}")
    print(f"Val:   {len(val_dataset)} patches, {val_dataset.get_class_counts()}")

    # DataLoaders
    train_sampler = make_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    model = create_model(num_classes=config.num_classes, pretrained=True)
    freeze_early_layers(model)
    model = model.to(device)

    # Loss with class weights
    train_labels = train_dataset.get_labels()
    class_counts = np.bincount(train_labels).astype(float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)

    # Training loop
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        # Unfreeze at specified epoch
        if epoch == config.unfreeze_epoch + 1:
            print(f"\n>>> Epoch {epoch}: Unfreezing all layers, reducing lr to {config.lr / 10}")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr / 10, weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs - epoch + 1,
            )

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
            f"LR: {lr:.6f}"
        )

        # TensorBoard logging
        writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("Accuracy", {"train": train_metrics["acc"], "val": val_metrics["acc"]}, epoch)
        writer.add_scalar("Val/F1", val_metrics["f1"], epoch)
        writer.add_scalar("Val/AUC", val_metrics["auc"], epoch)
        writer.add_scalar("LR", lr, epoch)

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            ckpt_path = Path(config.checkpoint_dir) / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1,
                "val_acc": val_metrics["acc"],
                "val_auc": val_metrics["auc"],
            }, ckpt_path)
            print(f"  >> Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.patience} epochs)")
                break

    writer.close()

    # Final evaluation with best model
    print(f"\n{'='*60}")
    print(f"Final evaluation with best model (F1={best_f1:.4f})")
    print(f"{'='*60}")

    ckpt = torch.load(Path(config.checkpoint_dir) / "best_model.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = validate(model, val_loader, criterion, device)

    print(f"\nClassification Report:")
    print(classification_report(
        val_metrics["labels"], val_metrics["preds"],
        target_names=config.class_names,
    ))

    save_confusion_matrix(
        val_metrics["labels"], val_metrics["preds"],
        config.class_names,
        Path(config.viz_dir) / "confusion_matrix.png",
    )
    print(f"Confusion matrix saved to {config.viz_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
