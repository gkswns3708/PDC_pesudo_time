"""
Leave-One-Slide-Out Cross-Validation for Gland vs Solid classification (DDP).

Launch with torchrun:
    torchrun --standalone --nnodes=1 --nproc_per_node=<N_GPUS> train_cv.py

Single-GPU fallback (no torchrun):
    python train_cv.py
"""

import os
import sys
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config import Config
from dataset import PatchDataset, get_train_transforms, get_val_transforms
from model import create_model, freeze_early_layers, unfreeze_all


# ─────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP from torchrun env vars. Returns (rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def barrier(device=None):
    if dist.is_available() and dist.is_initialized():
        if device is not None:
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


# ─────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────

def _amp_dtype_of(config):
    m = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return m.get(getattr(config, "amp_dtype", "float32"), torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs, rank,
                    fold_tag="", amp_dtype=torch.float32):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = amp_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"

    if is_main(rank):
        pbar = tqdm(loader, desc=f"  {fold_tag} Epoch {epoch:2d}/{epochs} [Train]",
                    leave=True, dynamic_ncols=True, mininterval=1.0)
        iterator = pbar
    else:
        iterator = loader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if is_main(rank):
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}",
                             acc=f"{correct/max(total,1):.4f}")

    # Reduce per-rank stats across ranks
    stats = torch.tensor([running_loss, correct, total], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, total_sum = stats.tolist()
    return {"loss": loss_sum / max(total_sum, 1), "acc": correct_sum / max(total_sum, 1)}


@torch.no_grad()
def validate(model, dataset, criterion, device, batch_size, num_workers, epoch, epochs,
             rank, world_size, fold_tag="", amp_dtype=torch.float32):
    """Distributed val: each rank evaluates its shard, then all-gather predictions."""
    model.eval()
    use_amp = amp_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True)

    local_loss = 0.0
    local_total = 0
    local_preds, local_labels, local_probs = [], [], []

    if is_main(rank):
        pbar = tqdm(loader, desc=f"  {fold_tag} Epoch {epoch:2d}/{epochs} [Val]  ",
                    leave=True, dynamic_ncols=True, mininterval=1.0)
        iterator = pbar
    else:
        iterator = loader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        local_loss += loss.item() * images.size(0)
        local_total += images.size(0)
        probs = torch.softmax(outputs.float(), dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        local_preds.append(preds.cpu())
        local_labels.append(labels.cpu())
        local_probs.append(probs.cpu())

    local_preds = torch.cat(local_preds) if local_preds else torch.empty(0, dtype=torch.long)
    local_labels = torch.cat(local_labels) if local_labels else torch.empty(0, dtype=torch.long)
    local_probs = torch.cat(local_probs) if local_probs else torch.empty(0, dtype=torch.float)

    # Gather across ranks (variable-sized tensors → pad + gather)
    def gather_tensor(t):
        if not (dist.is_available() and dist.is_initialized()):
            return t
        size = torch.tensor([t.numel()], device=device)
        sizes = [torch.zeros_like(size) for _ in range(world_size)]
        dist.all_gather(sizes, size)
        max_size = int(max(s.item() for s in sizes))
        padded = torch.zeros(max_size, dtype=t.dtype, device=device)
        padded[:t.numel()] = t.to(device)
        gathered = [torch.zeros(max_size, dtype=t.dtype, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        out = torch.cat([g[:int(s.item())] for g, s in zip(gathered, sizes)]).cpu()
        return out

    all_preds = gather_tensor(local_preds).numpy()
    all_labels = gather_tensor(local_labels).numpy()
    all_probs = gather_tensor(local_probs).numpy()

    # Loss: sum and reduce
    loss_stats = torch.tensor([local_loss, local_total], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    loss_sum, total_sum = loss_stats.tolist()

    total = len(all_labels)
    acc = (all_preds == all_labels).sum() / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": loss_sum / max(total_sum, 1),
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "preds": all_preds,
        "labels": all_labels,
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
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# One fold
# ─────────────────────────────────────────────────────────────

def train_one_fold(fold_idx, train_slides, val_slide, config, device,
                   rank, world_size, epochs_override=None, total_folds=None):
    fold_num = fold_idx + 1
    fold_tag = f"[Fold {fold_num}/{total_folds}]" if total_folds else f"[Fold {fold_num}]"
    epochs = epochs_override if epochs_override is not None else config.epochs

    if is_main(rank):
        print(f"\n{'='*70}", flush=True)
        print(f"{fold_tag} Val=[{val_slide}], Train={train_slides}", flush=True)
        print(f"{'='*70}", flush=True)

    train_dataset = PatchDataset(
        config.output_dir, train_slides, config.slides,
        transform=get_train_transforms(config.input_size),
    )
    val_dataset = PatchDataset(
        config.output_dir, [val_slide], config.slides,
        transform=get_val_transforms(config.input_size),
    )

    if is_main(rank):
        print(f"  Train: {len(train_dataset)} patches {train_dataset.get_class_counts()}", flush=True)
        print(f"  Val:   {len(val_dataset)} patches {val_dataset.get_class_counts()}", flush=True)

    if len(val_dataset) == 0:
        if is_main(rank):
            print("  SKIP: No val patches", flush=True)
        return None

    # Train uses DistributedSampler (shuffle=True). Class imbalance handled
    # via class_weights in CrossEntropyLoss.
    if is_main(rank):
        print(f"  [setup] Creating DistributedSampler + DataLoader...", flush=True)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    if is_main(rank):
        print(f"  [setup] Creating model (backbone={config.backbone}, pretrained=True)...", flush=True)
    model = create_model(num_classes=config.num_classes, pretrained=True,
                         backbone=config.backbone,
                         head_type=getattr(config, "head_type", "linear"))
    freeze_early_layers(model, backbone=config.backbone)
    if is_main(rank):
        print(f"  [setup] model.to(device={device})...", flush=True)
    model = model.to(device)

    if world_size > 1:
        if is_main(rank):
            print(f"  [setup] Syncing ranks before DDP wrap (barrier)...", flush=True)
        barrier(device)
        if is_main(rank):
            print(f"  [setup] Wrapping model in DDP (this NCCL-broadcasts weights)...", flush=True)
        # FoundationClassifier wraps backbone with torch.no_grad() during Phase 1,
        # so backbone params don't enter the autograd graph → find_unused_parameters=False
        # is safe for all foundation backbones. ResNet18 also fine with False.
        find_unused = False
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=True,
        )
        if is_main(rank):
            print(f"  [setup] DDP wrap done (find_unused_parameters={find_unused}).", flush=True)

    # Class weights for loss
    train_labels = train_dataset.get_labels()
    class_counts = np.bincount(train_labels).astype(float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Linear-scaling LR rule (effective batch scales with world_size)
    effective_batch = config.batch_size * world_size
    scaled_lr = config.lr * (effective_batch / config.lr_scale_base)
    if is_main(rank):
        print(f"  World size {world_size}, per-GPU batch {config.batch_size}, "
              f"effective batch {effective_batch}, scaled lr {scaled_lr:.2e}",
              flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=scaled_lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_dtype = _amp_dtype_of(config)
    if is_main(rank):
        print(f"  AMP dtype: {amp_dtype}", flush=True)

    best_f1 = 0.0
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)

        if epoch == config.unfreeze_epoch + 1:
            if is_main(rank):
                print(f"  {fold_tag} Phase 2: unfreeze_all (backbone={config.backbone})",
                      flush=True)
            raw_model = model.module if hasattr(model, "module") else model
            unfreeze_all(raw_model, backbone=config.backbone)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=scaled_lr / 10, weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch + 1,
            )

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                        device, epoch, epochs, rank, fold_tag=fold_tag,
                                        amp_dtype=amp_dtype)
        val_metrics = validate(model, val_dataset, criterion, device,
                               config.batch_size, config.num_workers,
                               epoch, epochs, rank, world_size, fold_tag=fold_tag,
                               amp_dtype=amp_dtype)
        scheduler.step()

        if is_main(rank):
            lr_now = optimizer.param_groups[0]["lr"]
            auc_str = f"{val_metrics['auc']:.4f}" if not np.isnan(val_metrics['auc']) else "n/a"
            print(
                f"  {fold_tag} Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1']:.4f} AUC: {auc_str} | "
                f"LR: {lr_now:.2e} | best F1 {best_f1:.4f}",
                flush=True,
            )

        improved = val_metrics["f1"] > best_f1
        if improved:
            best_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
            if is_main(rank):
                ckpt_path = Path(config.checkpoint_dir) / f"best_model_{config.backbone}_fold{fold_num}.pth"
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "fold": fold_num,
                    "backbone": config.backbone,
                    "val_slide": val_slide,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "val_f1": best_f1,
                    "val_acc": val_metrics["acc"],
                }, ckpt_path)
                print(f"  {fold_tag} >> Saved best model (F1={best_f1:.4f}) at epoch {epoch}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if is_main(rank):
                    print(f"  {fold_tag} Early stopping at epoch {epoch}", flush=True)
                break

        barrier(device)

    return best_val_metrics


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    config = Config()
    # Allow env-var override for dry-runs / alternate patch dirs
    override = os.environ.get("PATCHES_DIR_OVERRIDE")
    if override:
        config.output_dir = override
    if int(os.environ.get("RANK", "0")) == 0:
        config.ensure_dirs()

    rank, world_size, local_rank, device = setup_distributed()
    if is_main(rank):
        print(f"DDP: rank={rank}, world_size={world_size}, device={device}")
        print(f"Patches dir: {config.output_dir}")

    # Dry-run flags
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    dry_epochs = int(os.environ.get("DRY_EPOCHS", "1"))
    dry_folds = int(os.environ.get("DRY_FOLDS", "1"))
    epochs_override = dry_epochs if dry_run else None

    slide_names = list(config.slides.keys())
    if dry_run:
        slide_names = slide_names[:dry_folds]
        if is_main(rank):
            print(f"DRY_RUN: {dry_folds} fold(s), {dry_epochs} epoch(s)")

    fold_results = []
    all_slide_names = list(config.slides.keys())

    for i, val_slide in enumerate(slide_names):
        train_slides = [s for s in all_slide_names if s != val_slide]
        result = train_one_fold(i, train_slides, val_slide, config, device,
                                rank, world_size, epochs_override=epochs_override,
                                total_folds=len(slide_names))
        if result is not None and is_main(rank):
            val_class = config.slides[val_slide]["class"]
            fold_results.append({
                "fold": i + 1,
                "val_slide": val_slide,
                "val_class": val_class,
                "n_val": int(len(result["labels"])),
                "acc": result["acc"],
                "f1_macro": result["f1"],   # per-fold macro (existing)
                "auc": result["auc"],
                "labels": result["labels"],  # keep for pooling
                "preds": result["preds"],
            })
            suffix = "_G" if config.slides[val_slide]["class"] == "gland" else "_S"
            save_confusion_matrix(
                result["labels"], result["preds"], config.class_names,
                config.viz_dir_for("Matrix_Viz") / f"cm_fold{i+1}_{val_slide}{suffix}_{config.backbone}.png",
            )
        barrier(device)

    if is_main(rank) and fold_results:
        # ── Per-fold table ──
        print(f"\n{'='*78}")
        print("LOSO Cross-Validation — per-fold")
        print(f"{'='*78}")
        print(f"{'Fold':<5} {'Val Slide':<18} {'Class':<10} {'N_val':>8} "
              f"{'Acc':>8} {'F1(macro)':>10} {'AUC':>8}")
        print("-" * 78)
        for r in fold_results:
            auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "n/a"
            print(f"{r['fold']:<5} {r['val_slide']:<18} {r['val_class']:<10} "
                  f"{r['n_val']:>8d} {r['acc']:>8.4f} "
                  f"{r['f1_macro']:>10.4f} {auc_str:>8}")
        accs = [r["acc"] for r in fold_results]
        f1s  = [r["f1_macro"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results if not np.isnan(r["auc"])]
        print("-" * 78)
        print(f"{'Mean (fold-avg)':<33}  {'':>8} "
              f"{np.mean(accs):>8.4f} {np.mean(f1s):>10.4f} "
              f"{(np.mean(aucs) if aucs else float('nan')):>8.4f}")
        print(f"{'Std  (fold-avg)':<33}  {'':>8} "
              f"{np.std(accs):>8.4f} {np.std(f1s):>10.4f} "
              f"{(np.std(aucs) if aucs else float('nan')):>8.4f}")

        # ── Pooled (patch-level) metrics across all folds ──
        pooled_labels = np.concatenate([r["labels"] for r in fold_results])
        pooled_preds  = np.concatenate([r["preds"]  for r in fold_results])
        n_total = len(pooled_labels)

        acc_micro   = (pooled_labels == pooled_preds).mean()
        f1_micro    = f1_score(pooled_labels, pooled_preds, average="micro",    zero_division=0)
        f1_macro    = f1_score(pooled_labels, pooled_preds, average="macro",    zero_division=0)
        f1_weighted = f1_score(pooled_labels, pooled_preds, average="weighted", zero_division=0)
        f1_per_cls  = f1_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)
        prec_per_cls = precision_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)
        rec_per_cls  = recall_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)

        print(f"\n{'='*78}")
        print(f"Pooled across all val patches (N={n_total:,})")
        print(f"{'='*78}")
        print(f"  Acc  (micro)     : {acc_micro:.4f}")
        print(f"  F1   micro       : {f1_micro:.4f}")
        print(f"  F1   macro       : {f1_macro:.4f}")
        print(f"  F1   weighted    : {f1_weighted:.4f}")
        print(f"  F1   per-class   : gland(0)={f1_per_cls[0]:.4f}  non-gland(1)={f1_per_cls[1]:.4f}")
        print(f"  Prec per-class   : gland(0)={prec_per_cls[0]:.4f}  non-gland(1)={prec_per_cls[1]:.4f}")
        print(f"  Rec  per-class   : gland(0)={rec_per_cls[0]:.4f}  non-gland(1)={rec_per_cls[1]:.4f}")

        print(f"\n  Classification report:")
        print(classification_report(
            pooled_labels, pooled_preds,
            labels=[0, 1], target_names=config.class_names,
            digits=4, zero_division=0,
        ))

        # Pooled confusion matrix
        bb = config.backbone
        pooled_cm_path = config.viz_dir_for("Matrix_Viz") / f"cm_pooled_all_folds_{bb}.png"
        save_confusion_matrix(
            pooled_labels, pooled_preds, config.class_names,
            pooled_cm_path,
        )
        print(f"  Pooled CM saved → {pooled_cm_path}")

        # ── CSV outputs ──
        csv_dir = Path(config.log_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)

        per_fold_rows = [{
            "fold": r["fold"], "val_slide": r["val_slide"], "val_class": r["val_class"],
            "n_val": r["n_val"], "acc": r["acc"],
            "f1_macro": r["f1_macro"], "auc": r["auc"],
            "backbone": bb,
        } for r in fold_results]
        per_fold_df = pd.DataFrame(per_fold_rows)
        per_fold_df.to_csv(csv_dir / f"loso_per_fold_{bb}.csv", index=False)

        summary = {
            "backbone": bb,
            "n_folds": len(fold_results),
            "n_val_patches_total": n_total,
            "acc_micro": acc_micro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_gland": f1_per_cls[0], "f1_non_gland": f1_per_cls[1],
            "precision_gland": prec_per_cls[0], "precision_non_gland": prec_per_cls[1],
            "recall_gland": rec_per_cls[0], "recall_non_gland": rec_per_cls[1],
            "acc_fold_mean": np.mean(accs), "acc_fold_std": np.std(accs),
            "f1_fold_mean": np.mean(f1s),   "f1_fold_std": np.std(f1s),
            "auc_fold_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_fold_std":  float(np.std(aucs))  if aucs else float("nan"),
        }
        pd.DataFrame([summary]).to_csv(csv_dir / f"loso_summary_{bb}.csv", index=False)

        print(f"\n  CSV saved:")
        print(f"    per-fold : {csv_dir}/loso_per_fold_{bb}.csv")
        print(f"    summary  : {csv_dir}/loso_summary_{bb}.csv")

    cleanup_distributed()


if __name__ == "__main__":
    main()
