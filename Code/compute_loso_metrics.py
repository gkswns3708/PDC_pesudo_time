"""
Post-hoc LOSO-CV metric recomputation from saved checkpoints.

For each best_model_fold{N}.pth, runs inference on the corresponding
val slide (stored in the checkpoint) and:
  - Pools predictions across all folds
  - Computes micro/macro/weighted/per-class F1, precision, recall
  - Saves CSV (per-fold + overall summary) to Gland_Seg/logs/
  - Saves pooled confusion matrix to Gland_Seg/Viz/

Use when training finished but console-only summary was produced
(e.g., current running train_cv.py was started before CSV code was added).

Usage:
    python compute_loso_metrics.py           # default: use /app/Gland_Seg/checkpoints
    python compute_loso_metrics.py /custom/ckpt_dir
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from dataset import PatchDataset, get_val_transforms
from model import create_model


def save_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


@torch.no_grad()
def infer_slide(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_preds, all_labels = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    config = Config()
    ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(config.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Patches dir:    {config.output_dir}")
    print(f"Device:         {device}")

    # Prefer backbone-tagged checkpoints for the current config.backbone;
    # fall back to legacy `best_model_fold*.pth` (ResNet18) if none found.
    ckpt_paths = sorted(ckpt_dir.glob(f"best_model_{config.backbone}_fold*.pth"))
    if not ckpt_paths:
        ckpt_paths = sorted(ckpt_dir.glob("best_model_fold*.pth"))
    if not ckpt_paths:
        print(f"No checkpoint found in {ckpt_dir} (looked for best_model_{config.backbone}_fold*.pth and legacy best_model_fold*.pth)")
        sys.exit(1)
    print(f"Found {len(ckpt_paths)} fold checkpoint(s) for backbone={config.backbone}.")

    fold_rows, pooled_labels, pooled_preds = [], [], []

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        fold = ckpt["fold"]; val_slide = ckpt["val_slide"]
        val_class = config.slides[val_slide]["class"]
        suffix = "_G" if val_class == "gland" else "_S"

        val_dataset = PatchDataset(
            config.output_dir, [val_slide], config.slides,
            transform=get_val_transforms(config.input_size),
        )
        print(f"\n[Fold {fold}] val={val_slide} ({val_class}), N={len(val_dataset)} patches")
        if len(val_dataset) == 0:
            print("  skip (empty val)")
            continue

        # Prefer backbone recorded in checkpoint; fall back to config.backbone
        ckpt_backbone = ckpt.get("backbone", config.backbone)
        model = create_model(num_classes=config.num_classes, pretrained=False,
                             backbone=ckpt_backbone,
                             head_type=getattr(config, "head_type", "linear"))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)

        preds, labels = infer_slide(
            model, val_dataset, device,
            batch_size=config.batch_size, num_workers=config.num_workers,
        )
        acc = (preds == labels).mean()
        f1m = f1_score(labels, preds, average="macro", zero_division=0)
        try:
            auc = float("nan")  # single-class val → AUC undefined; skip
            if len(np.unique(labels)) > 1:
                # (Not applicable here: per-fold val is single-class.)
                auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = float("nan")

        print(f"  acc={acc:.4f}  f1_macro={f1m:.4f}  saved_val_f1={ckpt.get('val_f1', float('nan')):.4f}")

        # Per-fold confusion matrix (with class + backbone suffix)
        # NOTE: ckpt_backbone determines the viz subdir via config.viz_dir_for()
        # but config.backbone may differ; temporarily set it for routing
        prev_backbone = config.backbone
        config.backbone = ckpt_backbone
        save_confusion_matrix(
            labels, preds, config.class_names,
            config.viz_dir_for("Matrix_Viz") / f"cm_fold{fold}_{val_slide}{suffix}_{ckpt_backbone}.png",
        )
        config.backbone = prev_backbone

        fold_rows.append({
            "fold": fold, "val_slide": val_slide, "val_class": val_class,
            "n_val": int(len(labels)), "acc": acc,
            "f1_macro": f1m, "auc": auc,
        })
        pooled_labels.append(labels); pooled_preds.append(preds)

    if not fold_rows:
        print("No folds processed.")
        return

    pooled_labels = np.concatenate(pooled_labels)
    pooled_preds  = np.concatenate(pooled_preds)
    n_total = len(pooled_labels)

    # ── Per-fold table ──
    print(f"\n{'='*78}")
    print("LOSO Cross-Validation — per-fold (re-computed from checkpoints)")
    print(f"{'='*78}")
    print(f"{'Fold':<5} {'Val Slide':<18} {'Class':<10} {'N_val':>8} "
          f"{'Acc':>8} {'F1(macro)':>10} {'AUC':>8}")
    print("-" * 78)
    for r in fold_rows:
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r['auc']) else "n/a"
        print(f"{r['fold']:<5} {r['val_slide']:<18} {r['val_class']:<10} "
              f"{r['n_val']:>8d} {r['acc']:>8.4f} {r['f1_macro']:>10.4f} {auc_str:>8}")

    accs = [r["acc"] for r in fold_rows]
    f1s  = [r["f1_macro"] for r in fold_rows]
    aucs = [r["auc"] for r in fold_rows if not np.isnan(r["auc"])]
    print("-" * 78)
    print(f"{'Mean (fold-avg)':<33}  {'':>8} "
          f"{np.mean(accs):>8.4f} {np.mean(f1s):>10.4f} "
          f"{(np.mean(aucs) if aucs else float('nan')):>8.4f}")
    print(f"{'Std  (fold-avg)':<33}  {'':>8} "
          f"{np.std(accs):>8.4f} {np.std(f1s):>10.4f} "
          f"{(np.std(aucs) if aucs else float('nan')):>8.4f}")

    # ── Pooled metrics ──
    acc_micro   = (pooled_labels == pooled_preds).mean()
    f1_micro    = f1_score(pooled_labels, pooled_preds, average="micro",    zero_division=0)
    f1_macro    = f1_score(pooled_labels, pooled_preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(pooled_labels, pooled_preds, average="weighted", zero_division=0)
    f1_per_cls  = f1_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)
    prec_per    = precision_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)
    rec_per     = recall_score(pooled_labels, pooled_preds, average=None, labels=[0, 1], zero_division=0)

    print(f"\n{'='*78}")
    print(f"Pooled across all val patches (N={n_total:,})")
    print(f"{'='*78}")
    print(f"  Acc  (micro)     : {acc_micro:.4f}")
    print(f"  F1   micro       : {f1_micro:.4f}")
    print(f"  F1   macro       : {f1_macro:.4f}")
    print(f"  F1   weighted    : {f1_weighted:.4f}")
    print(f"  F1   per-class   : gland(0)={f1_per_cls[0]:.4f}  non-gland(1)={f1_per_cls[1]:.4f}")
    print(f"  Prec per-class   : gland(0)={prec_per[0]:.4f}  non-gland(1)={prec_per[1]:.4f}")
    print(f"  Rec  per-class   : gland(0)={rec_per[0]:.4f}  non-gland(1)={rec_per[1]:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(
        pooled_labels, pooled_preds,
        labels=[0, 1], target_names=config.class_names,
        digits=4, zero_division=0,
    ))

    # Pooled CM
    bb = config.backbone
    pooled_cm_path = config.viz_dir_for("Matrix_Viz") / f"cm_pooled_all_folds_{bb}.png"
    save_confusion_matrix(
        pooled_labels, pooled_preds, config.class_names,
        pooled_cm_path,
    )
    print(f"  Pooled CM saved → {pooled_cm_path}")

    # CSV (backbone-tagged)
    csv_dir = Path(config.log_dir); csv_dir.mkdir(parents=True, exist_ok=True)
    per_fold_rows = [{**r, "backbone": bb} for r in fold_rows]
    pd.DataFrame(per_fold_rows).to_csv(csv_dir / f"loso_per_fold_{bb}.csv", index=False)
    summary = {
        "backbone": bb,
        "n_folds": len(fold_rows), "n_val_patches_total": n_total,
        "acc_micro": acc_micro,
        "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "f1_gland": f1_per_cls[0], "f1_non_gland": f1_per_cls[1],
        "precision_gland": prec_per[0], "precision_non_gland": prec_per[1],
        "recall_gland": rec_per[0], "recall_non_gland": rec_per[1],
        "acc_fold_mean": float(np.mean(accs)), "acc_fold_std": float(np.std(accs)),
        "f1_fold_mean":  float(np.mean(f1s)),  "f1_fold_std":  float(np.std(f1s)),
    }
    pd.DataFrame([summary]).to_csv(csv_dir / f"loso_summary_{bb}.csv", index=False)
    print(f"  CSV saved:")
    print(f"    per-fold : {csv_dir}/loso_per_fold_{bb}.csv")
    print(f"    summary  : {csv_dir}/loso_summary_{bb}.csv")


if __name__ == "__main__":
    main()
