"""
Compute confusion matrix for a model on a slide using per_patch_predictions.csv
and annotation.npz (counter ≥ 2 = non-gland convention, matches GT used elsewhere).

Output: PNG confusion matrix in /app/Gland_Seg/results/_hires_overlays/

Usage:
    python make_confusion_matrix.py \
      --slide S14-2289-1-6 --run_tag _byext_224_20x_raw_ce \
      --model hibou-l --label "Hibou-L Raw+CE"
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from config import Config

PATCH_SIZE_L0 = 448  # config.patch_size for 224_20x


def build_counter(polys, H, W, scale):
    """Build counter map at thumb resolution (counter ≥ 2 = nested non-gland)."""
    counter = np.zeros((H, W), dtype=np.int16)
    for poly in polys:
        poly = np.asarray(poly, dtype=np.float64)
        pts = (poly / scale).round().astype(np.int32)
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        counter += m.astype(np.int16)
    return counter


def plot_cm(cm, slide, label, n_eval, acc, f1_m, f1_g, f1_ng, prec_ng, rec_ng, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    cls = ["gland", "non-gland"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(cls, fontsize=12)
    ax.set_yticklabels(cls, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Ground truth", fontsize=13, fontweight="bold")

    title = (f"{slide}   ·   {label}\n"
             f"n={n_eval}   Acc={acc:.3f}   F1_macro={f1_m:.3f}\n"
             f"F1 gland={f1_g:.3f}   F1 non-gland={f1_ng:.3f}   "
             f"Prec_ng={prec_ng:.3f}   Rec_ng={rec_ng:.3f}")
    ax.set_title(title, fontsize=12, pad=10)

    # Cell text: count + row-normalized %
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm / row_sum * 100, 0)
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            p = cm_pct[i, j]
            color = "white" if v > cm.max() / 2 else "black"
            ax.text(j, i, f"{v}\n({p:.1f}%)", ha="center", va="center",
                    color=color, fontsize=16, fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="p_gland threshold (default 0.5)")
    args = ap.parse_args()

    cfg = Config()
    results_dir = Path(cfg.base_dir) / "results" / f"{args.slide}{args.run_tag}"
    pred_csv = results_dir / "per_patch_predictions.csv"
    ann_path = results_dir / "annotation.npz"
    meta_path = results_dir / "slide_meta.npy"

    for p in [pred_csv, ann_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_csv(pred_csv)
    pred_col = f"p_gland_{args.model}"
    if pred_col not in df.columns:
        raise KeyError(f"{pred_col} not in {pred_csv}. Cols: {list(df.columns)}")

    meta = np.load(meta_path, allow_pickle=True).item()
    ann = np.load(ann_path, allow_pickle=True)
    pos = [np.asarray(p, dtype=np.float64) for p in ann["positive"]]
    neg = [np.asarray(p, dtype=np.float64) for p in ann["negative"]]
    scale = float(meta["scale"])
    thumb_W, thumb_H = int(meta["thumb_W"]), int(meta["thumb_H"])

    # Build counter on thumb
    all_polys = pos + neg
    counter = build_counter(all_polys, thumb_H, thumb_W, scale)
    mask_inROI = counter >= 1
    mask_nested = counter >= 2
    if len(neg) > 0:
        for poly in neg:
            poly = np.asarray(poly, dtype=np.float64)
            pts = (poly / scale).round().astype(np.int32)
            m = np.zeros((thumb_H, thumb_W), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            mask_nested = mask_nested | m.astype(bool)

    # Patch center on thumb (patch top-left at L0 (x,y), center at (x+224, y+224))
    cx = (df["x"].values + PATCH_SIZE_L0 / 2) / scale
    cy = (df["y"].values + PATCH_SIZE_L0 / 2) / scale
    cx_i = np.clip(cx.astype(int), 0, thumb_W - 1)
    cy_i = np.clip(cy.astype(int), 0, thumb_H - 1)

    in_roi = mask_inROI[cy_i, cx_i]
    in_ng = mask_nested[cy_i, cx_i]

    # GT label: 1 = non-gland (in_ng), 0 = gland (in_roi & ~in_ng). Skip if not in_roi.
    gt = np.full(len(df), -1, dtype=int)
    gt[in_roi & ~in_ng] = 0
    gt[in_ng] = 1
    valid = gt != -1

    pred = (df[pred_col].values < args.threshold).astype(int)  # 1=non-gland

    gt_e = gt[valid]
    pred_e = pred[valid]
    n_eval = len(gt_e)
    n_g = int((gt_e == 0).sum())
    n_ng = int((gt_e == 1).sum())
    print(f"[gt] in_roi={int(in_roi.sum())}  in_ng={int(in_ng.sum())}  "
          f"eval={n_eval} (gland={n_g}, non-gland={n_ng}, skipped={int((~valid).sum())})")

    acc = accuracy_score(gt_e, pred_e)
    f1_m = f1_score(gt_e, pred_e, average="macro", zero_division=0)
    f1_g = f1_score(gt_e, pred_e, pos_label=0, zero_division=0)
    f1_ng = f1_score(gt_e, pred_e, pos_label=1, zero_division=0)
    prec_ng = precision_score(gt_e, pred_e, pos_label=1, zero_division=0)
    rec_ng = recall_score(gt_e, pred_e, pos_label=1, zero_division=0)
    cm = confusion_matrix(gt_e, pred_e, labels=[0, 1])

    print(f"Acc={acc:.4f}  F1_macro={f1_m:.4f}  F1_g={f1_g:.4f}  F1_ng={f1_ng:.4f}")
    print(f"Prec_ng={prec_ng:.4f}  Rec_ng={rec_ng:.4f}")
    print(f"CM (rows=GT, cols=Pred):\n{cm}")

    if args.out is None:
        out_dir = Path("/app/Gland_Seg/results/_hires_overlays")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = args.label.replace(" ", "_").replace("+", "p").replace("/", "_")
        args.out = out_dir / f"CM__{args.slide}__{args.model}__{safe}.png"
    plot_cm(cm, args.slide, args.label, n_eval, acc, f1_m, f1_g, f1_ng,
            prec_ng, rec_ng, args.out)


if __name__ == "__main__":
    main()
