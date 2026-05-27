"""
Score SPIDER per-patch predictions vs the same parity-rule GT used by
compute_gt_metrics.py (positive=ROI, inner polygons=non-gland).

Three F1 variants are reported:
  1. spider_hard      — top1 class. Non-Adeno classes counted as misses
                        unless we coerce them; here we collapse top1 with:
                          "Adenocarcinoma high grade" -> non-gland (1)
                          "Adenocarcinoma low grade"  -> gland     (0)
                          any other class             -> gland     (0)   (most non-tumor regions read as gland in our parity GT)
  2. spider_softrenorm — restrict to the two adenoCa classes and pick the
                        larger. (Tells us how SPIDER discriminates *given*
                        tumor.) Patches whose top1 is neither adenoCa get
                        skipped from this version.
  3. spider_p_high    — threshold p_high_grade > 0.5 → non-gland.

Usage:
    python compute_gt_metrics_spider.py S14-2289-1-6
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from config import Config


def build_gt_counter(polys, H, W, scale):
    counter = np.zeros((H, W), dtype=np.int16)
    for poly in polys:
        pts = (poly / scale).round().astype(np.int32)
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        counter += m.astype(np.int16)
    return counter


def score(gt, pred, name, n_eval_skip=0):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    acc = accuracy_score(gt, pred)
    prec_per = precision_score(gt, pred, labels=[0, 1], average=None, zero_division=0)
    rec_per = recall_score(gt, pred, labels=[0, 1], average=None, zero_division=0)
    f1_per = f1_score(gt, pred, labels=[0, 1], average=None, zero_division=0)
    f1_macro = f1_score(gt, pred, average="macro", zero_division=0)
    return {
        "source": name,
        "n_eval": len(gt),
        "tp_gland": int(cm[0, 0]), "fp_gland": int(cm[1, 0]), "fn_gland": int(cm[0, 1]),
        "tp_nongland": int(cm[1, 1]), "fp_nongland": int(cm[0, 1]), "fn_nongland": int(cm[1, 0]),
        "accuracy": float(acc),
        "precision_gland": float(prec_per[0]), "recall_gland": float(rec_per[0]), "f1_gland": float(f1_per[0]),
        "precision_nongland": float(prec_per[1]), "recall_nongland": float(rec_per[1]), "f1_nongland": float(f1_per[1]),
        "f1_macro": float(f1_macro),
        "n_skipped": int(n_eval_skip),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_gt_metrics_spider.py <slide>")
        sys.exit(1)
    slide = sys.argv[1]
    config = Config()
    base = Path(config.base_dir) / "results" / slide
    if not base.exists():
        sys.exit(f"results dir not found: {base}")

    pred_csv = base / "per_patch_predictions_spider.csv"
    if not pred_csv.exists():
        sys.exit(f"missing {pred_csv} — run infer_spider_on_eval.py first")
    pred = pd.read_csv(pred_csv)
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    ann = np.load(base / "annotation.npz", allow_pickle=True)
    polys = list(ann["positive"]) + list(ann["negative"])
    thumb_W, thumb_H = meta["thumb_W"], meta["thumb_H"]
    scale = meta["scale"]
    patch_size = meta["patch_size"]

    print(f"Slide:      {slide}")
    print(f"Patches:    {len(pred):,}")
    print(f"Polygons:   {len(polys)} (positive {len(ann['positive'])}, negative {len(ann['negative'])})")

    counter = build_gt_counter(polys, thumb_H, thumb_W, scale)
    cx_thumb = ((pred["x"].values + patch_size / 2) / scale).astype(int)
    cy_thumb = ((pred["y"].values + patch_size / 2) / scale).astype(int)
    cx_thumb = np.clip(cx_thumb, 0, thumb_W - 1)
    cy_thumb = np.clip(cy_thumb, 0, thumb_H - 1)
    n_in = counter[cy_thumb, cx_thumb]

    gt = np.where(n_in == 0, -1, np.where(n_in == 1, 0, 1))
    eval_mask = gt >= 0
    gt_eval = gt[eval_mask]
    print(f"GT counts:  gland {(gt_eval==0).sum():,}  non-gland {(gt_eval==1).sum():,}  (no-GT skipped {(~eval_mask).sum():,})")

    # build prediction variants
    rows = []

    if "pred_binary" in pred.columns:
        pred_hard = pred["pred_binary"].values
        rows.append(score(gt_eval, pred_hard[eval_mask], "spider_hard"))

    if "pred_binary_softrenorm" in pred.columns:
        # only patches whose top1 is an adenocarcinoma class
        top1 = pred["top1_class"].values
        is_adeno = np.isin(top1, ["Adenocarcinoma high grade", "Adenocarcinoma low grade"])
        sub_mask = eval_mask & is_adeno
        rows.append(score(gt[sub_mask],
                          pred["pred_binary_softrenorm"].values[sub_mask],
                          "spider_softrenorm_adeno_only",
                          n_eval_skip=int((eval_mask & ~is_adeno).sum())))

    if "p_high_grade" in pred.columns:
        p_high = pred["p_high_grade"].values
        for thr in (0.30, 0.50, 0.70):
            pred_thr = (p_high > thr).astype(int)
            rows.append(score(gt_eval, pred_thr[eval_mask], f"spider_p_high>{thr:.2f}"))

    out_df = pd.DataFrame(rows)
    out_path = base / "metrics_vs_gt_spider.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(out_df[["source", "n_eval", "accuracy", "f1_gland", "f1_nongland", "f1_macro"]].to_string(index=False))

    # comparison table vs existing baselines
    base_csv = base / "metrics_vs_gt.csv"
    if base_csv.exists():
        base_df = pd.read_csv(base_csv)
        cmb = pd.concat([base_df[["source", "n_eval", "accuracy", "f1_gland", "f1_nongland", "f1_macro"]],
                         out_df[["source", "n_eval", "accuracy", "f1_gland", "f1_nongland", "f1_macro"]]],
                        ignore_index=True)
        print("\n=== Combined with phikon-v2 / ensemble baselines ===")
        print(cmb.to_string(index=False))


if __name__ == "__main__":
    main()
