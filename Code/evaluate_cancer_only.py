"""
Evaluate Virchow2 (or any ensemble member) on Cancer-only patches within ROI.

Reads:
  results/<slide>/per_patch_predictions.csv  — our Virchow2/UNI2/Phikon-v2 probs
  results/<slide>/kather_per_patch.csv       — Kather 9-class probs (ROI-filtered)
  Data/S14/Annotation/<slide>.xml             — GT (ROI boxes + non-gland sub-polygons)

For ROI ∩ Cancer-mask patches:
  GT label   : in non-gland polygon → 1 (non-gland), else 0 (gland)
  Prediction : virchow2 P(gland) >= threshold → 0 (gland), else 1 (non-gland)

Output:
  results/<slide>/evaluation_cancer_only.csv     — per-patch table
  results/<slide>/evaluation_metrics.json        — summary metrics
  results/<slide>/confusion_matrix_cancer_only.png

Usage:
  python evaluate_cancer_only.py S14-2289-1-6 [--cancer-threshold 0.5] \
    [--gland-threshold 0.5] [--virchow2-col p_gland_virchow2]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib.path import Path as MplPath
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

from config import Config


def parse_polygons(xml_path):
    """Return (rois_big_boxes, sub_polygons). Same heuristic as elsewhere."""
    tree = etree.parse(xml_path)
    polys = []
    for r in tree.getroot().findall(".//Region"):
        verts = [(float(v.get("X")), float(v.get("Y"))) for v in r.findall(".//Vertex")]
        if len(verts) < 3:
            continue
        polys.append(np.array(verts, dtype=np.float64))
    areas = np.array([
        0.5 * abs(np.dot(p[:,0], np.roll(p[:,1], -1)) - np.dot(p[:,1], np.roll(p[:,0], -1)))
        for p in polys
    ])
    med = np.median(areas) if len(areas) else 0.0
    rois, subs = [], []
    for p, a in zip(polys, areas):
        if a > med * 10.0 and len(p) < 30:
            rois.append(p)
        else:
            subs.append(p)
    return rois, subs


def points_in_polys(points_xy, polys):
    """For each point, True if inside any polygon."""
    if not polys:
        return np.zeros(len(points_xy), dtype=bool)
    in_any = np.zeros(len(points_xy), dtype=bool)
    for poly in polys:
        in_any |= MplPath(poly).contains_points(points_xy)
    return in_any


def plot_cm(cm, class_names, save_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    if title: ax.set_title(title, fontsize=12)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--cancer-threshold", type=float, default=0.5,
                        help="(P_TUM + P_STR) >= threshold → cancer")
    parser.add_argument("--gland-threshold", type=float, default=0.5,
                        help="P(gland) >= threshold → gland")
    parser.add_argument("--virchow2-col", type=str, default="p_gland_virchow2",
                        help="column in per_patch_predictions.csv to use as prediction")
    parser.add_argument("--xml", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.xml:
        xml_path = args.xml
    elif args.slide in getattr(config, "external_test_slides", {}):
        info = config.external_test_slides[args.slide]
        xml_path = str(Path(config.xml_dir) / info["xml"])
    else:
        raise ValueError(f"{args.slide} not configured; pass --xml")

    out_dir = Path(config.base_dir) / "results" / args.slide
    df_pred = pd.read_csv(out_dir / "per_patch_predictions.csv")
    df_kather = pd.read_csv(out_dir / "kather_per_patch.csv")
    print(f"Loaded {len(df_pred)} per-patch predictions, {len(df_kather)} Kather predictions")

    # ── Merge: Kather is ROI-filtered subset; merge on (x,y) ──
    merged = df_kather.merge(df_pred, on=["x", "y"], how="inner", suffixes=("", "_pred"))
    print(f"Merged (ROI patches with both Kather + Virchow2): {len(merged)}")

    if args.virchow2_col not in merged.columns:
        raise KeyError(f"{args.virchow2_col} not in merged DataFrame. Available: "
                       f"{[c for c in merged.columns if 'p_gland' in c]}")

    # ── GT labels (from non-gland sub-polygons) ──
    rois, sub_polys = parse_polygons(xml_path)
    print(f"Annotation: {len(rois)} ROI box(es) + {len(sub_polys)} non-gland sub-polygons")

    # patch centers
    cx = merged["x"].values + config.patch_size / 2.0
    cy = merged["y"].values + config.patch_size / 2.0
    pts = np.column_stack([cx, cy])
    in_nongland = points_in_polys(pts, sub_polys)
    gt_label = in_nongland.astype(int)   # 1 = non-gland, 0 = gland

    # ── Cancer mask filter ──
    cancer_mask = (merged["p_Cancer"].values >= args.cancer_threshold)

    # ── Prediction ──
    p_gland = merged[args.virchow2_col].values
    pred_label = (p_gland < args.gland_threshold).astype(int)   # 1 = non-gland

    # Add evaluation columns
    merged["gt_label"] = gt_label
    merged["pred_label"] = pred_label
    merged["cancer_mask"] = cancer_mask
    merged["gt_class"] = np.where(gt_label == 1, "non-gland", "gland")
    merged["pred_class"] = np.where(pred_label == 1, "non-gland", "gland")
    merged["correct"] = (gt_label == pred_label) & cancer_mask
    csv_out = out_dir / "evaluation_cancer_only.csv"
    merged.to_csv(csv_out, index=False)
    print(f"\nSaved per-patch eval CSV: {csv_out}")

    # ── Compute metrics on ROI ∩ Cancer subset ──
    eval_idx = np.where(cancer_mask)[0]
    n_eval = len(eval_idx)
    if n_eval == 0:
        raise RuntimeError("No patches pass cancer mask. Lower --cancer-threshold.")
    gt_e = gt_label[eval_idx]
    pred_e = pred_label[eval_idx]
    p_e = p_gland[eval_idx]

    print(f"\n=== Metrics (ROI ∩ Cancer-mask, n={n_eval}) ===")
    acc = accuracy_score(gt_e, pred_e)
    f1_macro = f1_score(gt_e, pred_e, average="macro", zero_division=0)
    f1_per = f1_score(gt_e, pred_e, average=None, labels=[0,1], zero_division=0)
    prec_per = precision_score(gt_e, pred_e, average=None, labels=[0,1], zero_division=0)
    rec_per  = recall_score(gt_e, pred_e, average=None, labels=[0,1], zero_division=0)
    cm = confusion_matrix(gt_e, pred_e, labels=[0,1])

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1_macro:.4f}")
    print(f"F1   per-class : gland={f1_per[0]:.4f}  non-gland={f1_per[1]:.4f}")
    print(f"Prec per-class : gland={prec_per[0]:.4f}  non-gland={prec_per[1]:.4f}")
    print(f"Rec  per-class : gland={rec_per[0]:.4f}  non-gland={rec_per[1]:.4f}")
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    print(f"              gland  non-gland")
    print(f"  gland     {cm[0,0]:>6} {cm[0,1]:>9}")
    print(f"  non-gland {cm[1,0]:>6} {cm[1,1]:>9}")
    print(f"\n{classification_report(gt_e, pred_e, labels=[0,1], target_names=['gland','non-gland'], digits=4, zero_division=0)}")

    # ── Also compute metrics WITHOUT cancer-mask for comparison ──
    acc_all = accuracy_score(gt_label, pred_label)
    f1_macro_all = f1_score(gt_label, pred_label, average="macro", zero_division=0)
    print(f"\n[Reference] Without cancer-mask filter (all ROI patches, n={len(gt_label)}):")
    print(f"  Accuracy: {acc_all:.4f}  F1 macro: {f1_macro_all:.4f}")

    # ── Save JSON summary ──
    summary = {
        "slide": args.slide,
        "n_patches_in_roi": int(len(gt_label)),
        "n_patches_cancer_eval": int(n_eval),
        "n_patches_excluded_by_cancer_mask": int(len(gt_label) - n_eval),
        "cancer_threshold": args.cancer_threshold,
        "gland_threshold": args.gland_threshold,
        "virchow2_column": args.virchow2_col,
        # Cancer-only metrics (main)
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_gland": float(f1_per[0]),
        "f1_non_gland": float(f1_per[1]),
        "precision_gland": float(prec_per[0]),
        "precision_non_gland": float(prec_per[1]),
        "recall_gland": float(rec_per[0]),
        "recall_non_gland": float(rec_per[1]),
        "confusion_matrix_cancer_only": cm.tolist(),
        # Reference (no cancer-mask)
        "accuracy_no_mask": float(acc_all),
        "f1_macro_no_mask": float(f1_macro_all),
        # GT distribution
        "n_gt_gland_in_eval": int((gt_e == 0).sum()),
        "n_gt_non_gland_in_eval": int((gt_e == 1).sum()),
    }
    json_path = out_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary: {json_path}")

    # ── Save confusion matrix PNG ──
    cm_path = out_dir / "confusion_matrix_cancer_only.png"
    plot_cm(cm, ["gland", "non-gland"], cm_path,
            title=f"{args.slide} — ROI ∩ Cancer-mask (n={n_eval})\n"
                  f"Acc={acc:.3f}, F1_macro={f1_macro:.3f}")
    print(f"Saved CM PNG: {cm_path}")


if __name__ == "__main__":
    main()
