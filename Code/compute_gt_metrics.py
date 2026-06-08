"""
Compute per-source classification metrics (acc / precision / recall / F1) on
S14-2289-1-6 using the professor's annotation as ground truth.

GT derivation (per professor's instruction in KakaoTalk):
  - Large rectangles = ROI; the small polygons inside them = non-gland cancer.
  - Inside ROI but outside small polygons = gland (mostly + some normal).
We use a simple parity rule on a thumb-resolution mask:
  patch center ∉ any polygon → no GT (outside ROI; skip)
  patch center ∈ exactly 1 polygon → GT = gland
  patch center ∈ ≥ 2 polygons    → GT = non-gland (inside an inner polygon)

Outputs:
  /app/Gland_Seg/results/<slide>/metrics_vs_gt.csv
  /app/Gland_Seg/results/<slide>/prediction_summary.md  (appended GT section)

Usage:
    python compute_gt_metrics.py S14-2289-1-6
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

from config import Config


SOURCES = ("virchow2", "uni2", "phikon-v2", "ensemble_mean_prob", "hardvote")
CLASS_NAMES = ("gland", "non-gland")  # 0, 1


def predicted_labels(df, source):
    """Return Nx int8: 0=gland, 1=non-gland for the given source."""
    if source == "hardvote":
        return (df["pred_hardvote"].values == "non-gland").astype(np.int8)
    if source == "ensemble_mean_prob":
        return (df["p_gland_ensemble"].values < 0.5).astype(np.int8)
    return (df[f"p_gland_{source}"].values < 0.5).astype(np.int8)


def build_gt_counter(polys, H, W, scale):
    """Accumulate +1 inside each polygon's filled area at thumb resolution."""
    counter = np.zeros((H, W), dtype=np.int16)
    for poly in polys:
        pts = (poly / scale).round().astype(np.int32)
        # cv2.fillPoly accumulates onto a same-shape uint8 mask; we add into counter
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        counter += m.astype(np.int16)
    return counter


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_gt_metrics.py <slide>")
        sys.exit(1)
    slide = sys.argv[1]
    config = Config()
    # Prefer tagged results dir (e.g. results/S14-2289-1-6_224_20x), fall back to untagged.
    base_tagged = Path(config.base_dir) / "results" / f"{slide}{config.run_tag}"
    base_plain = Path(config.base_dir) / "results" / slide
    if base_tagged.exists():
        base = base_tagged
    elif base_plain.exists():
        base = base_plain
    else:
        sys.exit(f"results dir not found: {base_tagged} (nor {base_plain})")
    print(f"Using results dir: {base}")

    df = pd.read_csv(base / "per_patch_predictions_with_hardvote.csv")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    ann = np.load(base / "annotation.npz", allow_pickle=True)
    polys = list(ann["positive"]) + list(ann["negative"])
    thumb_W, thumb_H = meta["thumb_W"], meta["thumb_H"]
    scale = meta["scale"]
    patch_size = meta["patch_size"]

    print(f"Slide:       {slide}")
    print(f"Polygons:    {len(polys)} (positive {len(ann['positive'])}, negative {len(ann['negative'])})")
    print(f"Thumb dims:  {thumb_W}×{thumb_H}, scale={scale:.3f}")

    # ── Build GT counter mask ──
    counter = build_gt_counter(polys, thumb_H, thumb_W, scale)

    # ── Per-patch GT (from patch center) ──
    cx_thumb = ((df["x"].values + patch_size / 2) / scale).astype(int)
    cy_thumb = ((df["y"].values + patch_size / 2) / scale).astype(int)
    cx_thumb = np.clip(cx_thumb, 0, thumb_W - 1)
    cy_thumb = np.clip(cy_thumb, 0, thumb_H - 1)
    n_in = counter[cy_thumb, cx_thumb]

    gt = np.where(n_in == 0, -1,
         np.where(n_in == 1,  0,
                              1))  # 0=gland, 1=non-gland, -1=no GT

    n_gland = int((gt == 0).sum())
    n_nongland = int((gt == 1).sum())
    n_nogt = int((gt == -1).sum())
    n_total = len(gt)
    print(f"\nPatch GT distribution (n={n_total:,}):")
    print(f"  gland       : {n_gland:>6,}  ({n_gland/n_total*100:5.2f}%)")
    print(f"  non-gland   : {n_nongland:>6,}  ({n_nongland/n_total*100:5.2f}%)")
    print(f"  no GT (skip): {n_nogt:>6,}  ({n_nogt/n_total*100:5.2f}%)")
    if n_nogt > n_total / 2:
        print("  [warn] over half the patches have no GT — verify ROI annotation coverage.")

    eval_mask = gt >= 0
    gt_eval = gt[eval_mask]
    if len(gt_eval) == 0:
        sys.exit("No patches with GT — cannot compute metrics.")

    # ── Per-source metrics ──
    rows = []
    for src in SOURCES:
        pred = predicted_labels(df, src)
        pred_eval = pred[eval_mask]
        cm = confusion_matrix(gt_eval, pred_eval, labels=[0, 1])
        acc = accuracy_score(gt_eval, pred_eval)
        prec_per = precision_score(gt_eval, pred_eval, labels=[0, 1],
                                   average=None, zero_division=0)
        rec_per = recall_score(gt_eval, pred_eval, labels=[0, 1],
                                average=None, zero_division=0)
        f1_per = f1_score(gt_eval, pred_eval, labels=[0, 1],
                          average=None, zero_division=0)
        f1_macro = f1_score(gt_eval, pred_eval, average="macro", zero_division=0)
        rows.append({
            "source": src,
            "n_eval": int(eval_mask.sum()),
            "tp_gland": int(cm[0, 0]),
            "fp_gland": int(cm[1, 0]),
            "fn_gland": int(cm[0, 1]),
            "tp_nongland": int(cm[1, 1]),
            "fp_nongland": int(cm[0, 1]),
            "fn_nongland": int(cm[1, 0]),
            "accuracy": float(acc),
            "precision_gland": float(prec_per[0]),
            "recall_gland": float(rec_per[0]),
            "f1_gland": float(f1_per[0]),
            "precision_nongland": float(prec_per[1]),
            "recall_nongland": float(rec_per[1]),
            "f1_nongland": float(f1_per[1]),
            "f1_macro": float(f1_macro),
        })

    metrics = pd.DataFrame(rows)
    metrics_path = base / "metrics_vs_gt.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")
    print(metrics[["source", "n_eval", "accuracy",
                   "f1_gland", "f1_nongland", "f1_macro"]].to_string(index=False))

    # ── Append section to prediction_summary.md ──
    summary_path = base / "prediction_summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    marker = "## S14-2289-1-6 GT 기준 성능"
    new_section = [
        "",
        marker,
        "",
        f"교수님 XML annotation을 GT로 사용 (parity 규칙: ROI 박스 안 단일 polygon → gland, 중첩 polygon 안 → non-gland).",
        "",
        f"- GT 가진 패치: **{int(eval_mask.sum()):,}** "
        f"(gland {n_gland:,} / non-gland {n_nongland:,}). "
        f"ROI 밖 {n_nogt:,} 패치는 평가 제외.",
        "",
        "| Source | n_eval | accuracy | F1 (gland) | F1 (non-gland) | **macro-F1** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        new_section.append(
            f"| {r['source']} | {r['n_eval']:,} | {r['accuracy']:.4f} | "
            f"{r['f1_gland']:.4f} | {r['f1_nongland']:.4f} | **{r['f1_macro']:.4f}** |"
        )
    new_section.append("")
    new_section.append(
        "Precision / recall 등 세부 지표는 `metrics_vs_gt.csv` 참고."
    )

    if marker in summary_text:
        # replace existing section (everything from the marker onwards)
        head = summary_text.split(marker)[0].rstrip() + "\n"
        summary_text = head + "\n".join(new_section) + "\n"
    else:
        summary_text = summary_text.rstrip() + "\n" + "\n".join(new_section) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
