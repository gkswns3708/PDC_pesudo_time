"""
Stage 3 — score SPIDER predictions at SPIDER-FOV-scale + overlay viz.

Reads per_patch_predictions_spider_scale.csv.
Computes F1/precision/recall/CM for:
  - GT thr=0.50 (primary)  and  GT thr=0.25 (sensitivity)
  - prediction variants: hard top1, softrenorm (p_high>p_low), threshold sweep
Writes metrics_vs_gt_spider_scale.csv.

Visual: thumbnail with grid boxes
  GT non-gland     : red
  pred non-gland   : blue
  agreement (both) : purple
  GT gland         : grey
  skip / no GT     : not drawn
"""

import sys
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from compute_gt_metrics_spider import score

PATCH_SIZE_LEVEL0 = 2240


def make_overlay(slide_path, pred_df, gt_col, scale, thumb_W, thumb_H,
                 out_path, pred_binary_col="pred_binary"):
    slide = openslide.OpenSlide(str(slide_path))
    thumb = slide.get_thumbnail((thumb_W, thumb_H)).convert("RGBA")
    slide.close()
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for _, r in pred_df.iterrows():
        x0 = int(r.x / scale); y0 = int(r.y / scale)
        x1 = int((r.x + PATCH_SIZE_LEVEL0) / scale)
        y1 = int((r.y + PATCH_SIZE_LEVEL0) / scale)
        gt = int(r[gt_col])
        pr = int(r[pred_binary_col])
        if gt == -1:
            continue  # skip ambiguous
        if gt == 1 and pr == 1:
            color = (160, 32, 240, 180)   # purple — both non-gland
        elif gt == 1:
            color = (220, 0, 0, 180)      # red — GT non-gland only
        elif pr == 1:
            color = (0, 120, 255, 160)    # blue — pred non-gland only
        else:
            color = (140, 140, 140, 90)   # grey — both gland
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
    out = Image.alpha_composite(thumb, overlay)
    out.convert("RGB").save(out_path, optimize=True)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python score_spider_scale.py <slide>")
    slide_id = sys.argv[1]
    cfg = Config()
    base = Path(cfg.base_dir) / "results" / slide_id
    pred = pd.read_csv(base / "per_patch_predictions_spider_scale.csv")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    thumb_W, thumb_H, scale = meta["thumb_W"], meta["thumb_H"], meta["scale"]
    print(f"[input] {len(pred):,} patches  thumb=({thumb_W},{thumb_H})")

    # locate svs for overlay
    svs_path = Path(cfg.svs_dir) / f"{slide_id}.svs"
    if not svs_path.exists():
        ext = cfg.external_test_slides.get(slide_id, {})
        if ext.get("svs"):
            svs_path = Path(cfg.svs_dir) / ext["svs"]

    rows = []
    for gt_col, tau in [("gt_label_thr050", 0.50), ("gt_label_thr025", 0.25)]:
        gt = pred[gt_col].values
        eval_mask = gt >= 0
        gt_eval = gt[eval_mask]
        n_g = (gt_eval == 0).sum(); n_ng = (gt_eval == 1).sum()
        print(f"\n=== GT τ={tau:.2f} ===  gland {n_g}  non-gland {n_ng}  "
              f"(skip {(~eval_mask).sum()})")
        if len(gt_eval) == 0:
            continue

        if "pred_binary" in pred.columns:
            s = score(gt_eval, pred["pred_binary"].values[eval_mask],
                      f"spider_scale_hard_tau{tau:.2f}")
            rows.append(s)

        if "pred_binary_softrenorm" in pred.columns:
            top1 = pred["top1_class"].values
            is_adeno = np.isin(top1, ["Adenocarcinoma high grade",
                                      "Adenocarcinoma low grade"])
            sub = eval_mask & is_adeno
            if sub.sum() > 0:
                s = score(gt[sub],
                          pred["pred_binary_softrenorm"].values[sub],
                          f"spider_scale_softrenorm_adeno_only_tau{tau:.2f}",
                          n_eval_skip=int((eval_mask & ~is_adeno).sum()))
                rows.append(s)

        if "p_high_grade" in pred.columns:
            p_high = pred["p_high_grade"].values
            for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
                pred_thr = (p_high > thr).astype(int)
                s = score(gt_eval, pred_thr[eval_mask],
                          f"spider_scale_p_high>{thr:.2f}_tau{tau:.2f}")
                rows.append(s)

    out_df = pd.DataFrame(rows)
    out_csv = base / "metrics_vs_gt_spider_scale.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")
    print(out_df[["source", "n_eval", "accuracy",
                  "f1_gland", "f1_nongland", "f1_macro"]].to_string(index=False))

    # head-to-head vs original 256-grid SPIDER metric
    old = base / "metrics_vs_gt_spider.csv"
    if old.exists():
        old_df = pd.read_csv(old)
        cmb = pd.concat([
            old_df[["source", "n_eval", "accuracy",
                    "f1_gland", "f1_nongland", "f1_macro"]],
            out_df[["source", "n_eval", "accuracy",
                    "f1_gland", "f1_nongland", "f1_macro"]],
        ], ignore_index=True)
        print("\n=== old 512-grid metric  vs  SPIDER-scale 2240-grid metric ===")
        print(cmb.to_string(index=False))

    # overlay viz for each τ
    if svs_path.exists():
        for gt_col, tau in [("gt_label_thr050", 0.50), ("gt_label_thr025", 0.25)]:
            out_png = base / f"spider_scale_overlay_tau{int(tau*100):03d}.png"
            make_overlay(svs_path, pred, gt_col, scale, thumb_W, thumb_H, out_png)
            print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
