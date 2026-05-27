"""
Visualize Cancer-only Virchow2 evaluation for S14-2289-1-6 (ROI focus).

Reads:
  results/<slide>/per_patch_predictions.csv
  results/<slide>/kather_per_patch.csv
  results/<slide>/evaluation_cancer_only.csv  (must run evaluate_cancer_only.py first)
  results/<slide>/thumbnail.npy + slide_meta.npy
  Data/S14/Annotation/<slide>.xml

Produces (in results/<slide>/viz_cancer_only/, cropped to ROI bounding box):
  01_kather_3group_map.png      — Cancer/Normal/Others dominant per patch
  02_cancer_mask.png            — binary cancer mask
  03_evaluation_area.png        — ROI ∩ Cancer evaluation region
  04_GT_label_map.png           — GT gland/non-gland labels (from XML)
  05_virchow2_prediction.png    — Virchow2 P(gland) in cancer mask
  06_error_map.png              — green correct / red wrong
  07_overview_panel.png         — 2×3 combined panel
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree


from config import Config
from evaluate_cancer_only import parse_polygons


# colors
GLAND_BLUE = (0.15, 0.40, 0.95)
NONGLAND_RED = (0.90, 0.25, 0.20)
CANCER_GREEN = (0.30, 0.80, 0.40)
NORMAL_YELLOW = (0.95, 0.80, 0.30)
OTHERS_GRAY = (0.60, 0.60, 0.60)
ROI_OUTLINE = "#000000"
NONGLAND_OUTLINE = "#1A66E0"
HEATMAP_ALPHA = 0.60


def patch_to_thumb_rect(x, y, patch_size, scale):
    tx0 = int(x / scale); ty0 = int(y / scale)
    tx1 = int((x + patch_size) / scale)
    ty1 = int((y + patch_size) / scale)
    return tx0, ty0, tx1, ty1


def build_pixel_label_map(df, key, H, W, patch_size, scale, missing=np.nan):
    """For each patch, fill its area in a (H,W) array with df[key]."""
    out = np.full((H, W), missing, dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)
    for x, y, v in zip(df["x"].values, df["y"].values, df[key].values):
        tx0, ty0, tx1, ty1 = patch_to_thumb_rect(x, y, patch_size, scale)
        tx1 = min(tx1, W); ty1 = min(ty1, H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        # average overlapping
        prev = out[ty0:ty1, tx0:tx1]
        c = count[ty0:ty1, tx0:tx1]
        prev_safe = np.where(np.isnan(prev), 0.0, prev)
        new = (prev_safe * c + float(v)) / (c + 1)
        out[ty0:ty1, tx0:tx1] = new
        count[ty0:ty1, tx0:tx1] = c + 1
    return out, count > 0


def build_argmax_map(df, key, H, W, patch_size, scale, value_map=None):
    """For categorical key (e.g. group_argmax), build integer-coded label map.
    value_map: dict {string: int}. Returns (label_map, valid_mask)."""
    out = np.full((H, W), -1, dtype=np.int8)
    for x, y, v in zip(df["x"].values, df["y"].values, df[key].values):
        tx0, ty0, tx1, ty1 = patch_to_thumb_rect(x, y, patch_size, scale)
        tx1 = min(tx1, W); ty1 = min(ty1, H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        if value_map is not None:
            iv = value_map.get(v, -1)
        else:
            iv = int(v)
        out[ty0:ty1, tx0:tx1] = iv
    return out, out >= 0


def crop_to_bbox_thumb(thumb, roi_polys, scale, pad=80):
    """Compute crop bbox (in thumb coords) covering all ROI polygons + padding."""
    if not roi_polys:
        return None
    all_pts = np.vstack([p / scale for p in roi_polys])
    x0, y0 = all_pts.min(axis=0); x1, y1 = all_pts.max(axis=0)
    H, W = thumb.shape[:2]
    bx0 = max(0, int(x0) - pad); by0 = max(0, int(y0) - pad)
    bx1 = min(W, int(x1) + pad); by1 = min(H, int(y1) + pad)
    return (bx0, by0, bx1, by1)


def draw_polys_on_ax(ax, polys, scale, color, lw=1.2, ls="-", alpha_fill=0.0):
    for poly in polys:
        xs = poly[:, 0] / scale; ys = poly[:, 1] / scale
        if alpha_fill > 0:
            ax.fill(xs, ys, color=color, alpha=alpha_fill, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls)


def save_panel(thumb_crop, overlay, cmap, vmin, vmax, label, save_path,
               rois_crop, sub_polys_crop, valid_mask=None,
               handles=None, alpha=HEATMAP_ALPHA, colorbar_ticks=None):
    fig, ax = plt.subplots(figsize=(thumb_crop.shape[1] / 200, thumb_crop.shape[0] / 200))
    ax.imshow(thumb_crop)
    if overlay is not None:
        if valid_mask is not None:
            masked = np.ma.masked_where(~valid_mask, overlay)
        else:
            masked = overlay
        im = ax.imshow(masked, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        if colorbar_ticks is not None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_ticks(colorbar_ticks)
    # draw ROI + non-gland polygon outlines (cropped coords)
    for poly in rois_crop:
        ax.plot(poly[:, 0], poly[:, 1], color=ROI_OUTLINE, linewidth=2.0)
    for poly in sub_polys_crop:
        ax.plot(poly[:, 0], poly[:, 1], color=NONGLAND_OUTLINE, linewidth=0.6, linestyle="--")
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--cancer-threshold", type=float, default=0.5)
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
    viz_dir = out_dir / "viz_cancer_only"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(out_dir / "evaluation_cancer_only.csv")
    print(f"Loaded {len(df)} evaluated patches")
    thumb = np.load(out_dir / "thumbnail.npy")
    meta = np.load(out_dir / "slide_meta.npy", allow_pickle=True).item()
    scale = meta["scale"]
    H, W = thumb.shape[:2]
    print(f"Thumb {W}x{H}, scale={scale:.3f}")

    rois, sub_polys = parse_polygons(xml_path)
    print(f"{len(rois)} ROI box(es) + {len(sub_polys)} non-gland sub-polygons")

    # Crop bbox to ROI region
    bbox = crop_to_bbox_thumb(thumb, rois, scale, pad=80)
    if bbox is None:
        raise RuntimeError("No ROI boxes found in XML")
    bx0, by0, bx1, by1 = bbox
    thumb_crop = thumb[by0:by1, bx0:bx1]
    print(f"ROI crop bbox: [{bx0}:{bx1}, {by0}:{by1}]  -> {thumb_crop.shape}")

    # Translate polys to crop coords
    rois_crop = [np.column_stack([p[:, 0] / scale - bx0, p[:, 1] / scale - by0]) for p in rois]
    sub_polys_crop = [np.column_stack([p[:, 0] / scale - bx0, p[:, 1] / scale - by0]) for p in sub_polys]

    # ── Build per-pixel maps in CROP coordinates ──
    # Adjust df coords: subtract crop offset (in thumb pixels → multiply by scale for slide coords)
    # Easier: build full-thumb maps then crop.
    def crop(arr):
        return arr[by0:by1, bx0:bx1]

    # Kather 3-group map (Cancer=0, Normal=1, Others=2)
    grp_vmap = {"Cancer": 0, "Normal": 1, "Others": 2}
    grp_map, grp_valid = build_argmax_map(df, "group_argmax", H, W, config.patch_size, scale, grp_vmap)
    grp_map_crop = crop(grp_map.astype(np.float32))
    grp_valid_crop = crop(grp_valid)
    grp_cmap = matplotlib.colors.ListedColormap([CANCER_GREEN, NORMAL_YELLOW, OTHERS_GRAY])
    save_panel(
        thumb_crop, grp_map_crop, grp_cmap, vmin=-0.5, vmax=2.5,
        label=f"{args.slide} — Kather 3-group dominant\n(Cancer / Normal / Others)",
        save_path=viz_dir / "01_kather_3group_map.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=grp_valid_crop,
        handles=[
            mpatches.Patch(facecolor=CANCER_GREEN, alpha=HEATMAP_ALPHA, label="Cancer (TUM+STR)"),
            mpatches.Patch(facecolor=NORMAL_YELLOW, alpha=HEATMAP_ALPHA, label="Normal (NORM+ADI+MUS+LYM)"),
            mpatches.Patch(facecolor=OTHERS_GRAY, alpha=HEATMAP_ALPHA, label="Others (BACK+DEB(necrosis)+MUC)"),
        ],
    )

    # Cancer mask binary
    cancer_map, _ = build_pixel_label_map(df, "p_Cancer", H, W, config.patch_size, scale)
    cancer_binary_crop = crop((cancer_map >= args.cancer_threshold).astype(np.float32))
    cancer_valid_crop = crop(~np.isnan(cancer_map))
    save_panel(
        thumb_crop, cancer_binary_crop,
        matplotlib.colors.ListedColormap(["#cccccc", CANCER_GREEN]),
        vmin=-0.5, vmax=1.5,
        label=f"{args.slide} — Cancer mask (p_Cancer ≥ {args.cancer_threshold})",
        save_path=viz_dir / "02_cancer_mask.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=cancer_valid_crop,
        handles=[
            mpatches.Patch(facecolor=CANCER_GREEN, alpha=HEATMAP_ALPHA, label="Cancer (TUM+STR)"),
            mpatches.Patch(facecolor="#cccccc", alpha=HEATMAP_ALPHA, label="Normal / Other / Necrosis"),
        ],
    )

    # Evaluation area = cancer_mask AND in_roi (df is already roi-filtered)
    eval_map, _ = build_pixel_label_map(df, "cancer_mask", H, W, config.patch_size, scale)
    eval_binary_crop = crop((eval_map >= 0.5).astype(np.float32))
    save_panel(
        thumb_crop, eval_binary_crop,
        matplotlib.colors.ListedColormap(["#bbbbbb", "#FFAA33"]),
        vmin=-0.5, vmax=1.5,
        label=f"{args.slide} — Evaluation area (ROI ∩ Cancer mask)",
        save_path=viz_dir / "03_evaluation_area.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=crop(~np.isnan(eval_map)),
        handles=[
            mpatches.Patch(facecolor="#FFAA33", alpha=HEATMAP_ALPHA, label="evaluated patches"),
            mpatches.Patch(facecolor="#bbbbbb", alpha=HEATMAP_ALPHA, label="excluded (non-cancer)"),
        ],
    )

    # GT label map (only within ROI ∩ Cancer for clarity)
    df_eval = df[df["cancer_mask"]]
    gt_map, _ = build_pixel_label_map(df_eval, "gt_label", H, W, config.patch_size, scale)
    save_panel(
        thumb_crop, crop(gt_map),
        matplotlib.colors.ListedColormap([GLAND_BLUE, NONGLAND_RED]),
        vmin=-0.5, vmax=1.5,
        label=f"{args.slide} — Ground Truth label (eval patches only)",
        save_path=viz_dir / "04_GT_label_map.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=crop(~np.isnan(gt_map)),
        handles=[
            mpatches.Patch(facecolor=GLAND_BLUE, alpha=HEATMAP_ALPHA, label="GT gland (outside non-gland poly)"),
            mpatches.Patch(facecolor=NONGLAND_RED, alpha=HEATMAP_ALPHA, label="GT non-gland (inside poly)"),
        ],
    )

    # Virchow2 prediction map (only within ROI ∩ Cancer)
    # Need original p_gland_virchow2 column. Compute pred_class color map.
    pred_map, _ = build_pixel_label_map(df_eval, "pred_label", H, W, config.patch_size, scale)
    save_panel(
        thumb_crop, crop(pred_map),
        matplotlib.colors.ListedColormap([GLAND_BLUE, NONGLAND_RED]),
        vmin=-0.5, vmax=1.5,
        label=f"{args.slide} — Virchow2 prediction (Cancer-only)",
        save_path=viz_dir / "05_virchow2_prediction.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=crop(~np.isnan(pred_map)),
        handles=[
            mpatches.Patch(facecolor=GLAND_BLUE, alpha=HEATMAP_ALPHA, label="pred gland"),
            mpatches.Patch(facecolor=NONGLAND_RED, alpha=HEATMAP_ALPHA, label="pred non-gland"),
        ],
    )

    # Error map (green correct / red wrong)
    df_eval = df_eval.copy()
    df_eval["correct_int"] = df_eval["correct"].astype(int)
    err_map, _ = build_pixel_label_map(df_eval, "correct_int", H, W, config.patch_size, scale)
    save_panel(
        thumb_crop, crop(err_map),
        matplotlib.colors.ListedColormap(["#dd2222", "#22aa55"]),
        vmin=-0.5, vmax=1.5,
        label=f"{args.slide} — Correctness (Cancer-only)",
        save_path=viz_dir / "06_error_map.png",
        rois_crop=rois_crop, sub_polys_crop=sub_polys_crop,
        valid_mask=crop(~np.isnan(err_map)),
        handles=[
            mpatches.Patch(facecolor="#22aa55", alpha=HEATMAP_ALPHA, label="correct"),
            mpatches.Patch(facecolor="#dd2222", alpha=HEATMAP_ALPHA, label="wrong"),
        ],
    )

    # 2x3 overview panel
    metrics_path = out_dir / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f: metrics = json.load(f)
    else:
        metrics = {}

    fig, axes = plt.subplots(2, 3, figsize=(thumb_crop.shape[1]/130*3, thumb_crop.shape[0]/130*2))
    panels_data = [
        ("01: Kather 3-group", crop(grp_map.astype(np.float32)),
         matplotlib.colors.ListedColormap([CANCER_GREEN, NORMAL_YELLOW, OTHERS_GRAY]), -0.5, 2.5, grp_valid_crop),
        ("02: Cancer mask", cancer_binary_crop,
         matplotlib.colors.ListedColormap(["#cccccc", CANCER_GREEN]), -0.5, 1.5, cancer_valid_crop),
        ("03: Eval area (ROI∩Cancer)", eval_binary_crop,
         matplotlib.colors.ListedColormap(["#bbbbbb", "#FFAA33"]), -0.5, 1.5, crop(~np.isnan(eval_map))),
        ("04: GT (eval patches only)", crop(gt_map),
         matplotlib.colors.ListedColormap([GLAND_BLUE, NONGLAND_RED]), -0.5, 1.5, crop(~np.isnan(gt_map))),
        ("05: Virchow2 pred", crop(pred_map),
         matplotlib.colors.ListedColormap([GLAND_BLUE, NONGLAND_RED]), -0.5, 1.5, crop(~np.isnan(pred_map))),
        ("06: Correctness", crop(err_map),
         matplotlib.colors.ListedColormap(["#dd2222", "#22aa55"]), -0.5, 1.5, crop(~np.isnan(err_map))),
    ]
    for ax, (title, m, cmap, vmin, vmax, valid) in zip(axes.ravel(), panels_data):
        ax.imshow(thumb_crop)
        if m is not None:
            ax.imshow(np.ma.masked_where(~valid, m), cmap=cmap, alpha=HEATMAP_ALPHA, vmin=vmin, vmax=vmax)
        for poly in rois_crop:
            ax.plot(poly[:, 0], poly[:, 1], color=ROI_OUTLINE, linewidth=1.5)
        for poly in sub_polys_crop:
            ax.plot(poly[:, 0], poly[:, 1], color=NONGLAND_OUTLINE, linewidth=0.4, linestyle="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)

    metric_str = ""
    if metrics:
        metric_str = (f"  Acc={metrics.get('accuracy',0):.3f}, "
                      f"F1_macro={metrics.get('f1_macro',0):.3f}, "
                      f"F1_gland={metrics.get('f1_gland',0):.3f}, "
                      f"F1_non={metrics.get('f1_non_gland',0):.3f}  "
                      f"(n={metrics.get('n_patches_cancer_eval',0)})")
    plt.suptitle(f"{args.slide} — Cancer-only Virchow2 evaluation{metric_str}", fontsize=14)
    plt.tight_layout()
    plt.savefig(viz_dir / "07_overview_panel.png", dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\nAll viz saved to: {viz_dir}/")


if __name__ == "__main__":
    main()
