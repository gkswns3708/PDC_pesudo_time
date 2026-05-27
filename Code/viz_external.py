"""
WSI heatmap visualization for external held-out slide (S14-2289-1-6).

Reads prob_map_*.npy and prob_map_ensemble.npy produced by infer_external_slide.py
and produces:
  1. full_thumbnail_4k.png        — raw thumbnail
  2. annotation_overlay_4k.png    — thumbnail + professor's XML annotation polygons
  3. ensemble_heatmap_4k.png      — ★ main: ensemble P(gland) heatmap + annotation outline
  4. per_model_comparison.png     — 2×2: each model + ensemble side-by-side
  5. hard_prediction_map.png      — binary class map (gland vs non-gland)

Usage:
    python viz_external.py S14-2289-1-6
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import Config


HEATMAP_ALPHA = 0.55
ANNOT_COLOR = "#1A66E0"   # blue for prof's annotation outline (non-gland regions)
ROI_COLOR = "#000000"      # ROI big-box outline


def load_arrays(out_dir: Path):
    thumb = np.load(out_dir / "thumbnail.npy")
    meta = np.load(out_dir / "slide_meta.npy", allow_pickle=True).item()
    ann = np.load(out_dir / "annotation.npz", allow_pickle=True)
    positive = list(ann["positive"])
    negative = list(ann["negative"]) if "negative" in ann.files else []
    models = meta["models"]
    prob_maps = {bb: np.load(out_dir / f"prob_map_{bb}.npy") for bb in models}
    valid_masks = {bb: np.load(out_dir / f"valid_mask_{bb}.npy") for bb in models}
    ens_map = np.load(out_dir / "prob_map_ensemble.npy")
    ens_valid = np.load(out_dir / "valid_mask_ensemble.npy")
    return {
        "thumb": thumb, "meta": meta,
        "positive": positive, "negative": negative,
        "prob_maps": prob_maps, "valid_masks": valid_masks,
        "ensemble_map": ens_map, "ensemble_valid": ens_valid,
    }


def split_polygons(polygons, area_threshold_factor=10.0):
    """Heuristic split: ROI big boxes (large area, few vertices) vs non-gland sub-polygons.
    Returns (rois, sub_polys)."""
    if not polygons:
        return [], []
    areas = []
    for poly in polygons:
        x, y = poly[:, 0], poly[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(a)
    areas = np.array(areas)
    # heuristic: ROI = area > median * threshold AND vertices < 30 (rectangles)
    med = np.median(areas)
    rois, subs = [], []
    for poly, a in zip(polygons, areas):
        if a > med * area_threshold_factor and len(poly) < 30:
            rois.append(poly)
        else:
            subs.append(poly)
    return rois, subs


def draw_polys(ax, polys, scale, color, alpha=0.0, lw=1.0, ls="-"):
    for poly in polys:
        xs = poly[:, 0] / scale
        ys = poly[:, 1] / scale
        if alpha > 0:
            ax.fill(xs, ys, color=color, alpha=alpha, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls)


def save_thumbnail(thumb, out_path, title=None):
    fig, ax = plt.subplots(figsize=(thumb.shape[1] / 250, thumb.shape[0] / 250))
    ax.imshow(thumb)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_annotation_overlay(thumb, rois, sub_polys, scale, out_path, slide_name):
    fig, ax = plt.subplots(figsize=(thumb.shape[1] / 250, thumb.shape[0] / 250))
    ax.imshow(thumb)
    ax.set_xticks([]); ax.set_yticks([])
    draw_polys(ax, rois, scale, ROI_COLOR, alpha=0.0, lw=2.5, ls="-")
    draw_polys(ax, sub_polys, scale, ANNOT_COLOR, alpha=0.20, lw=0.8)
    handles = [
        mpatches.Patch(facecolor=ANNOT_COLOR, alpha=0.20, edgecolor=ANNOT_COLOR,
                       label=f"professor's non-gland polygons ({len(sub_polys)})"),
        mpatches.Patch(facecolor="none", edgecolor=ROI_COLOR,
                       label=f"ROI big-box ({len(rois)}) — region for evaluation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(f"{slide_name} — professor's annotation overlay", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_heatmap(thumb, prob_map, valid, rois, sub_polys, scale,
                 out_path, title, with_annotation=True):
    masked = np.ma.masked_where(~valid, prob_map)
    fig, ax = plt.subplots(figsize=(thumb.shape[1] / 250, thumb.shape[0] / 250))
    ax.imshow(thumb)
    im = ax.imshow(masked, cmap="RdBu", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if with_annotation:
        draw_polys(ax, rois, scale, ROI_COLOR, lw=2.0)
        draw_polys(ax, sub_polys, scale, ANNOT_COLOR, lw=0.6, ls="--")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("P(gland)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_per_model_comparison(thumb, prob_maps, valid_masks, ensemble_map, ensemble_valid,
                              rois, sub_polys, scale, out_path, slide_name):
    n_models = len(prob_maps)
    n_panels = n_models + 1  # + ensemble
    cols = 2
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(thumb.shape[1] / 350 * cols, thumb.shape[0] / 350 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    panels = list(prob_maps.items()) + [("ENSEMBLE", ensemble_map)]
    valids = list(valid_masks.values()) + [ensemble_valid]

    for ax, (name, pmap), valid in zip(axes, panels, valids):
        ax.imshow(thumb)
        masked = np.ma.masked_where(~valid, pmap)
        im = ax.imshow(masked, cmap="RdBu", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        draw_polys(ax, rois, scale, ROI_COLOR, lw=1.5)
        draw_polys(ax, sub_polys, scale, ANNOT_COLOR, lw=0.4, ls="--")
        emphasis = " ★" if name == "ENSEMBLE" else ""
        frac = (pmap[valid] >= 0.5).mean() if valid.sum() else 0.0
        ax.set_title(f"{name}{emphasis}  (frac pred-gland = {frac:.2f})", fontsize=11)

    for ax in axes[len(panels):]:
        ax.axis("off")

    plt.suptitle(f"{slide_name} — per-model comparison (P(gland) heatmaps)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_hard_prediction(thumb, ensemble_map, ensemble_valid,
                          rois, sub_polys, scale, out_path, slide_name):
    pred_class = (ensemble_map >= 0.5).astype(np.float32)
    pred_class[~ensemble_valid] = np.nan
    fig, ax = plt.subplots(figsize=(thumb.shape[1] / 250, thumb.shape[0] / 250))
    ax.imshow(thumb)
    cmap = matplotlib.colormaps.get_cmap("RdBu").copy()
    cmap.set_bad(alpha=0.0)
    masked = np.ma.masked_invalid(pred_class)
    ax.imshow(masked, cmap=cmap, alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
    draw_polys(ax, rois, scale, ROI_COLOR, lw=2.0)
    draw_polys(ax, sub_polys, scale, ANNOT_COLOR, lw=0.6, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    handles = [
        mpatches.Patch(facecolor=plt.cm.RdBu(0.99), alpha=HEATMAP_ALPHA, label="predicted gland"),
        mpatches.Patch(facecolor=plt.cm.RdBu(0.01), alpha=HEATMAP_ALPHA, label="predicted non-gland"),
        mpatches.Patch(facecolor="none", edgecolor=ROI_COLOR, label="ROI box"),
        mpatches.Patch(facecolor="none", edgecolor=ANNOT_COLOR, linestyle="--",
                       label="GT non-gland polygons"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(f"{slide_name} — ensemble HARD prediction (threshold 0.5)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_summary_stats(data, slide_name, out_path):
    """Compute simple summary stats inside ROI."""
    thumb = data["thumb"]
    H, W = thumb.shape[:2]
    rois, sub_polys = split_polygons(data["positive"])

    # Build masks at thumb resolution
    import cv2
    scale = data["meta"]["scale"]
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    sub_mask = np.zeros((H, W), dtype=np.uint8)
    for poly in rois:
        pts = (poly / scale).astype(np.int32)
        cv2.fillPoly(roi_mask, [pts], 1)
    for poly in sub_polys:
        pts = (poly / scale).astype(np.int32)
        cv2.fillPoly(sub_mask, [pts], 1)
    inside_roi_outside_sub = (roi_mask == 1) & (sub_mask == 0)
    inside_sub = sub_mask == 1
    outside_roi = roi_mask == 0

    ens_map = data["ensemble_map"]
    ens_valid = data["ensemble_valid"]

    def stats(mask, label):
        m = mask & ens_valid
        n = m.sum()
        if n == 0:
            return f"{label}: no valid pixels"
        p = ens_map[m]
        return (f"{label}: pixels={n}  "
                f"mean P(gland)={p.mean():.3f}  "
                f"frac>=0.5={(p>=0.5).mean():.3f}")

    lines = [
        f"=== {slide_name} ensemble prediction summary ===",
        stats(inside_sub, "INSIDE prof's non-gland polygons (GT non-gland)"),
        stats(inside_roi_outside_sub, "INSIDE ROI but OUTSIDE non-gland polygons (GT gland)"),
        stats(outside_roi, "OUTSIDE ROI (no GT label)"),
        f"",
        f"Models in ensemble: {data['meta']['models']}",
        f"Total positive polygons: {len(data['positive'])}",
        f"  → ROI big-boxes:        {len(rois)}",
        f"  → non-gland sub-polys:  {len(sub_polys)}",
    ]
    txt = "\n".join(lines)
    print(txt)
    out_path.write_text(txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    args = parser.parse_args()

    config = Config()
    out_dir = Path(config.base_dir) / "results" / args.slide
    if not out_dir.exists():
        raise FileNotFoundError(f"No inference output at {out_dir} — run infer_external_slide.py first")

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading inference results from {out_dir}")
    data = load_arrays(out_dir)
    thumb = data["thumb"]
    scale = data["meta"]["scale"]
    rois, sub_polys = split_polygons(data["positive"])
    print(f"Annotation split: {len(rois)} ROI big-boxes + {len(sub_polys)} non-gland sub-polygons")

    save_thumbnail(thumb, viz_dir / "01_full_thumbnail.png",
                   title=f"{args.slide} — raw thumbnail ({thumb.shape[1]}×{thumb.shape[0]} px)")
    save_annotation_overlay(thumb, rois, sub_polys, scale,
                            viz_dir / "02_annotation_overlay.png", args.slide)
    save_heatmap(thumb, data["ensemble_map"], data["ensemble_valid"], rois, sub_polys, scale,
                 viz_dir / "03_ensemble_heatmap.png",
                 f"{args.slide} — ★ ENSEMBLE P(gland) heatmap (blue=gland, red=non-gland)")
    save_per_model_comparison(thumb, data["prob_maps"], data["valid_masks"],
                              data["ensemble_map"], data["ensemble_valid"],
                              rois, sub_polys, scale,
                              viz_dir / "04_per_model_comparison.png", args.slide)
    save_hard_prediction(thumb, data["ensemble_map"], data["ensemble_valid"],
                         rois, sub_polys, scale,
                         viz_dir / "05_hard_prediction.png", args.slide)
    save_summary_stats(data, args.slide, viz_dir / "summary_stats.txt")

    print(f"\nViz saved to: {viz_dir}/")


if __name__ == "__main__":
    main()
