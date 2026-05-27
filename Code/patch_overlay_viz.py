"""
Patch-level overlay — paint each 512x512 patch on the WSI thumbnail with its
hard label color (blue=gland, red=non-gland), per model + hard-vote.

Companion to `prediction_overlay.png` (smooth heatmap version): this version
shows discrete per-patch decisions so the professor can see exactly which
patches each model labeled which way.

Inputs (from /app/Gland_Seg/results/<slide>/):
  per_patch_predictions_with_hardvote.csv
  thumbnail.npy / slide_meta.npy / annotation.npz

Output:
  /app/Gland_Seg/results/<slide>/patch_overlay.png

Usage:
    python patch_overlay_viz.py S14-2289-1-6
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_external_predictions import (
    build_hardvote_pixel_map, render_polys,
)


GLAND_COLOR = (0.15, 0.40, 0.95)      # blue
NONGLAND_COLOR = (0.90, 0.25, 0.20)   # red
ANNOT_COLOR = (1.00, 0.85, 0.10)      # yellow
ALPHA = 0.55


def draw_patch_panel(ax, thumb, hard_map, valid, polys_pos, polys_neg, scale, title):
    ax.imshow(thumb)
    masked = np.ma.array(hard_map, mask=~valid)
    # RdBu_r: 0 → blue (gland), 1 → red (non-gland)
    ax.imshow(masked, cmap="RdBu_r", vmin=0.0, vmax=1.0, alpha=ALPHA,
              interpolation="none")
    render_polys(ax, polys_pos, scale, edge=ANNOT_COLOR, lw=0.6, ls="-")
    render_polys(ax, polys_neg, scale, edge=(1, 0.4, 0.7), lw=0.6, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_overlay_viz.py <slide_name>")
        sys.exit(1)
    slide = sys.argv[1]
    base = Path("/app/Gland_Seg") / "results" / slide
    if not base.exists():
        sys.exit(f"results dir not found: {base}")

    df = pd.read_csv(base / "per_patch_predictions_with_hardvote.csv")
    thumb = np.load(base / "thumbnail.npy")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    ann = np.load(base / "annotation.npz", allow_pickle=True)
    polys_pos = list(ann["positive"])
    polys_neg = list(ann["negative"])

    H, W = thumb.shape[:2]
    scale = meta["scale"]
    patch_size = meta["patch_size"]

    xs = df["x"].values
    ys = df["y"].values

    panels = [
        ("virchow2",  df["pred_virchow2"].values),
        ("uni2",      df["pred_uni2"].values),
        ("phikon-v2", df["pred_phikon-v2"].values),
        ("hardvote",  df["pred_hardvote"].values),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    # Panel 0 — thumbnail + GT
    axes[0].imshow(thumb)
    render_polys(axes[0], polys_pos, scale, edge=ANNOT_COLOR, lw=0.6, ls="-")
    render_polys(axes[0], polys_neg, scale, edge=(1, 0.4, 0.7), lw=0.6, ls="--")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(f"{slide} — thumbnail + GT annotation\n"
                      f"{len(polys_pos)} positive, {len(polys_neg)} negative polygons",
                      fontsize=10)

    # Panels 1..4 — per-model + hardvote patch boxes
    for ax, (name, pred_str) in zip(axes[1:], panels):
        # 1 = non-gland, 0 = gland
        hard = (pred_str == "non-gland").astype(np.int8)
        label_map, valid = build_hardvote_pixel_map(
            xs, ys, hard, H, W, patch_size, scale)
        n_g = int((hard == 0).sum())
        n_n = int((hard == 1).sum())
        draw_patch_panel(
            ax, thumb, label_map, valid, polys_pos, polys_neg, scale,
            f"{name} — patch-level hard label\n"
            f"gland={n_g:,}  non-gland={n_n:,}  "
            f"frac_gland={n_g/(n_g+n_n):.3f}",
        )

    # Hide leftover axis (we have 5 panels in a 2×3 grid)
    for j in range(1 + len(panels), len(axes)):
        axes[j].axis("off")

    handles = [
        mpatches.Patch(color=GLAND_COLOR, label="gland"),
        mpatches.Patch(color=NONGLAND_COLOR, label="non-gland"),
        mpatches.Patch(facecolor="none", edgecolor=ANNOT_COLOR,
                       label="annotation polygon (positive)"),
        mpatches.Patch(facecolor="none", edgecolor=(1, 0.4, 0.7),
                       label="annotation polygon (negative ROA)", linestyle="--"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"{slide} — per-patch hard-label overlay (each box = 512×512 patch)",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = base / "patch_overlay.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
