"""
Stage 1 — build a SPIDER-FOV-scale evaluation grid and assign GT labels by
the "annotation overlap ≥ 50%" rule (generalized from the old parity rule).

Output CSV  : per_patch_grid_spider_scale.csv  (in results/<slide>/)
Columns     : x, y, tissue_pct, pct_inROI, pct_nested,
              gt_label_thr050, gt_label_thr025
              (-1 = skip, 0 = gland, 1 = non-gland)

Rule per patch (2240×2240 @ level-0 = 564 µm = SPIDER input FOV):
  pct_inROI  = (counter ≥ 1) / patch_area  (any annotated polygon present)
  pct_nested = (counter ≥ 2) / patch_area  (polygon overlap = our non-gland)
  gt_label   = 1  if pct_nested ≥ τ
             = 0  if pct_inROI  ≥ τ and pct_nested < τ
             = -1 otherwise
  τ = 0.50 (default) and 0.25 (sensitivity column)

Tissue filter: same as visualize_prediction_wsi._is_tissue (sat > 20 ≥ 0.7)
applied on a downscaled patch read (level-2 ~16× downsample for speed).

Usage:
    /root/miniconda3/envs/tiatoolbox/bin/python build_spider_scale_grid.py S14-2289-1-6
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from compute_gt_metrics_spider import build_gt_counter

PATCH_SIZE_LEVEL0 = 2240  # 564 µm @ 0.252 µm/px  ≈ SPIDER 1120 @ 0.504 µm/px
STRIDE_LEVEL0 = 2240      # non-overlap
TAU_PRIMARY = 0.50
TAU_SECONDARY = 0.25
TISSUE_THR = 0.7


def is_tissue_rgb(rgb, threshold=TISSUE_THR):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 1] > 20).mean() >= threshold, (hsv[:, :, 1] > 20).mean()


def find_thumb_level(slide, target_down=16):
    """Pick the openslide level closest to but >= target downsample."""
    best = 0
    for i, d in enumerate(slide.level_downsamples):
        if d <= target_down:
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide", help="slide id (e.g. S14-2289-1-6)")
    args = ap.parse_args()

    cfg = Config()
    results_dir = Path(cfg.base_dir) / "results" / args.slide
    if not results_dir.exists():
        sys.exit(f"results dir missing: {results_dir}")

    # locate svs (may be in external_test_slides map)
    svs_path = Path(cfg.svs_dir) / f"{args.slide}.svs"
    if not svs_path.exists():
        ext = cfg.external_test_slides.get(args.slide, {})
        if ext.get("svs"):
            svs_path = Path(cfg.svs_dir) / ext["svs"]
    if not svs_path.exists():
        sys.exit(f"missing svs: {svs_path}")

    slide = openslide.OpenSlide(str(svs_path))
    W0, H0 = slide.level_dimensions[0]
    print(f"[slide] {svs_path.name}  level-0 dim=({W0},{H0})  MPP={slide.properties.get('openslide.mpp-x')}")

    # GT counter on thumbnail (same as compute_gt_metrics_spider)
    meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
    ann = np.load(results_dir / "annotation.npz", allow_pickle=True)
    polys = list(ann["positive"]) + list(ann["negative"])
    thumb_W, thumb_H, scale = meta["thumb_W"], meta["thumb_H"], meta["scale"]
    print(f"[gt] {len(polys)} polygons "
          f"(positive {len(ann['positive'])}, negative {len(ann['negative'])})  "
          f"thumb=({thumb_W},{thumb_H})  scale={scale:.3f}")
    counter = build_gt_counter(polys, thumb_H, thumb_W, scale)
    # precompute masks once
    mask_inROI = (counter >= 1).astype(np.uint8)
    mask_nested = (counter >= 2).astype(np.uint8)

    # tissue read at a coarse level for speed
    tlevel = find_thumb_level(slide, target_down=16)
    tdown = slide.level_downsamples[tlevel]
    tile_at_tlevel = max(1, int(round(PATCH_SIZE_LEVEL0 / tdown)))
    print(f"[tissue] read at level {tlevel} (down={tdown:.2f})  tile={tile_at_tlevel}px")

    # grid
    n_x = (W0 - PATCH_SIZE_LEVEL0) // STRIDE_LEVEL0 + 1
    n_y = (H0 - PATCH_SIZE_LEVEL0) // STRIDE_LEVEL0 + 1
    total = n_x * n_y
    print(f"[grid] {n_x}×{n_y} = {total} candidate patches  (stride={STRIDE_LEVEL0})")

    rows = []
    t0 = time.time()
    for iy in range(n_y):
        for ix in range(n_x):
            x0 = ix * STRIDE_LEVEL0
            y0 = iy * STRIDE_LEVEL0

            # tissue check at coarse level
            x_t = int(round(x0 / tdown))
            y_t = int(round(y0 / tdown))
            tile_rgba = slide.read_region((x0, y0), tlevel, (tile_at_tlevel, tile_at_tlevel))
            tile_rgb = np.asarray(tile_rgba.convert("RGB"), dtype=np.uint8)
            is_tis, tis_pct = is_tissue_rgb(tile_rgb)

            # GT overlap fraction on thumbnail mask
            # map patch box to thumbnail box
            x0_t = int(np.floor(x0 / scale))
            y0_t = int(np.floor(y0 / scale))
            x1_t = int(np.ceil((x0 + PATCH_SIZE_LEVEL0) / scale))
            y1_t = int(np.ceil((y0 + PATCH_SIZE_LEVEL0) / scale))
            x0_t = max(0, x0_t); y0_t = max(0, y0_t)
            x1_t = min(thumb_W, x1_t); y1_t = min(thumb_H, y1_t)
            sub_in = mask_inROI[y0_t:y1_t, x0_t:x1_t]
            sub_ne = mask_nested[y0_t:y1_t, x0_t:x1_t]
            area_t = max(1, sub_in.size)
            pct_inROI = sub_in.sum() / area_t
            pct_nested = sub_ne.sum() / area_t

            # labels at two thresholds
            def label(t):
                if pct_nested >= t:
                    return 1
                if pct_inROI >= t and pct_nested < t:
                    return 0
                return -1

            rows.append({
                "x": x0,
                "y": y0,
                "is_tissue": int(is_tis),
                "tissue_pct": float(tis_pct),
                "pct_inROI": float(pct_inROI),
                "pct_nested": float(pct_nested),
                "gt_label_thr050": label(TAU_PRIMARY),
                "gt_label_thr025": label(TAU_SECONDARY),
            })
        if (iy + 1) % 5 == 0 or iy == n_y - 1:
            done = (iy + 1) * n_x
            rate = done / (time.time() - t0)
            print(f"  row {iy+1:>3}/{n_y}   {done:>6,}/{total:,}   {rate:5.1f} p/s")

    slide.close()
    df = pd.DataFrame(rows)
    out = results_dir / "per_patch_grid_spider_scale.csv"
    df.to_csv(out, index=False)

    # report
    print(f"\n[save] {out}  ({len(df):,} rows)")
    tis = df[df.is_tissue == 1]
    print(f"[summary] tissue-positive patches: {len(tis):,} / {len(df):,}")
    for thr_col, t in [("gt_label_thr050", 0.50), ("gt_label_thr025", 0.25)]:
        counts = tis[thr_col].value_counts().to_dict()
        n_skip = counts.get(-1, 0)
        n_g = counts.get(0, 0)
        n_ng = counts.get(1, 0)
        print(f"  τ={t}  → gland {n_g:,}   non-gland {n_ng:,}   skip {n_skip:,}")


if __name__ == "__main__":
    main()
