"""
Generate 2048×2048 high-resolution prediction overlay from prob_map.npy.

Per (slide, model_tag):
  - thumbnail (slide RGB) background
  - prob heatmap (jet colormap, alpha-blended) for non-gland probability
  - GT polygon overlay (yellow outline = our annotation)
  - title + legend

Usage:
    python make_hires_overlay.py \
      --slide S14-2289-1-6 \
      --run_tag _byext_224_20x_raw_ce \
      --model hibou-l \
      --label "Hibou-L Raw+CE"
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config import Config

OUT_SIZE = 2048


def render(slide_id, run_tag, model, label, out_path, crop_anno=False):
    cfg = Config()
    results_dir = Path(cfg.base_dir) / "results" / f"{slide_id}{run_tag}"

    thumb_path = results_dir / "thumbnail.npy"
    prob_path = results_dir / f"prob_map_{model}.npy"
    valid_path = results_dir / f"valid_mask_{model}.npy"
    ann_path = results_dir / "annotation.npz"
    meta_path = results_dir / "slide_meta.npy"

    for p in [thumb_path, prob_path, valid_path, ann_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    thumb = np.load(thumb_path)              # (H, W, 3) uint8
    prob = np.load(prob_path)                # (H, W) float — p_gland (so non-gland = 1-prob)
    valid = np.load(valid_path).astype(bool) # (H, W) bool
    meta = np.load(meta_path, allow_pickle=True).item()
    ann = np.load(ann_path, allow_pickle=True)
    pos = [np.asarray(p, dtype=np.float64) for p in ann["positive"]]
    neg = [np.asarray(p, dtype=np.float64) for p in ann["negative"]]
    scale = float(meta["scale"])
    H, W = thumb.shape[:2]

    # ── Optional: crop to annotation bounding box (L0 coords -> thumb coords) ──
    if crop_anno and (pos or neg):
        all_pts = np.concatenate([p for p in (pos + neg) if len(p) > 0])
        x_min_L0, y_min_L0 = all_pts.min(axis=0)
        x_max_L0, y_max_L0 = all_pts.max(axis=0)
        # convert L0 to thumbnail pixels (divide by `scale`)
        x_min_t = max(0, int(np.floor(x_min_L0 / scale)))
        y_min_t = max(0, int(np.floor(y_min_L0 / scale)))
        x_max_t = min(W, int(np.ceil(x_max_L0 / scale)))
        y_max_t = min(H, int(np.ceil(y_max_L0 / scale)))
        # add 5% padding
        pad_x = int(0.05 * (x_max_t - x_min_t))
        pad_y = int(0.05 * (y_max_t - y_min_t))
        x_min_t = max(0, x_min_t - pad_x)
        y_min_t = max(0, y_min_t - pad_y)
        x_max_t = min(W, x_max_t + pad_x)
        y_max_t = min(H, y_max_t + pad_y)
        # crop all maps
        thumb = thumb[y_min_t:y_max_t, x_min_t:x_max_t]
        prob = prob[y_min_t:y_max_t, x_min_t:x_max_t]
        valid = valid[y_min_t:y_max_t, x_min_t:x_max_t]
        # shift polygons in L0 coords so cropped region's top-left = (0,0)
        x_off_L0 = x_min_t * scale
        y_off_L0 = y_min_t * scale
        pos = [p - np.array([x_off_L0, y_off_L0]) for p in pos]
        neg = [p - np.array([x_off_L0, y_off_L0]) for p in neg]
        H, W = thumb.shape[:2]
        print(f"[crop] anno bbox L0=({x_min_L0:.0f},{y_min_L0:.0f})-({x_max_L0:.0f},{y_max_L0:.0f}) "
              f"→ thumb crop ({x_min_t},{y_min_t})-({x_max_t},{y_max_t}) = {W}×{H}")

    # ── Resize all maps to OUT_SIZE keeping aspect ratio ──
    aspect = W / H
    if aspect >= 1:
        new_w = OUT_SIZE
        new_h = int(round(OUT_SIZE / aspect))
    else:
        new_h = OUT_SIZE
        new_w = int(round(OUT_SIZE * aspect))

    thumb_r = cv2.resize(thumb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    prob_r = cv2.resize(prob, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    valid_r = cv2.resize(valid.astype(np.uint8), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)

    # Compute scale from L0 coords to NEW pixel coords:
    # L0 -> thumb: divide by `scale`
    # thumb -> resized: multiply by new_w/W
    ann_scale = scale * (W / new_w)  # divide L0 coords by ann_scale to get resized pixel

    # Non-gland prob = 1 - p_gland (your prob_map stores p_gland)
    nongland = 1.0 - prob_r

    # ── Render ──
    fig, ax = plt.subplots(1, 1, figsize=(OUT_SIZE/100, OUT_SIZE/100), dpi=100)
    ax.imshow(thumb_r)

    # Solid CYAN overlay for "predicted non-gland" (prob >= 0.5)
    # Alpha proportional to confidence above 0.5 → strongest at prob = 1.0
    # Pink/purple H&E background → bright cyan is maximally contrasting (complementary)
    pred_ng_mask = (nongland >= 0.5) & valid_r
    pred_alpha = np.where(pred_ng_mask, np.clip((nongland - 0.5) * 2 * 0.85, 0, 0.85), 0)
    cyan_overlay = np.zeros((new_h, new_w, 4), dtype=np.float32)
    cyan_overlay[..., 1] = 1.0       # G
    cyan_overlay[..., 2] = 1.0       # B  (cyan = G+B)
    cyan_overlay[..., 3] = pred_alpha
    ax.imshow(cyan_overlay)

    # Patch grid (light gray, very thin) at 448 L0 stride = ~14.7 thumb px = ~24 resized px
    # Only draw within valid region to avoid clutter
    patch_size_L0 = 448
    grid_step = patch_size_L0 / ann_scale  # resized pixels per patch
    # vertical lines
    for x in np.arange(0, new_w, grid_step):
        ax.axvline(x, color="white", linewidth=0.15, alpha=0.20)
    for y in np.arange(0, new_h, grid_step):
        ax.axhline(y, color="white", linewidth=0.15, alpha=0.20)

    # GT — convention-aware (handles both styles):
    #   S14-177 style: 1 outer pos + 32 NegativeROA nested → use polygons directly
    #   S14-2289 style: 149 overlapping positives, 0 negatives → "counter ≥ 2" = non-gland
    # We rasterize a counter map at resized resolution and use that for both display layers.
    counter = np.zeros((new_h, new_w), dtype=np.int16)
    for poly in pos + neg:
        pts = (poly / ann_scale).round().astype(np.int32)
        m = np.zeros((new_h, new_w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        counter += m.astype(np.int16)
    mask_inROI = counter >= 1     # tumor region (any annotation)
    mask_nested = counter >= 2    # non-gland (overlap) OR explicit NegativeROA
    # for S14-177 style, also mark NegativeROA explicitly as nested
    if len(neg) > 0:
        for poly in neg:
            pts = (poly / ann_scale).round().astype(np.int32)
            m = np.zeros((new_h, new_w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            mask_nested = mask_nested | (m.astype(bool))

    # Outer tumor ROI outline (lime green)
    outer_contours, _ = cv2.findContours((mask_inROI & ~mask_nested).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in outer_contours:
        if len(c) >= 3:
            pts = c.reshape(-1, 2)
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="lime", linewidth=1.0, alpha=0.85)
    # Non-gland region (yellow filled, thick outline)
    yellow_overlay = np.zeros((new_h, new_w, 4), dtype=np.float32)
    yellow_overlay[mask_nested, 0] = 1.0  # R
    yellow_overlay[mask_nested, 1] = 1.0  # G
    yellow_overlay[mask_nested, 3] = 0.35 # alpha fill
    ax.imshow(yellow_overlay)
    ng_contours, _ = cv2.findContours(mask_nested.astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in ng_contours:
        if len(c) >= 3:
            pts = c.reshape(-1, 2)
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="yellow", linewidth=1.8, alpha=0.95)

    # Title + legend
    title = f"{slide_id}   ·   {label}"
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10, color="white")
    legend_elems = [
        Patch(facecolor=(0, 1, 1, 0.85), label="Predicted non-gland (p ≥ 0.5)"),
        Patch(facecolor=(1, 1, 0, 0.35), edgecolor="yellow", label="GT non-gland"),
        Patch(edgecolor="lime", facecolor="none", label="GT outer ROI (tumor)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=11, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[save] {out_path}  ({new_w}×{new_h} content in {OUT_SIZE}×{OUT_SIZE} canvas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--run_tag", required=True, help="e.g. _byext_224_20x_raw_ce")
    ap.add_argument("--model", required=True, help="e.g. hibou-l")
    ap.add_argument("--label", required=True, help="display label, e.g. 'Hibou-L Raw+CE'")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--crop_anno", action="store_true",
                    help="crop to annotation bounding box (+5% padding) before rendering")
    args = ap.parse_args()

    if args.out is None:
        out_dir = Path("/app/Gland_Seg/results/_hires_overlays")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = args.label.replace(" ", "_").replace("+", "p").replace("/", "_")
        suffix = "__cropped" if args.crop_anno else ""
        args.out = out_dir / f"{args.slide}__{args.model}__{safe_label}{suffix}.png"

    render(args.slide, args.run_tag, args.model, args.label, args.out,
           crop_anno=args.crop_anno)


if __name__ == "__main__":
    main()
