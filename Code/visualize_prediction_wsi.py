"""
WSI-wide prediction heatmap — scan the ENTIRE slide (all tissue), not just
annotated regions.

For each fold's best_model_fold{N}.pth:
    1. Load the val slide
    2. Sliding-window over level 0 with stride config.wsi_stride (default 512)
    3. Multiprocess workers: per-patch read → tissue filter → Macenko transform
    4. Main process: batched GPU inference
    5. Accumulate P(gland) and predicted-class maps at thumbnail resolution
    6. Save 2×2 figure:
         (0,0) raw thumbnail
         (0,1) raw + XML annotation polygons (reference)
         (1,0) P(gland) heatmap — RdBu divergent (blue=gland, red=non-gland)
         (1,1) Hard prediction map — same palette, confidence-faded
       Annotation boundaries dashed overlay on bottom panels.

Usage:
    python visualize_prediction_wsi.py               # all folds' val slides
    python visualize_prediction_wsi.py 6             # fold 6 only
    python visualize_prediction_wsi.py 3 6           # folds 3 and 6
    python visualize_prediction_wsi.py --slide S14-1255-1-3 --ckpt best_model_fold6.pth
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from lxml import etree

from config import Config
from model import create_model
from stain_normalizer import MacenkoNormalizer


THUMB_MAX_DIM = 3000
HEATMAP_ALPHA = 0.55
CLASS_COLOR = {"gland": (0.15, 0.40, 0.95), "non-gland": (0.90, 0.25, 0.20)}
DEFAULT_WSI_STRIDE = 512  # no overlap → ~4× faster than stride=256 with same coverage


# ───────────────────────────────────────────────────────────────
# Worker: per-process slide handle + Macenko instance
# ───────────────────────────────────────────────────────────────
_W_SLIDE = None
_W_NORM = None
_W_PARAMS = None


def _worker_init(svs_path, target_rgb, params):
    global _W_SLIDE, _W_NORM, _W_PARAMS
    _W_SLIDE = openslide.OpenSlide(svs_path)
    _W_PARAMS = params
    if target_rgb is not None:
        n = MacenkoNormalizer()
        n.fit(target_rgb)
        _W_NORM = n
    else:
        _W_NORM = None


def _is_tissue(patch_rgb, threshold=0.7):
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    return (sat > 20).mean() >= threshold


def _process_position(ix_iy):
    """Read patch at (ix, iy), tissue filter, Macenko; return (x, y, patch224_uint8) or None."""
    ix, iy = ix_iy
    p = _W_PARAMS
    patch_size = p["patch_size"]
    stride = p["stride"]

    abs_x = ix * stride
    abs_y = iy * stride

    # Keep positions that fit entirely within slide bounds
    if abs_x + patch_size > p["slide_w"] or abs_y + patch_size > p["slide_h"]:
        return None

    img = _W_SLIDE.read_region((abs_x, abs_y), p["level"], (patch_size, patch_size))
    patch_rgb = np.array(img.convert("RGB"))

    if not _is_tissue(patch_rgb, p["tissue_threshold"]):
        return None

    if _W_NORM is not None:
        try:
            patch_rgb = _W_NORM.transform(patch_rgb)
        except Exception:
            return None

    # Resize to input_size for model
    if patch_rgb.shape[0] != p["input_size"]:
        patch_rgb = cv2.resize(patch_rgb, (p["input_size"], p["input_size"]),
                               interpolation=cv2.INTER_AREA)
    return (abs_x, abs_y, patch_rgb.astype(np.uint8))


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def parse_aperio_xml(xml_path):
    tree = etree.parse(xml_path)
    positive, negative = [], []
    for ann in tree.getroot().findall(".//Annotation"):
        for reg in ann.findall(".//Region"):
            v = [(float(x.get("X")), float(x.get("Y"))) for x in reg.findall(".//Vertex")]
            if not v:
                continue
            poly = np.array(v, dtype=np.float64)
            (negative if reg.get("NegativeROA", "0") == "1" else positive).append(poly)
    return positive, negative


def get_thumbnail(slide, max_dim):
    w, h = slide.level_dimensions[0]
    scale = max(w, h) / max_dim
    thumb = slide.get_thumbnail((int(w/scale), int(h/scale)))
    return np.array(thumb.convert("RGB")), scale


def class_suffix(cls):
    return "_G" if cls == "gland" else "_S"


def draw_polygons(ax, positive, negative, scale, color, solid=True):
    for poly in positive:
        if solid:
            ax.fill(poly[:,0]/scale, poly[:,1]/scale, color=color, alpha=0.2, linewidth=0)
        ax.plot(poly[:,0]/scale, poly[:,1]/scale, color=color, linewidth=1.0,
                linestyle="-" if solid else "--")
    for poly in negative:
        ax.plot(poly[:,0]/scale, poly[:,1]/scale, color=color, linewidth=0.8, linestyle="--")


# ───────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────
def imagenet_normalize(batch_uint8):
    """(B, H, W, 3) uint8 RGB → (B, 3, H, W) float32 normalized."""
    x = batch_uint8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (0, 3, 1, 2))  # BHWC → BCHW


@torch.no_grad()
def run_inference_on_slide(svs_path, target_rgb, config, model, device, workers, stride, verbose=True):
    """Scan slide, return list of (x, y, probs_gland)."""
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.level_dimensions[0]
    slide.close()  # each worker opens its own

    patch_size = config.patch_size
    n_x = (w - patch_size) // stride + 1
    n_y = (h - patch_size) // stride + 1
    positions = [(ix, iy) for iy in range(n_y) for ix in range(n_x)]
    if verbose:
        print(f"  Grid {n_x}×{n_y} = {len(positions):,} positions "
              f"(stride {stride}, patch {patch_size})")

    params = {
        "patch_size": patch_size,
        "stride": stride,
        "slide_w": w, "slide_h": h,
        "level": config.extraction_level,
        "tissue_threshold": config.tissue_threshold,
        "input_size": config.input_size,
    }

    # Multiprocess read + Macenko
    ctx = mp.get_context("spawn")
    batch_buf = []
    batch_coords = []
    batch_size = max(32, config.batch_size // 2)  # smaller batch for viz
    all_x, all_y, all_p = [], [], []

    t0 = time.time()
    with ctx.Pool(
        processes=max(1, workers),
        initializer=_worker_init,
        initargs=(svs_path, target_rgb, params),
    ) as pool:
        chunksize = max(1, len(positions) // (max(1, workers) * 16))
        n_tissue = 0
        for result in pool.imap_unordered(_process_position, positions, chunksize=chunksize):
            if result is None:
                continue
            abs_x, abs_y, patch_rgb = result
            batch_buf.append(patch_rgb)
            batch_coords.append((abs_x, abs_y))
            n_tissue += 1
            if len(batch_buf) >= batch_size:
                arr = np.stack(batch_buf)
                x = torch.from_numpy(imagenet_normalize(arr)).to(device, non_blocking=True)
                probs = torch.softmax(model(x), dim=1).cpu().numpy()
                for (xi, yi), p in zip(batch_coords, probs):
                    all_x.append(xi); all_y.append(yi); all_p.append(p[0])  # prob_gland
                batch_buf.clear(); batch_coords.clear()
        # tail
        if batch_buf:
            arr = np.stack(batch_buf)
            x = torch.from_numpy(imagenet_normalize(arr)).to(device, non_blocking=True)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            for (xi, yi), p in zip(batch_coords, probs):
                all_x.append(xi); all_y.append(yi); all_p.append(p[0])

    dt = time.time() - t0
    if verbose:
        print(f"  tissue patches: {n_tissue:,} / {len(positions):,}  "
              f"({n_tissue / max(1, len(positions)) * 100:.1f}%)  in {dt:.1f}s")
    return np.array(all_x), np.array(all_y), np.array(all_p)


def build_prob_map_wsi(xs, ys, probs_gland, H, W, patch_size, scale):
    """Pixel-level average of P(gland) over thumbnail."""
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for x_src, y_src, p in zip(xs, ys, probs_gland):
        tx0 = int(x_src / scale); ty0 = int(y_src / scale)
        tx1 = min(int((x_src + patch_size) / scale), W)
        ty1 = min(int((y_src + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += p
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    return np.ma.masked_where(count == 0, avg)


def build_hard_pred_map(xs, ys, probs_gland, H, W, patch_size, scale):
    """Same layout as prob map but values are the hard predicted class probability
    (i.e., max(p, 1-p)) signed so gland→positive (blue), non-gland→negative (red).
    We encode: value ∈ [-1, 1] where +1 = fully confident gland, -1 = fully confident non-gland.
    """
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for x_src, y_src, p in zip(xs, ys, probs_gland):
        signed = (2.0 * p - 1.0)  # p=1 → +1 (gland), p=0 → -1 (non-gland), p=0.5 → 0
        tx0 = int(x_src / scale); ty0 = int(y_src / scale)
        tx1 = min(int((x_src + patch_size) / scale), W)
        ty1 = min(int((y_src + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += signed
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    return np.ma.masked_where(count == 0, avg)


# ───────────────────────────────────────────────────────────────
# Visualization
# ───────────────────────────────────────────────────────────────
def visualize_slide(fold, slide_name, ckpt_path, config, device, out_dir, stride):
    info = config.slides[slide_name]
    val_class = info["class"]
    suffix = class_suffix(val_class)

    print(f"\n[Fold {fold}] slide={slide_name} ({val_class})  ckpt={ckpt_path.name}")

    svs_path = str(Path(config.svs_dir) / info["svs"])
    xml_path = str(Path(config.xml_dir) / info["xml"])
    if not Path(svs_path).exists():
        print(f"  SVS not found: {svs_path}"); return

    # Stain target
    target_rgb = None
    if config.stain_normalize:
        t = cv2.imread(config.stain_target_path)
        if t is not None:
            target_rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        else:
            print(f"  WARN: stain_target not found at {config.stain_target_path}; "
                  f"proceeding WITHOUT normalization.")

    # Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_backbone = ckpt.get("backbone", config.backbone)
    model = create_model(num_classes=config.num_classes, pretrained=False,
                         backbone=ckpt_backbone,
                         head_type=getattr(config, "head_type", "linear"))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Inference over full slide
    xs, ys, probs_gland = run_inference_on_slide(
        svs_path, target_rgb, config, model, device,
        workers=config.extract_workers, stride=stride,
    )
    if len(xs) == 0:
        print("  No tissue patches; skip."); return

    frac_gland = (probs_gland >= 0.5).mean()
    print(f"  WSI inference: {len(xs):,} tissue patches; "
          f"predicted gland fraction = {frac_gland:.2%}")

    # Thumbnail + annotation
    slide = openslide.OpenSlide(svs_path)
    thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
    slide.close()
    positive, negative = parse_aperio_xml(xml_path)
    ann_color = CLASS_COLOR[val_class]

    H, W = thumb_rgb.shape[:2]
    prob_map      = build_prob_map_wsi(xs, ys, probs_gland, H, W, config.patch_size, scale)
    hard_pred_map = build_hard_pred_map(xs, ys, probs_gland, H, W, config.patch_size, scale)

    # ── 2×2 figure ──
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))

    ax = axes[0, 0]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{slide_name} — raw thumbnail ({W}×{H} px)")

    ax = axes[0, 1]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    draw_polygons(ax, positive, negative, scale, ann_color, solid=True)
    ax.set_title(f"ground-truth annotation (class: {val_class})\n"
                 f"{len(positive)} positive + {len(negative)} negative regions")
    ann_handles = [
        mpatches.Patch(facecolor=ann_color, alpha=0.2, edgecolor=ann_color,
                       label=f"positive ({val_class})"),
        mpatches.Patch(facecolor="none", edgecolor=ann_color, linestyle="--",
                       label="negative (exclude)"),
    ]
    ax.legend(handles=ann_handles, loc="lower right", fontsize=9)

    # (1,0) P(gland) heatmap — whole tissue
    ax = axes[1, 0]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    im_prob = ax.imshow(prob_map, cmap="RdBu", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
    draw_polygons(ax, positive, negative, scale, "#333333", solid=False)
    ax.set_title(f"WSI-wide P(gland)  (blue=gland, red=non-gland)\n"
                 f"all-tissue inference, N={len(xs):,}, "
                 f"predicted-gland fraction = {frac_gland:.2%}")
    cbar = fig.colorbar(im_prob, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(gland)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # (1,1) hard prediction map — signed confidence (-1..+1) on same RdBu palette
    ax = axes[1, 1]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    im_hard = ax.imshow(hard_pred_map, cmap="RdBu", alpha=HEATMAP_ALPHA, vmin=-1, vmax=1)
    draw_polygons(ax, positive, negative, scale, "#333333", solid=False)
    ax.set_title(f"hard prediction map  (signed confidence)\n"
                 f"blue = confident gland, red = confident non-gland, white = undecided")
    cbar = fig.colorbar(im_hard, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("2·P(gland) − 1")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.suptitle(f"Fold {fold} WSI-wide inference — {slide_name} "
                 f"(true class = {val_class}, stride={stride})",
                 fontsize=15, y=0.995)
    out_path = out_dir / f"fold{fold}_{slide_name}{suffix}_{ckpt_backbone}_wsi.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folds", nargs="*", type=int, default=[],
                        help="fold numbers to process (default: all)")
    parser.add_argument("--slide", type=str, default=None,
                        help="run on a specific slide (requires --ckpt)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint filename under checkpoints/ (used with --slide)")
    parser.add_argument("--stride", type=int, default=DEFAULT_WSI_STRIDE,
                        help=f"WSI sliding-window stride (default {DEFAULT_WSI_STRIDE})")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = config.viz_dir_for("Prediction_WSI")
    ckpt_dir = Path(config.checkpoint_dir)

    if args.slide and args.ckpt:
        ckpt_path = ckpt_dir / args.ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        visualize_slide(ckpt.get("fold", -1), args.slide, ckpt_path,
                        config, device, out_dir, args.stride)
        return

    if args.folds:
        ckpt_paths = [ckpt_dir / f"best_model_fold{f}.pth" for f in args.folds]
    else:
        ckpt_paths = sorted(ckpt_dir.glob("best_model_fold*.pth"))

    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            print(f"  skip (not found): {ckpt_path}"); continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        visualize_slide(ckpt["fold"], ckpt["val_slide"], ckpt_path,
                        config, device, out_dir, args.stride)


if __name__ == "__main__":
    main()
