"""
Visualize model predictions on a slide as a spatial heatmap.

For each fold's best_model_fold{N}.pth, runs inference on the val slide and
produces a 3-panel figure:
    1. Slide thumbnail (raw)
    2. Thumbnail + XML annotation polygons (ground truth regions)
    3. Thumbnail + patch-level predictions as colored squares
         gland pred    → blue   (intensity = prob_gland)
         non-gland pred → red   (intensity = prob_non_gland)
         border color (green/red) = correct / wrong vs ground truth

Also produces a grid of:
    - top-20 most-confident CORRECT patches
    - top-20 most-confident WRONG patches

Output:
    Gland_Seg/Viz/Prediction_Viz/fold{N}_{slide}_{S|G}_heatmap.png
    Gland_Seg/Viz/Prediction_Viz/fold{N}_{slide}_{S|G}_samples.png

Usage:
    python visualize_prediction.py                  # all folds
    python visualize_prediction.py 4                # fold 4 only
    python visualize_prediction.py 4 6              # folds 4 and 6
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from lxml import etree
from torch.utils.data import DataLoader

from config import Config
from dataset import PatchDataset, get_val_transforms
from model import create_model


THUMB_MAX_DIM = 3000
N_SAMPLES = 20
CLASS_COLOR = {"gland": (0.15, 0.40, 0.95), "non-gland": (0.90, 0.25, 0.20)}
# Class index 0 = gland, 1 = non-gland
PRED_COLOR = {0: np.array([0.15, 0.40, 0.95]),   # gland pred → blue
              1: np.array([0.90, 0.25, 0.20])}   # non-gland pred → red
HEATMAP_ALPHA = 0.55


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


@torch.no_grad()
def infer(model, dataset, device, batch_size=512, num_workers=4):
    """Return probs (N,2), preds (N,), labels (N,)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_probs, all_labels = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    return probs, preds, labels


def draw_polygons(ax, positive, negative, scale, color):
    for poly in positive:
        ax.fill(poly[:,0]/scale, poly[:,1]/scale, color=color, alpha=0.25, linewidth=0)
        ax.plot(poly[:,0]/scale, poly[:,1]/scale, color=color, linewidth=1.0)
    for poly in negative:
        ax.plot(poly[:,0]/scale, poly[:,1]/scale, color=color, linewidth=0.8, linestyle="--")


def build_prob_map(meta_df, probs, H, W, patch_size, scale):
    """Accumulate per-pixel average of P(gland). Overlapping patches are averaged.

    Returns a masked float32 array (H, W) with NaN/masked where no patch covers.
    """
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    xy = meta_df[["x", "y"]].to_numpy()
    for (x_src, y_src), p in zip(xy, probs):
        tx0 = int(x_src / scale); ty0 = int(y_src / scale)
        tx1 = min(int((x_src + patch_size) / scale), W)
        ty1 = min(int((y_src + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += p[0]   # prob_gland
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    return np.ma.masked_where(count == 0, avg)


def build_correctness_map(meta_df, preds, labels, H, W, patch_size, scale):
    """Per-pixel average correctness (1.0 = correct, 0.0 = wrong)."""
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    xy = meta_df[["x", "y"]].to_numpy()
    correct = (preds == labels).astype(np.float32)
    for (x_src, y_src), c in zip(xy, correct):
        tx0 = int(x_src / scale); ty0 = int(y_src / scale)
        tx1 = min(int((x_src + patch_size) / scale), W)
        ty1 = min(int((y_src + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += c
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    return np.ma.masked_where(count == 0, avg)


def plot_sample_grid(patch_paths, titles, n_cols, main_title, out_path):
    n = len(patch_paths)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.2, n_rows*2.4))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if i >= n:
            ax.axis("off"); continue
        img = cv2.imread(str(patch_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(titles[i], fontsize=7)
    plt.suptitle(main_title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def class_suffix(cls):
    return "_G" if cls == "gland" else "_S"


def visualize_fold(fold, ckpt_path, config, device, out_dir):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    val_slide = ckpt["val_slide"]
    val_class = config.slides[val_slide]["class"]
    suffix = class_suffix(val_class)

    print(f"\n[Fold {fold}] val={val_slide} ({val_class})  ckpt epoch={ckpt['epoch']}  "
          f"saved_val_f1={ckpt.get('val_f1', float('nan')):.4f}")

    # Build dataset preserving patch order matching metadata
    val_dataset = PatchDataset(
        config.output_dir, [val_slide], config.slides,
        transform=get_val_transforms(config.input_size),
    )
    if len(val_dataset) == 0:
        print("  skip (empty)"); return

    # Patch path → (x, y) from filename pattern: <slide>_<x>_<y>.png
    meta_rows = []
    for p, _ in val_dataset.samples:
        stem = p.stem  # e.g. "S14-248-1-3_12345_67890"
        parts = stem.rsplit("_", 2)
        x, y = int(parts[-2]), int(parts[-1])
        meta_rows.append({"path": p, "x": x, "y": y})
    meta_df = pd.DataFrame(meta_rows)

    # Inference
    ckpt_backbone = ckpt.get("backbone", config.backbone)
    model = create_model(num_classes=config.num_classes, pretrained=False,
                         backbone=ckpt_backbone,
                         head_type=getattr(config, "head_type", "linear"))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    probs, preds, labels = infer(model, val_dataset, device,
                                 batch_size=config.batch_size, num_workers=config.num_workers)

    acc = (preds == labels).mean()
    n_correct = int((preds == labels).sum()); n_wrong = int(len(preds) - n_correct)
    print(f"  N={len(preds)}  correct={n_correct}  wrong={n_wrong}  acc={acc:.4f}")

    # Slide thumbnail + annotation
    info = config.slides[val_slide]
    svs_path = str(Path(config.svs_dir) / info["svs"])
    xml_path = str(Path(config.xml_dir) / info["xml"])
    slide = openslide.OpenSlide(svs_path)
    thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
    slide.close()
    positive, negative = parse_aperio_xml(xml_path)
    ann_color = CLASS_COLOR[val_class]

    # ── Heatmap figure (2x2) ──
    H, W = thumb_rgb.shape[:2]
    prob_map    = build_prob_map(meta_df, probs, H, W, config.patch_size, scale)
    correct_map = build_correctness_map(meta_df, preds, labels, H, W,
                                        config.patch_size, scale)

    fig, axes = plt.subplots(2, 2, figsize=(17, 14))

    # (0,0) raw
    ax = axes[0, 0]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{val_slide} — raw thumbnail ({W}×{H} px)")

    # (0,1) annotation overlay
    ax = axes[0, 1]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    draw_polygons(ax, positive, negative, scale, ann_color)
    ax.set_title(f"ground-truth annotation (class: {val_class})\n"
                 f"{len(positive)} positive + {len(negative)} negative regions")
    ann_handles = [
        mpatches.Patch(facecolor=ann_color, alpha=0.25, edgecolor=ann_color,
                       label=f"positive ({val_class})"),
        mpatches.Patch(facecolor="none", edgecolor=ann_color, linestyle="--",
                       label="negative (exclude)"),
    ]
    ax.legend(handles=ann_handles, loc="lower right", fontsize=9)

    # (1,0) probability heatmap: P(gland). Blue = gland, Red = non-gland
    ax = axes[1, 0]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    im_prob = ax.imshow(prob_map, cmap="RdBu", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
    ax.set_title(f"prediction heatmap — P(gland)  "
                 f"(blue=gland, red=non-gland)\n"
                 f"overall acc = {acc:.3f}  (N={len(preds)})")
    cbar = fig.colorbar(im_prob, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(gland)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # (1,1) correctness heatmap: green=correct, red=wrong
    ax = axes[1, 1]
    ax.imshow(thumb_rgb); ax.set_xticks([]); ax.set_yticks([])
    im_corr = ax.imshow(correct_map, cmap="RdYlGn", alpha=HEATMAP_ALPHA, vmin=0, vmax=1)
    ax.set_title(f"correctness heatmap  "
                 f"(green=correct, red=wrong)\n"
                 f"correct {n_correct} / wrong {n_wrong}")
    cbar = fig.colorbar(im_corr, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("fraction correct")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.suptitle(f"Fold {fold} — {val_slide} (true class = {val_class})",
                 fontsize=15, y=0.995)
    heatmap_out = out_dir / f"fold{fold}_{val_slide}{suffix}_{ckpt_backbone}_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  heatmap → {heatmap_out.name}")

    # ── Sample grid: top-N confident correct vs wrong ──
    probs_pred = probs[np.arange(len(preds)), preds]  # confidence of the predicted class
    correct_mask = (preds == labels)
    # Most-confident correct
    correct_idx = np.where(correct_mask)[0]
    wrong_idx   = np.where(~correct_mask)[0]
    top_correct = correct_idx[np.argsort(probs_pred[correct_idx])[-N_SAMPLES:][::-1]]
    top_wrong   = wrong_idx[np.argsort(probs_pred[wrong_idx])[-N_SAMPLES:][::-1]]

    def make_titles(idx):
        t = []
        for i in idx:
            pred_name = config.class_names[preds[i]]
            t.append(f"pred={pred_name}\np_gland={probs[i,0]:.2f}")
        return t

    if len(top_correct) > 0 or len(top_wrong) > 0:
        # Combine both into one big figure (2 rows of N_SAMPLES each)
        n_cols = max(len(top_correct), len(top_wrong), 1)
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols*2.2, 5.2))
        if n_cols == 1:
            axes = np.array(axes).reshape(2, 1)

        for j, row_idx, row_title in [(0, top_correct, f"Most-confident CORRECT (acc={acc:.3f})"),
                                      (1, top_wrong,   "Most-confident WRONG")]:
            for col in range(n_cols):
                ax = axes[j, col]; ax.set_xticks([]); ax.set_yticks([])
                if col >= len(row_idx):
                    ax.axis("off"); continue
                i = row_idx[col]
                img = cv2.imread(str(meta_df.iloc[i]["path"]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                pred_name = config.class_names[preds[i]]
                ax.set_title(f"pred={pred_name}\np_g={probs[i,0]:.2f}", fontsize=7)
            axes[j, 0].set_ylabel(row_title, fontsize=10, rotation=90, labelpad=10)

        plt.suptitle(f"Fold {fold} — {val_slide} (true class={val_class})",
                     fontsize=13)
        samples_out = out_dir / f"fold{fold}_{val_slide}{suffix}_{ckpt_backbone}_samples.png"
        plt.tight_layout()
        plt.savefig(samples_out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  samples → {samples_out.name}")


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = config.viz_dir_for("Prediction_Viz")

    ckpt_dir = Path(config.checkpoint_dir)
    args = sys.argv[1:]
    bb = config.backbone

    def _resolve_ckpt(fold_n):
        # Prefer backbone-tagged: best_model_{bb}_fold{N}.pth, fall back to legacy
        cand = ckpt_dir / f"best_model_{bb}_fold{fold_n}.pth"
        if cand.exists():
            return cand
        return ckpt_dir / f"best_model_fold{fold_n}.pth"

    if args:
        folds = [int(a) for a in args]
        ckpt_paths = [_resolve_ckpt(f) for f in folds]
    else:
        ckpt_paths = sorted(ckpt_dir.glob(f"best_model_{bb}_fold*.pth"))
        if not ckpt_paths:
            ckpt_paths = sorted(ckpt_dir.glob("best_model_fold*.pth"))

    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            print(f"  skip (not found): {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        visualize_fold(ckpt["fold"], ckpt_path, config, device, out_dir)


if __name__ == "__main__":
    main()
