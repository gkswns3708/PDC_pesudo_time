"""
Produce overlay visualizations of the METU-CCTGS downsampled dataset to share
with the supervising pathologist. For each annotated image (train+validation
splits, 87 total) generate a 3-panel PNG:

  | raw image | annotation (color) | raw + annotation alpha-blend |

with a legend mapping the 5 tissue classes to their canonical METU colors.
A top-level summary page lists which slides contain Grade-3 (PDC) regions —
the class most directly relevant to our gland / non-gland (PDC) study.

Outputs:
  /app/metucctgs_ds/overlays/<split>/<basename>.png   (per-image)
  /app/metucctgs_ds/overlays/summary_grade3.png       (Grade-3 contact sheet)
  /app/metucctgs_ds/overlays/class_distribution.csv

Run:
    python make_metucctgs_overlays.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("/app/metucctgs_ds")
OUT = ROOT / "overlays"

CLASS_RGB = {
    0: ((0, 0, 0), "Others"),
    1: ((0, 192, 0), "Tumor Grade-1 (well diff.)"),
    2: ((255, 224, 32), "Tumor Grade-2 (mod. diff.)"),
    3: ((255, 0, 0), "Tumor Grade-3 (poorly diff. / PDC)"),
    4: ((0, 32, 255), "Normal Mucosa"),
}


def colorize(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, (rgb, _) in CLASS_RGB.items():
        out[mask == cid] = rgb
    return out


def legend_handles():
    return [
        mpatches.Patch(facecolor=np.array(rgb) / 255.0,
                       edgecolor="black", linewidth=0.4, label=name)
        for cid, (rgb, name) in CLASS_RGB.items()
    ]


def render_one(img_path, ann_path, out_path, title):
    img = np.asarray(Image.open(img_path).convert("RGB"))
    mask = np.asarray(Image.open(ann_path))
    color = colorize(mask)
    blend = (0.55 * img + 0.45 * color).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img); axes[0].set_title("Raw H&E")
    axes[1].imshow(color); axes[1].set_title("METU annotation")
    axes[2].imshow(blend); axes[2].set_title("Overlay")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    classes_present = sorted(int(c) for c in np.unique(mask))
    cls_str = ", ".join(CLASS_RGB[c][1].split(" ")[0] + CLASS_RGB[c][1].split(" ")[1][-2:] if False
                        else f"{c}:{CLASS_RGB[c][1].split(' (')[0]}" for c in classes_present)
    fig.suptitle(f"{title}  |  classes: {cls_str}", fontsize=12)
    fig.legend(handles=legend_handles(), loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return classes_present


def contact_sheet(items, out_path, title, cols=4):
    n = len(items)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.4))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    for i, (img_path, ann_path, name) in enumerate(items):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(ann_path))
        blend = (0.55 * img + 0.45 * colorize(mask)).astype(np.uint8)
        ax.imshow(blend); ax.set_title(name, fontsize=9); ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.legend(handles=legend_handles(), loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    grade3_items = []
    for split in ("train", "validation"):
        img_dir = ROOT / "images" / split
        ann_dir = ROOT / "annotations" / split
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(os.listdir(ann_dir))
        print(f"[{split}] {len(files)} images")
        for f in files:
            ip = img_dir / f
            ap = ann_dir / f
            op = out_dir / f
            classes = render_one(ip, ap, op, title=f"{split}/{f}")
            row = {"split": split, "file": f}
            row.update({f"has_{cid}_{CLASS_RGB[cid][1].split(' (')[0].replace(' ','_')}": int(cid in classes)
                        for cid in CLASS_RGB})
            rows.append(row)
            if 3 in classes:
                grade3_items.append((ip, ap, f"{split}/{f}"))
    df = pd.DataFrame(rows)
    csv_path = OUT / "class_distribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  ({len(df)} rows)")

    if grade3_items:
        contact_sheet(grade3_items,
                      OUT / "summary_grade3.png",
                      f"All slides containing Tumor Grade-3 (PDC) — n={len(grade3_items)}",
                      cols=4)
        print(f"[save] {OUT/'summary_grade3.png'}  ({len(grade3_items)} grade-3 slides)")


if __name__ == "__main__":
    main()
