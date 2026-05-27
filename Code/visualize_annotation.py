"""
Visualize SVS slide thumbnail with XML annotation polygons overlaid.

Usage:
    python visualize_annotation.py                 # all 8 slides
    python visualize_annotation.py S14-177-1-5     # single slide
    python visualize_annotation.py S14-177-1-5 S14-252-3

Output: Gland_Seg/Viz/annotation_<slide>.png per slide (grid view).

Each image shows:
  - Left : full slide thumbnail (low-res)
  - Right: same with positive polygons (filled) + negative polygons (outlined)
Class color: gland=blue, non-gland(solid)=red.
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
from lxml import etree

from config import Config


THUMB_MAX_DIM = 2000  # thumbnail long-side pixels
CLASS_COLOR = {"gland": (0.15, 0.40, 0.95), "non-gland": (0.90, 0.25, 0.20)}


def parse_aperio_xml(xml_path):
    tree = etree.parse(xml_path)
    positive, negative = [], []
    for annotation in tree.getroot().findall(".//Annotation"):
        for region in annotation.findall(".//Region"):
            verts = [(float(v.get("X")), float(v.get("Y")))
                     for v in region.findall(".//Vertex")]
            if not verts:
                continue
            poly = np.array(verts, dtype=np.float64)
            if region.get("NegativeROA", "0") == "1":
                negative.append(poly)
            else:
                positive.append(poly)
    return positive, negative


def get_thumbnail(slide, max_dim):
    """Return RGB uint8 thumbnail and the scale factor (slide-level px / thumb-level px)."""
    w, h = slide.level_dimensions[0]
    scale = max(w, h) / max_dim
    thumb_size = (int(w / scale), int(h / scale))
    thumb = slide.get_thumbnail(thumb_size)
    thumb_rgb = np.array(thumb.convert("RGB"))
    return thumb_rgb, scale


def plot_slide(ax, thumb_rgb, positive, negative, scale, color, title):
    ax.imshow(thumb_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    for poly in positive:
        xs = poly[:, 0] / scale
        ys = poly[:, 1] / scale
        ax.fill(xs, ys, color=color, alpha=0.35, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=1.2)

    for poly in negative:
        xs = poly[:, 0] / scale
        ys = poly[:, 1] / scale
        ax.plot(xs, ys, color=color, linewidth=1.0, linestyle="--")


def class_suffix(class_name):
    """gland → _G, non-gland (solid) → _S."""
    return "_G" if class_name == "gland" else "_S"


def visualize_one(slide_name, info, config, out_dir):
    svs_path = Path(config.svs_dir) / info["svs"]
    xml_path = Path(config.xml_dir) / info["xml"]

    if not svs_path.exists():
        print(f"  [{slide_name}] SVS not found: {svs_path}. Skip.")
        return
    if not xml_path.exists():
        print(f"  [{slide_name}] XML not found: {xml_path}. Skip.")
        return

    slide = openslide.OpenSlide(str(svs_path))
    thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
    positive, negative = parse_aperio_xml(str(xml_path))
    slide_w, slide_h = slide.level_dimensions[0]
    slide.close()

    # Total annotated area (in μm² approx)
    mpp = 0.2521
    total_px = 0
    for poly in positive:
        # shoelace
        x = poly[:, 0]; y = poly[:, 1]
        area_px = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        total_px += area_px
    for poly in negative:
        x = poly[:, 0]; y = poly[:, 1]
        area_px = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        total_px -= area_px
    total_mm2 = max(0, total_px * (mpp / 1000) ** 2)

    color = CLASS_COLOR[info["class"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_slide(axes[0], thumb_rgb, [], [], scale, color,
               f"{slide_name}  —  raw thumbnail\nslide {slide_w}×{slide_h} px")
    plot_slide(axes[1], thumb_rgb, positive, negative, scale, color,
               f"{slide_name}  —  class: {info['class']} (XML: {info['xml']})\n"
               f"{len(positive)} positive + {len(negative)} negative region(s), "
               f"~{total_mm2:.2f} mm²")

    # Legend
    handles = [
        mpatches.Patch(facecolor=color, alpha=0.35, edgecolor=color,
                       label=f"positive ({info['class']})"),
        mpatches.Patch(facecolor="none", edgecolor=color, linestyle="--",
                       label="negative (exclude hole)"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9)

    out_path = out_dir / f"annotation_{slide_name}{class_suffix(info['class'])}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [{slide_name}] class={info['class']:<10} "
          f"pos={len(positive):>2} neg={len(negative):>2}  "
          f"area~{total_mm2:.2f}mm²  →  {out_path.name}")


def main():
    config = Config()
    out_dir = config.viz_dir_for("Annotation_Viz")

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(config.slides.keys())
    # Also expose external_test_slides for inspection
    ext = getattr(config, "external_test_slides", {})
    for slide_name in targets:
        if slide_name in config.slides:
            info = config.slides[slide_name]
        elif slide_name in ext:
            # External test slide — class unknown; treat as 'non-gland' for color/suffix only
            info = {**ext[slide_name], "class": "non-gland", "label": 1}
            print(f"  [{slide_name}] EXTERNAL evaluation slide (no class label)")
        else:
            print(f"  [skip] {slide_name} not in config.slides nor external_test_slides")
            continue
        visualize_one(slide_name, info, config, out_dir)


if __name__ == "__main__":
    main()
