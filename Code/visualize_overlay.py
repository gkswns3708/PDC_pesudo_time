"""
Overlay Aperio XML annotations on SVS slide thumbnails.

Reads:
  /app/Gland_Seg/Data/S14/SVS/<slide>.svs
  /app/Gland_Seg/Data/S14/Annotation/<slide>[_S|_G].xml
Writes:
  /app/Gland_Seg/Data/S14/Overlay/<slide>_overlay.png

Usage:
    python visualize_overlay.py                 # all slides
    python visualize_overlay.py S14-177-1-5     # single slide
    python visualize_overlay.py S14-177-1-5 S14-252-3
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
from lxml import etree


SVS_DIR = Path("/app/Gland_Seg/Data/S14/SVS")
XML_DIR = Path("/app/Gland_Seg/Data/S14/Annotation")
OUT_DIR = Path("/app/Gland_Seg/Data/S14/Overlay")

THUMB_MAX_DIM = 2000  # thumbnail long-side pixels
POSITIVE_COLOR = (0.15, 0.40, 0.95)   # blue — positive region (annotated tissue)
NEGATIVE_COLOR = (0.90, 0.25, 0.20)   # red  — negative ROA (excluded hole)


def parse_aperio_xml(xml_path):
    """Return (positive_polys, negative_polys); each is a list of Nx2 float arrays."""
    tree = etree.parse(str(xml_path))
    positive, negative = [], []
    for annotation in tree.getroot().findall(".//Annotation"):
        for region in annotation.findall(".//Region"):
            verts = [(float(v.get("X")), float(v.get("Y")))
                     for v in region.findall(".//Vertex")]
            if len(verts) < 3:
                continue
            poly = np.array(verts, dtype=np.float64)
            if region.get("NegativeROA", "0") == "1":
                negative.append(poly)
            else:
                positive.append(poly)
    return positive, negative


def get_thumbnail(slide, max_dim):
    """Return RGB uint8 thumbnail and the scale factor (level-0 px / thumb px)."""
    w, h = slide.level_dimensions[0]
    scale = max(w, h) / max_dim
    thumb_size = (int(w / scale), int(h / scale))
    thumb = slide.get_thumbnail(thumb_size)
    return np.array(thumb.convert("RGB")), scale


def find_xml(slide_stem):
    """Match <stem>.xml or <stem>_S.xml or <stem>_G.xml in XML_DIR."""
    for cand in (f"{slide_stem}.xml", f"{slide_stem}_S.xml", f"{slide_stem}_G.xml"):
        p = XML_DIR / cand
        if p.exists():
            return p
    return None


def polygon_area_px(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def draw_polys(ax, polys, scale, *, face=None, edge, linestyle="-", linewidth=1.2, alpha=0.35):
    for poly in polys:
        xs = poly[:, 0] / scale
        ys = poly[:, 1] / scale
        if face is not None:
            ax.fill(xs, ys, color=face, alpha=alpha, linewidth=0)
        ax.plot(xs, ys, color=edge, linestyle=linestyle, linewidth=linewidth)


def visualize_one(svs_path):
    slide_stem = svs_path.stem
    xml_path = find_xml(slide_stem)
    if xml_path is None:
        print(f"  [{slide_stem}] XML not found in {XML_DIR}. Skip.")
        return

    slide = openslide.OpenSlide(str(svs_path))
    try:
        thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
        slide_w, slide_h = slide.level_dimensions[0]
        mpp = float(slide.properties.get("openslide.mpp-x", 0.2521) or 0.2521)
    finally:
        slide.close()

    positive, negative = parse_aperio_xml(xml_path)

    # Net annotated area in mm² (positive minus negative)
    pos_px = sum(polygon_area_px(p) for p in positive)
    neg_px = sum(polygon_area_px(p) for p in negative)
    net_mm2 = max(0.0, (pos_px - neg_px) * (mpp / 1000.0) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: raw thumbnail
    axes[0].imshow(thumb_rgb)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(f"{slide_stem} — raw\n{slide_w}×{slide_h} px @ {mpp:.4f} μm/px",
                      fontsize=11)

    # Right: overlay
    axes[1].imshow(thumb_rgb)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    draw_polys(axes[1], positive, scale,
               face=POSITIVE_COLOR, edge=POSITIVE_COLOR, linewidth=1.2, alpha=0.35)
    draw_polys(axes[1], negative, scale,
               face=None, edge=NEGATIVE_COLOR, linestyle="--", linewidth=1.0)
    axes[1].set_title(
        f"{slide_stem} — overlay ({xml_path.name})\n"
        f"{len(positive)} positive + {len(negative)} negative regions, "
        f"~{net_mm2:.2f} mm²",
        fontsize=11)

    handles = [
        mpatches.Patch(facecolor=POSITIVE_COLOR, alpha=0.35, edgecolor=POSITIVE_COLOR,
                       label="positive region"),
        mpatches.Patch(facecolor="none", edgecolor=NEGATIVE_COLOR, linestyle="--",
                       label="negative ROA (excluded)"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{slide_stem}_overlay.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"  [{slide_stem}] pos={len(positive):>2} neg={len(negative):>2}  "
          f"area~{net_mm2:.2f}mm²  →  {out_path.name}")


def main():
    targets = sys.argv[1:]
    if targets:
        svs_paths = [SVS_DIR / f"{t}.svs" for t in targets]
    else:
        svs_paths = sorted(SVS_DIR.glob("*.svs"))

    if not svs_paths:
        print(f"No SVS files found under {SVS_DIR}.")
        return

    print(f"Overlaying {len(svs_paths)} slide(s) →  {OUT_DIR}")
    for svs_path in svs_paths:
        if not svs_path.exists():
            print(f"  [skip] {svs_path.name} not found")
            continue
        visualize_one(svs_path)


if __name__ == "__main__":
    main()
