"""
Convert pixel-level prediction map(s) to Aperio ImageScope XML.

For a given source (single model / hard-vote / mean-prob ensemble), reads the
corresponding pixel map at thumbnail resolution + slide_meta.npy and produces
an XML with two Annotation groups:
  - Annotation Id="1", LineColor=green: predicted gland regions
  - Annotation Id="2", LineColor=red:   predicted non-gland regions

Each Annotation contains multiple Region polygons in slide-level (level 0)
pixel coordinates so the professor can open this in ImageScope alongside
the SVS and edit/correct directly.

Source → input maps:
  virchow2 / uni2 / phikon-v2  → prob_map_<src>.npy   + valid_mask_<src>.npy
  ensemble_mean                → prob_map_ensemble.npy + valid_mask_ensemble.npy
  hardvote                     → hardvote_pixel_map.npy + hardvote_valid_mask.npy
                                 (semantics inverted: this map is non-gland fraction)

Usage:
    python prediction_to_xml.py S14-2289-1-6 --source virchow2
    python prediction_to_xml.py S14-2289-1-6 --all              # 3 models + hardvote
    python prediction_to_xml.py S14-2289-1-6                    # default: ensemble_mean
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import Config


def color_int_from_rgb(r, g, b):
    """Aperio LineColor packs as B<<16 | G<<8 | R."""
    return (b << 16) | (g << 8) | r


def extract_polygons(binary_mask_thumb, min_area_px_slide,
                     epsilon_px_slide, scale):
    """Find external contours on a binary mask (thumb-res), upscale to slide-res,
    simplify with Douglas-Peucker, and filter by minimum slide-level area.

    Returns list of (N, 2) ndarrays in slide-level coords.
    """
    # cv2.findContours expects CV_8U
    mask = (binary_mask_thumb > 0).astype(np.uint8) * 255
    # morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys_slide = []
    for c in contours:
        # contour shape (N, 1, 2) in (x, y) thumb coords
        # convert to slide-level
        c_slide = (c.squeeze(1).astype(np.float32) * scale)
        # area at slide-level
        area = cv2.contourArea(c_slide.astype(np.float32))
        if area < min_area_px_slide:
            continue
        # simplify (epsilon at slide-level)
        eps = epsilon_px_slide
        approx = cv2.approxPolyDP(c_slide.astype(np.float32), eps, closed=True)
        approx = approx.squeeze(1)  # (M, 2)
        if len(approx) < 3:
            continue
        polys_slide.append(approx)
    return polys_slide


def write_xml(out_path, gland_polys, nongland_polys, mpp=0.2521):
    """Write Aperio ImageScope-compatible XML."""
    parts = [f'<Annotations MicronsPerPixel="{mpp:.6f}">']

    def emit_annotation(ann_id, name, color_int, polys):
        parts.append(
            f'\t<Annotation Id="{ann_id}" Name="{name}" ReadOnly="0" '
            f'NameReadOnly="0" LineColorReadOnly="0" Incremental="0" Type="4" '
            f'LineColor="{color_int}" Visible="1" Selected="0" '
            f'MarkupImagePath="" MacroName="">'
        )
        parts.append('\t\t<Attributes/>')
        parts.append('\t\t<Regions>')
        parts.append('\t\t\t<RegionAttributeHeaders/>')
        for i, poly in enumerate(polys, start=1):
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            length = float(np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1).sum())
            parts.append(
                f'\t\t\t<Region Id="{i}" Type="0" Zoom="0.5" Selected="0" '
                f'ImageLocation="" ImageFocus="-1" Length="{length:.1f}" Area="{area:.1f}" '
                f'LengthMicrons="{length*mpp:.1f}" AreaMicrons="{area*mpp*mpp:.1f}" '
                f'Text="" NegativeROA="0" InputRegionId="0" Analyze="1" DisplayId="{i}">'
            )
            parts.append('\t\t\t\t<Attributes/>')
            parts.append('\t\t\t\t<Vertices>')
            for vx, vy in poly:
                parts.append(f'\t\t\t\t\t<Vertex X="{int(round(vx))}" Y="{int(round(vy))}" Z="0"/>')
            parts.append('\t\t\t\t</Vertices>')
            parts.append('\t\t\t</Region>')
        parts.append('\t\t</Regions>')
        parts.append('\t\t<Plots/>')
        parts.append('\t</Annotation>')

    # Aperio LineColor format: B<<16 | G<<8 | R (encoded as BGR int)
    # Green for predicted gland: (0, 255, 0)  → 65280
    # Red for predicted non-gland: (255, 0, 0) → 255
    GREEN = color_int_from_rgb(0, 255, 0)   # = 65280
    RED   = color_int_from_rgb(255, 0, 0)   # = 255

    emit_annotation(1, "predicted_gland",     GREEN, gland_polys)
    emit_annotation(2, "predicted_non_gland", RED,   nongland_polys)

    parts.append('</Annotations>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


SOURCE_CHOICES = ("virchow2", "uni2", "phikon-v2", "hardvote", "ensemble_mean")
ALL_SOURCES = ("virchow2", "uni2", "phikon-v2", "hardvote")


def load_source_maps(out_dir, source):
    """Return (gland_mask_thumb, nongland_mask_thumb) for the requested source.

    For prob-style sources the map is P(gland); we threshold ≥ 0.5 → gland.
    For 'hardvote' the saved map is the non-gland VOTE fraction (0/1 averaged
    across overlapping patches), so the semantics flip: ≥ 0.5 → non-gland.
    """
    if source == "hardvote":
        m = np.load(out_dir / "hardvote_pixel_map.npy")
        v = np.load(out_dir / "hardvote_valid_mask.npy")
        nongland = (m >= 0.5) & v
        gland = (m < 0.5) & v
    else:
        if source == "ensemble_mean":
            m = np.load(out_dir / "prob_map_ensemble.npy")
            v = np.load(out_dir / "valid_mask_ensemble.npy")
        else:
            m = np.load(out_dir / f"prob_map_{source}.npy")
            v = np.load(out_dir / f"valid_mask_{source}.npy")
        gland = (m >= 0.5) & v
        nongland = (m < 0.5) & v
    return gland, nongland, v


def emit_one_xml(out_dir, slide, source, scale, mpp, min_area_px, epsilon_px):
    gland_mask, nongland_mask, valid = load_source_maps(out_dir, source)

    print(f"\n[{source}] mask coverage at thumb-res:")
    print(f"  predicted gland     : {gland_mask.sum()} px ({gland_mask.mean()*100:.1f}%)")
    print(f"  predicted non-gland : {nongland_mask.sum()} px ({nongland_mask.mean()*100:.1f}%)")
    print(f"  no valid prediction : {(~valid).sum()} px (outside tissue)")

    gland_polys = extract_polygons(gland_mask, min_area_px, epsilon_px, scale)
    nongland_polys = extract_polygons(nongland_mask, min_area_px, epsilon_px, scale)
    print(f"  → polygons:  gland={len(gland_polys)}  non-gland={len(nongland_polys)}")

    xml_path = out_dir / f"{slide}_prediction_{source}.xml"
    write_xml(xml_path, gland_polys, nongland_polys, mpp=mpp)
    print(f"  → wrote {xml_path.name}")
    return xml_path, len(gland_polys), len(nongland_polys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--source", type=str, default="ensemble_mean",
                        choices=SOURCE_CHOICES,
                        help="which prediction source to convert")
    parser.add_argument("--all", action="store_true",
                        help=f"emit one XML per source in {ALL_SOURCES}")
    parser.add_argument("--min-area-px", type=float, default=5000.0,
                        help="minimum polygon area in slide-level pixels (level 0)")
    parser.add_argument("--epsilon-px", type=float, default=8.0,
                        help="Douglas-Peucker simplification epsilon in slide-level pixels")
    args = parser.parse_args()

    config = Config()
    out_dir = Path(config.base_dir) / "results" / args.slide
    if not out_dir.exists():
        raise FileNotFoundError(f"No inference output at {out_dir}")

    meta = np.load(out_dir / "slide_meta.npy", allow_pickle=True).item()
    scale = meta["scale"]
    mpp = float(meta.get("mpp", 0.2521) or 0.2521)

    print(f"Slide: {args.slide}")
    print(f"  Slide-level dims: {meta['slide_w']}×{meta['slide_h']}")
    print(f"  Thumbnail dims:   {meta['thumb_W']}×{meta['thumb_H']}, scale={scale:.3f}")
    print(f"  Min polygon area (slide-level px): {args.min_area_px}")
    print(f"  Simplification epsilon (slide-level px): {args.epsilon_px}")

    sources = ALL_SOURCES if args.all else (args.source,)
    written = []
    for src in sources:
        written.append(emit_one_xml(out_dir, args.slide, src, scale, mpp,
                                     args.min_area_px, args.epsilon_px))

    print("\nDone. Open XML in ImageScope alongside SVS to inspect/edit.")
    for path, ng, nn in written:
        print(f"  {path.name:50s}  gland={ng:>3}  non-gland={nn:>3}")


if __name__ == "__main__":
    main()
