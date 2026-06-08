"""
Minimal prerequisite generator for SPIDER eval pipeline on a new slide.

Saves the two files needed by build_spider_scale_grid.py:
  - results/<slide>/slide_meta.npy
  - results/<slide>/annotation.npz

Does NOT run any model inference. Just parses the XML and saves slide dims.

Usage:
    python prepare_spider_eval_prereq.py S14-177-1-5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import openslide

from config import Config
from visualize_prediction_wsi import get_thumbnail, parse_aperio_xml

THUMB_MAX_DIM = 4000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide", help="slide id (e.g. S14-177-1-5)")
    ap.add_argument("--xml-suffix", default="_S.xml",
                    help="XML suffix in config.xml_dir (default _S.xml)")
    args = ap.parse_args()

    cfg = Config()
    out_dir = Path(cfg.base_dir) / "results" / args.slide
    out_dir.mkdir(parents=True, exist_ok=True)

    svs_path = Path(cfg.svs_dir) / f"{args.slide}.svs"
    if not svs_path.exists():
        sys.exit(f"missing svs: {svs_path}")
    xml_path = Path(cfg.xml_dir) / f"{args.slide}{args.xml_suffix}"
    if not xml_path.exists():
        # try plain name
        alt = Path(cfg.xml_dir) / f"{args.slide}.xml"
        if alt.exists():
            xml_path = alt
        else:
            print(f"[warn] no XML found at {xml_path} or {alt}, will write empty annotation")
            xml_path = None

    slide = openslide.OpenSlide(str(svs_path))
    slide_w, slide_h = slide.level_dimensions[0]
    thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
    H, W = thumb_rgb.shape[:2]
    slide.close()
    print(f"[slide] {svs_path.name}  L0=({slide_w},{slide_h})  thumb=({W},{H})  scale={scale:.3f}")

    # parse XML
    if xml_path and xml_path.exists():
        pos, neg = parse_aperio_xml(str(xml_path))
        print(f"[xml] {xml_path.name}  pos={len(pos)}  neg={len(neg)}")
    else:
        pos, neg = [], []

    np.savez(out_dir / "annotation.npz",
             positive=np.array([p for p in pos], dtype=object),
             negative=np.array([n for n in neg], dtype=object),
             allow_pickle=True)
    np.save(out_dir / "slide_meta.npy", np.array({
        "slide_w": slide_w, "slide_h": slide_h,
        "thumb_W": W, "thumb_H": H, "scale": scale,
        "patch_size": cfg.patch_size, "stride": cfg.stride,
    }, dtype=object))
    np.save(out_dir / "thumbnail.npy", thumb_rgb)
    print(f"[save] {out_dir}/annotation.npz")
    print(f"[save] {out_dir}/slide_meta.npy")
    print(f"[save] {out_dir}/thumbnail.npy")


if __name__ == "__main__":
    main()
