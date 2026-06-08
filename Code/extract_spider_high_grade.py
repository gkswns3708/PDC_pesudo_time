"""
After all 17 SPIDER-colorectal tar parts are downloaded, do:
  1. concatenate tar.* and extract metadata.json (small) and images/ index
  2. read metadata.json → filter records with class == "Adenocarcinoma high grade"
  3. extract just those patches (central + 24 context) into a flat folder
  4. build composite 1120×1120 stitched images using SPIDERDataset logic
  5. render a contact sheet comparing SPIDER high-grade vs OUR putative-PDC patches

Outputs (in /app/spider_samples/high_grade_extract/):
  metadata.json
  metadata_high_grade.json      (filtered)
  raw_patches/<patch_filename>  (only patches referenced by high-grade records)
  stitched_1120/<slide>__<center_name>.png
  contact_sheet.png

Usage:
  /root/miniconda3/envs/tiatoolbox/bin/python extract_spider_high_grade.py [--limit N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from PIL import Image, ImageFile, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
ImageFile.LOAD_TRUNCATED_IMAGES = True

TAR_DIR = Path("/app/spider_samples/tar")
OUT = Path("/app/spider_samples/high_grade_extract")
TMP_FIFO_DIR = Path("/tmp/spider_extract")

TARGET_CLASS = "Adenocarcinoma high grade"


def stage1_extract_metadata(limit_files=None):
    """Cat all tar.* → tar | extract metadata.json (small file)."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta_target = OUT / "metadata.json"
    if meta_target.exists():
        print(f"[skip] metadata.json already at {meta_target}")
        return meta_target
    tar_parts = sorted(TAR_DIR.glob("spider-colorectal.tar.*"))
    print(f"[stage1] cat {len(tar_parts)} tar parts → extract metadata.json only")
    cmd = f"cat {' '.join(str(p) for p in tar_parts)} | tar -xf - -C {OUT} --wildcards '*/metadata.json'"
    print(f"  $ {cmd[:120]}...")
    subprocess.check_call(cmd, shell=True)
    found = list(OUT.rglob("metadata.json"))
    if not found:
        sys.exit("metadata.json not found after extraction")
    if found[0] != meta_target:
        shutil.move(found[0], meta_target)
    # Clean up SPIDER-colorectal/ dir if exists
    spider_dir = OUT / "SPIDER-colorectal"
    if spider_dir.exists() and spider_dir.is_dir():
        try: spider_dir.rmdir()
        except OSError: pass
    print(f"[stage1] {meta_target}  ({meta_target.stat().st_size/1e6:.1f} MB)")
    return meta_target


def stage2_filter_metadata(meta_path):
    """Filter records by class == high grade. Return filtered list."""
    with open(meta_path) as f:
        records = json.load(f)
    print(f"[stage2] total records: {len(records):,}")
    # class distribution
    from collections import Counter
    cc = Counter(r["class"] for r in records)
    print("  class distribution:")
    for c, n in cc.most_common():
        print(f"    {n:>7,}  {c}")
    high = [r for r in records if r["class"] == TARGET_CLASS]
    print(f"[stage2] '{TARGET_CLASS}' records: {len(high):,}")
    with open(OUT / "metadata_high_grade.json", "w") as f:
        json.dump(high, f)
    return high


def stage3_extract_patches(records):
    """Build set of patch filenames needed (center + 24 context) and extract."""
    needed = set()
    for r in records:
        needed.add(r["image_name"])
        for v in r.get("context_info", {}).values():
            needed.add(v)
    print(f"[stage3] need {len(needed):,} patch images (~{len(needed)*0.05:.1f} MB)")
    raw_dir = OUT / "raw_patches"
    raw_dir.mkdir(exist_ok=True)
    # use tar with --files-from to extract only needed
    list_file = OUT / "_needed_files.txt"
    with open(list_file, "w") as f:
        for name in sorted(needed):
            f.write(f"SPIDER-colorectal/images/{name}\n")
    tar_parts = sorted(TAR_DIR.glob("spider-colorectal.tar.*"))
    cmd = (f"cat {' '.join(str(p) for p in tar_parts)} | "
           f"tar -xf - -C {OUT} --files-from {list_file}")
    t0 = time.time()
    print(f"  $ extracting ...")
    subprocess.check_call(cmd, shell=True)
    # move from SPIDER-colorectal/images/ to raw_patches/
    src_dir = OUT / "SPIDER-colorectal" / "images"
    if src_dir.exists():
        print(f"  moving {len(list(src_dir.iterdir()))} files to {raw_dir}")
        for p in src_dir.iterdir():
            dst = raw_dir / p.name
            if not dst.exists():
                shutil.move(str(p), str(dst))
        try: src_dir.rmdir(); (OUT/"SPIDER-colorectal").rmdir()
        except OSError: pass
    print(f"[stage3] extracted in {(time.time()-t0)/60:.1f} min  raw_patches: {len(list(raw_dir.iterdir()))} files")
    return raw_dir


def stage4_stitch(records, raw_dir, limit=None):
    """Build 1120×1120 composites for each high-grade record."""
    stitched_dir = OUT / "stitched_1120"
    stitched_dir.mkdir(exist_ok=True)
    PS = 224
    GRID = 5
    CENTER = 2
    if limit:
        records = records[:limit]
    print(f"[stage4] stitching {len(records):,} records into 1120×1120 composites")
    for i, rec in enumerate(records):
        canvas = Image.new("RGB", (PS*GRID, PS*GRID), (255, 255, 255))
        for ii in range(GRID):
            for jj in range(GRID):
                if ii == CENTER and jj == CENTER:
                    name = rec["image_name"]
                else:
                    name = rec.get("context_info", {}).get(f"{ii}_{jj}")
                if name is None:
                    continue
                p = raw_dir / name
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                canvas.paste(img, (jj*PS, ii*PS))
        slide = rec.get("slide_id", "unknown")
        stem = Path(rec["image_name"]).stem
        out = stitched_dir / f"{slide}__{stem}.png"
        canvas.save(out, optimize=True)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}]")
    print(f"[stage4] saved {len(records)} composites to {stitched_dir}")
    return stitched_dir


def stage5_contact_sheet(stitched_dir, n=12):
    """Build a contact sheet of SPIDER high-grade composites alongside our slide.
       n composites: pick n random ones."""
    import random, numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spider_files = sorted(stitched_dir.glob("*.png"))
    if not spider_files:
        print("[stage5] no stitched files"); return
    random.seed(0)
    pick_spider = random.sample(spider_files, min(n, len(spider_files)))

    # OUR putative-PDC patches: take the 9 GT-non-gland patches from spider-scale analysis
    our_dir = Path("/app/spider_input_check")
    our_files = []
    # generate 4 OURs from results CSV (top GT non-gland)
    import pandas as pd, openslide
    sys.path.insert(0, '/app/Gland_Seg/Code')
    from config import Config
    from infer_spider_on_eval import patch_to_spider_input
    cfg = Config()
    pred = pd.read_csv('/app/Gland_Seg/results/S14-2289-1-6/per_patch_predictions_spider_scale.csv')
    ng = pred[pred.gt_label_thr025 == 1].sort_values('pct_nested', ascending=False).head(4)
    svs = openslide.OpenSlide(str(Path(cfg.svs_dir)/'S14-2289-1-6.svs'))
    ours_save = OUT / "our_pdc_composites"
    ours_save.mkdir(exist_ok=True)
    for _, r in ng.iterrows():
        cx = int(r.x) + 1120; cy = int(r.y) + 1120
        pil = patch_to_spider_input(svs, cx, cy, level0_size=2240, out_size=1120)
        f = ours_save / f"OUR__x{int(r.x)}_y{int(r.y)}__pct{r.pct_nested:.2f}.png"
        pil.save(f, optimize=True)
        our_files.append(f)
    svs.close()

    rows = 4; cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    # row 0-2 (12): SPIDER
    for i, f in enumerate(pick_spider):
        r, c = i // cols, i % cols
        axes[r,c].imshow(np.asarray(Image.open(f)))
        axes[r,c].set_title(f'SPIDER high-grade\n{f.stem[:25]}', fontsize=8)
    # row 3 (4): OURs
    for i, f in enumerate(our_files):
        axes[3,i].imshow(np.asarray(Image.open(f)))
        axes[3,i].set_title(f'OUR putative-PDC\n{f.stem[:25]}', fontsize=8)
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('SPIDER "Adenocarcinoma high grade" composites  vs  OUR GT non-gland composites\n'
                 '(both at SPIDER input scale 1120×1120 @ 20×, 564 µm FOV)', fontsize=12)
    fig.tight_layout()
    out = OUT / "contact_sheet.png"
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"[stage5] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stitch only first N high-grade records (default all)")
    ap.add_argument("--stage", type=int, default=0,
                    help="run from this stage (1..5)  0=all")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage <= 1:
        meta = stage1_extract_metadata()
    else:
        meta = OUT / "metadata.json"

    if args.stage <= 2:
        records = stage2_filter_metadata(meta)
    else:
        records = json.load(open(OUT/"metadata_high_grade.json"))

    if args.stage <= 3:
        raw_dir = stage3_extract_patches(records)
    else:
        raw_dir = OUT / "raw_patches"

    if args.stage <= 4:
        stage4_stitch(records, raw_dir, limit=args.limit)

    if args.stage <= 5:
        stage5_contact_sheet(OUT / "stitched_1120", n=12)


if __name__ == "__main__":
    main()
