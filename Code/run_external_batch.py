"""
Batch inference over N external (unannotated) slides drawn from /Public,
then aggregate predicted patches into a single flat delivery folder.

Pipeline per slide:
    1. infer_external_slide.py --svs-path ...  (writes per-slide outputs)
    2. summarize_external_predictions.py        (writes prediction CSV + hardvote)
    3. prediction_to_xml.py --source virchow2  + --source hardvote
After all slides:
    4. Aggregate patches: prediction/{virchow2,hardvoting}/{gland,non-gland}/
       Filename: <slide>_x{X}_y{Y}.png (level-0 absolute coords).
       Each (source, class) gets 100 high-confidence + 100 boundary per slide
       (=> 10 slides × 200 = 2,000 patches per folder).
    5. Copy XMLs into xmls/  (2 per slide × N slides)
    6. Zip annotation (deflate) and prediction (stored — PNG already compressed).

Usage:
    python run_external_batch.py
        --pool-dir /Public/05-WSI/Rawdata/CRC-14-15-gist/S14_CRC0001-0176
        --n-slides 10
        --seed 42
        --out-dir /app/Gland_Seg/results/external10
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

from config import Config
from extract_sample_patches import select_patches


PYTHON = "/root/miniconda3/envs/tiatoolbox/bin/python"

# Slides used in training (8) + held-out evaluation (1).
EXCLUDE_STEMS = {
    "S14-1255-1-3", "S14-1382-4", "S14-1639-1-7", "S14-1720-6",
    "S14-177-1-5", "S14-2162-1-5", "S14-248-1-3", "S14-252-3",
    "S14-2289-1-6",
}

SOURCES = ("virchow2", "hardvote")
CLASSES = ("gland", "non-gland")
PRED_COL = {"virchow2": "pred_virchow2", "hardvote": "pred_hardvote"}


def pick_slides(pool_dir, n, seed, explicit=None):
    pool = sorted(Path(pool_dir).glob("*.svs"))
    if explicit:
        wanted = set(explicit)
        chosen = [p for p in pool if p.stem in wanted]
        missing = wanted - {p.stem for p in chosen}
        if missing:
            raise RuntimeError(f"explicit slides not found in pool: {sorted(missing)}")
        return sorted(chosen, key=lambda p: p.stem)
    candidates = [p for p in pool if p.stem not in EXCLUDE_STEMS]
    if len(candidates) < n:
        raise RuntimeError(f"only {len(candidates)} candidates available, need {n}")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n), key=lambda p: p.stem)


def run(cmd, log_prefix=""):
    print(f"\n>>> {log_prefix}{' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    if p.returncode != 0:
        sys.exit(f"FAILED ({p.returncode}) after {dt:.1f}s: {' '.join(map(str,cmd))}")
    print(f"    done in {dt:.1f}s", flush=True)


def aggregate_patches_for_slide(slide_stem, svs_path, results_dir, target_root,
                                 patch_size, n_each=100):
    """For each (source, class, tag) save patches into a SEPARATE subfolder.

    Layout:
      target_root/<source>/<class>_high_conf/<slide>_x_y.png
      target_root/<source>/<class>_boundary/<slide>_x_y.png
    """
    df = pd.read_csv(results_dir / "per_patch_predictions_with_hardvote.csv")
    so = openslide.OpenSlide(str(svs_path))
    try:
        for source in SOURCES:
            for cls in CLASSES:
                df_cls = df[df[PRED_COL[source]] == cls]
                if len(df_cls) == 0:
                    print(f"  [{slide_stem}] {source}/{cls}: 0 patches, skip")
                    continue
                high, boundary = select_patches(df_cls, source, cls, n_each)
                for tag, sub in (("high_conf", high), ("boundary", boundary)):
                    folder = target_root / source / f"{cls}_{tag}"
                    folder.mkdir(parents=True, exist_ok=True)
                    for _, row in sub.iterrows():
                        x, y = int(row["x"]), int(row["y"])
                        fname = f"{slide_stem}_x{x}_y{y}.png"
                        img = so.read_region((x, y), 0, (patch_size, patch_size))
                        img.convert("RGB").save(folder / fname, optimize=True)
                    print(f"  [{slide_stem}] {source}/{cls}_{tag}: saved {len(sub)} patches")
    finally:
        so.close()


def make_zip(out_zip, root_dir, store_only=False):
    import zipfile
    method = zipfile.ZIP_STORED if store_only else zipfile.ZIP_DEFLATED
    kwargs = {} if store_only else {"compresslevel": 9}
    with zipfile.ZipFile(out_zip, "w", method, allowZip64=True, **kwargs) as z:
        for dp, _, files in __import__("os").walk(root_dir):
            for f in files:
                p = Path(dp) / f
                z.write(p, p.relative_to(root_dir.parent))
    return out_zip.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default="/Public/05-WSI/Rawdata/CRC-14-15-gist/S14_CRC0001-0176")
    parser.add_argument("--n-slides", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slides", nargs="+", default=None,
                        help="explicit slide stems (overrides --n-slides/--seed)")
    parser.add_argument("--out-dir", default="/app/Gland_Seg/results/external10")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-each", type=int, default=100,
                        help="patches per (source,class,tag) per slide (200 total per source-class)")
    args = parser.parse_args()

    config = Config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = out_dir / "prediction"
    xmls_dir = out_dir / "xmls"
    xmls_dir.mkdir(parents=True, exist_ok=True)

    slides = pick_slides(args.pool_dir, args.n_slides, args.seed, explicit=args.slides)
    sel_path = out_dir / "selected_slides.txt"
    sel_path.write_text("\n".join(p.stem for p in slides) + "\n")
    print(f"Selected {len(slides)} slides → {sel_path}:")
    total_gb = 0.0
    for p in slides:
        gb = p.stat().st_size / 1e9
        total_gb += gb
        print(f"  {p.stem}  ({gb:.2f} GB)")
    print(f"  total = {total_gb:.1f} GB")

    # ── Copy SVS from /Public to local /app for faster IO + training reuse ──
    local_svs_dir = Path(config.svs_dir)
    local_svs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying SVS to {local_svs_dir} (skip if already present) ...")
    local_paths = []
    for p in slides:
        dest = local_svs_dir / p.name
        if dest.exists() and dest.stat().st_size == p.stat().st_size:
            print(f"  [skip] {p.name} already at {dest}")
        else:
            t0 = time.time()
            shutil.copy2(p, dest)
            print(f"  copied {p.name}  ({dest.stat().st_size/1e9:.2f} GB, {time.time()-t0:.1f}s)")
        local_paths.append(dest)

    code_dir = Path("/app/Gland_Seg/Code")
    t_total = time.time()
    for idx, svs_path in enumerate(local_paths, 1):
        slide_stem = svs_path.stem
        results_dir = Path(config.base_dir) / "results" / slide_stem
        print(f"\n========================================================")
        print(f"[{idx}/{len(local_paths)}] SLIDE: {slide_stem}")
        print(f"========================================================")

        # 1) Inference
        run([PYTHON, str(code_dir / "infer_external_slide.py"),
             slide_stem,
             "--svs-path", str(svs_path),
             "--models", "virchow2", "uni2", "phikon-v2",
             "--stride", str(args.stride),
             "--batch-size", str(args.batch_size),
             "--workers", str(args.workers)],
            log_prefix=f"[{slide_stem}] infer ")

        # 2) Summarize (writes per_patch_predictions_with_hardvote.csv etc.)
        run([PYTHON, str(code_dir / "summarize_external_predictions.py"), slide_stem],
            log_prefix=f"[{slide_stem}] summarize ")

        # 3) XMLs (virchow2 + hardvote only)
        for src in SOURCES:
            run([PYTHON, str(code_dir / "prediction_to_xml.py"),
                 slide_stem, "--source", src],
                log_prefix=f"[{slide_stem}] xml-{src} ")
            xml_path = results_dir / f"{slide_stem}_prediction_{src}.xml"
            if xml_path.exists():
                shutil.copy2(xml_path, xmls_dir / xml_path.name)

        # 4) Aggregate patches into flat structure (read from local SVS copy)
        meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
        aggregate_patches_for_slide(
            slide_stem, str(svs_path), results_dir, pred_root,
            patch_size=meta["patch_size"], n_each=args.n_each,
        )

        elapsed = time.time() - t_total
        print(f"\n  cumulative elapsed: {elapsed/60:.1f} min")

    # ── Final zips ──
    print("\n========================================================")
    print("Building delivery zips")
    print("========================================================")
    ann_zip = out_dir / "external10_annotation.zip"
    pred_zip = out_dir / "external10_prediction.zip"
    sz_ann = make_zip(ann_zip, xmls_dir, store_only=False)
    sz_pred = make_zip(pred_zip, pred_root, store_only=True)
    print(f"  {ann_zip.name}: {sz_ann/1e6:.1f} MB")
    print(f"  {pred_zip.name}: {sz_pred/1e9:.2f} GB")

    # ── Counts sanity ──
    print("\nFinal patch counts:")
    for s in SOURCES:
        for c in CLASSES:
            n = len(list((pred_root / s / c).glob("*.png")))
            print(f"  {s}/{c}: {n}")

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
