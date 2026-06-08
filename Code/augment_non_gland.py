"""Pre-generate augmented non-gland patches on disk to rebalance the dataset.

For each non-gland PNG under `<patches_dir>/<slide>/non-gland/`, write `--copies` (default 3)
augmented copies named `<orig>_aug{1..N}.png` next to it. Geometric augmentations only —
labels are preserved. Stain normalization is NOT re-applied; we operate on the already-
Macenko-normalized patches under `patches_stainnorm_256/`.

Augmentation per copy:
    Rotate(limit=180, reflect)            p=1.0
    Affine(scale, translate, shear)       p=0.8
    OneOf(Elastic, GridDistortion)        p=0.5

Reproducibility: per-copy seed = stable hash(orig_path) ^ copy_idx.

Usage:
    python augment_non_gland.py \\
        --patches-dir /app/Gland_Seg/patches_stainnorm_256 \\
        --copies 3 --workers 32 \\
        --viz-dir /app/Gland_Seg/Viz/augmentation_check
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

NON_GLAND_DIR = "non-gland"
AUG_SUFFIX_PREFIX = "_aug"  # files named "<stem>_aug<N>.png"


def build_transform():
    return A.Compose([
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            shear=(-10, 10),
            rotate=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=20, sigma=5, border_mode=cv2.BORDER_REFLECT_101),
            A.GridDistortion(num_steps=5, distort_limit=0.2, border_mode=cv2.BORDER_REFLECT_101),
        ], p=0.5),
    ])


def stable_seed(path_str, copy_idx):
    h = hashlib.md5(f"{path_str}|{copy_idx}".encode()).hexdigest()
    return int(h[:8], 16)


_TRANSFORM = None


def _worker_init():
    global _TRANSFORM
    _TRANSFORM = build_transform()


def _process_one(args):
    src_path_str, copies, overwrite = args
    src = Path(src_path_str)
    img = cv2.imread(str(src))
    if img is None:
        return src_path_str, 0, "read_failed"

    written = 0
    for i in range(1, copies + 1):
        dst = src.with_name(f"{src.stem}{AUG_SUFFIX_PREFIX}{i}.png")
        if dst.exists() and not overwrite:
            continue
        seed = stable_seed(src_path_str, i)
        np.random.seed(seed)
        out = _TRANSFORM(image=img)["image"]
        cv2.imwrite(str(dst), out)
        written += 1
    return src_path_str, written, "ok"


def collect_originals(patches_dir):
    """Return list of original (non-augmented) non-gland PNG paths, grouped by slide."""
    root = Path(patches_dir)
    by_slide = {}
    for slide_dir in sorted(root.iterdir()):
        if not slide_dir.is_dir():
            continue
        nong = slide_dir / NON_GLAND_DIR
        if not nong.is_dir():
            continue
        originals = []
        for p in sorted(nong.glob("*.png")):
            if AUG_SUFFIX_PREFIX in p.stem:
                continue  # skip prior augs
            originals.append(p)
        if originals:
            by_slide[slide_dir.name] = originals
    return by_slide


def save_sample_grid(slide_name, orig_paths, copies, out_dir, n_samples=4):
    """For one slide, pick n_samples originals, regenerate their aug copies in-memory,
    and save a (n_samples × (1+copies)) grid PNG."""
    if not orig_paths:
        return None
    rng = np.random.default_rng(42)
    picks = list(rng.choice(orig_paths, size=min(n_samples, len(orig_paths)), replace=False))
    t = build_transform()
    cells = []
    for p in picks:
        img = cv2.imread(str(p))
        if img is None:
            continue
        row = [img]
        for i in range(1, copies + 1):
            np.random.seed(stable_seed(str(p), i))
            row.append(t(image=img)["image"])
        cells.append(row)
    if not cells:
        return None
    h, w = cells[0][0].shape[:2]
    pad = 4
    grid_h = len(cells) * h + (len(cells) + 1) * pad
    grid_w = (1 + copies) * w + (1 + copies + 1) * pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for r, row in enumerate(cells):
        y0 = pad + r * (h + pad)
        for c, im in enumerate(row):
            x0 = pad + c * (w + pad)
            grid[y0:y0 + h, x0:x0 + w] = im
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slide_name}_sample.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-dir", required=True)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--viz-dir", default=None)
    ap.add_argument("--viz-samples", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[scan] {args.patches_dir}", flush=True)
    by_slide = collect_originals(args.patches_dir)
    total_orig = sum(len(v) for v in by_slide.values())
    print(f"[scan] {len(by_slide)} slides with non-gland, {total_orig} originals", flush=True)
    for s, paths in by_slide.items():
        print(f"    {s}: {len(paths)} originals", flush=True)
    expected_new = total_orig * args.copies
    print(f"[plan] copies/orig = {args.copies}  → {expected_new} new PNGs", flush=True)

    if args.viz_dir:
        print(f"[viz] sample grids → {args.viz_dir}", flush=True)
        for s, paths in by_slide.items():
            out = save_sample_grid(s, paths, args.copies, args.viz_dir, n_samples=args.viz_samples)
            if out:
                print(f"    {out}", flush=True)

    if args.dry_run:
        print("[dry-run] exit before write", flush=True)
        return

    tasks = []
    for paths in by_slide.values():
        for p in paths:
            tasks.append((str(p), args.copies, args.overwrite))
    print(f"[run] {len(tasks)} tasks, {args.workers} workers", flush=True)

    t0 = time.time()
    n_done = 0
    n_written = 0
    n_failed = 0
    log_every = max(500, len(tasks) // 50)

    with mp.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for src_path_str, w, status in pool.imap_unordered(_process_one, tasks, chunksize=16):
            n_done += 1
            n_written += w
            if status != "ok":
                n_failed += 1
                if n_failed <= 10:
                    print(f"    [warn] {status}: {src_path_str}", flush=True)
            if n_done % log_every == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-6)
                eta = (len(tasks) - n_done) / max(rate, 1e-6)
                print(
                    f"    [{n_done}/{len(tasks)}] written={n_written}  "
                    f"failed={n_failed}  elapsed={dt:.0f}s  eta={eta:.0f}s  "
                    f"rate={rate:.1f}/s",
                    flush=True,
                )

    dt = time.time() - t0
    print(f"[done] {n_done} originals processed in {dt:.1f}s "
          f"({n_done/max(dt,1e-6):.1f}/s); wrote {n_written} PNGs; failed {n_failed}", flush=True)


if __name__ == "__main__":
    main()
