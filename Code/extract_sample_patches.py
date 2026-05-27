"""
Extract per-class sample patches (raw 512×512 PNG, level 0) for the professor.

For each prediction source (virchow2 / uni2 / phikon-v2 / hardvote) and each
predicted class (gland / non-gland), extract:
  - 100 "high-confidence" patches  — farthest from the 0.5 decision boundary
  - 100 "boundary" patches         — closest to 0.5 (least-certain still in class)

Confidence proxy:
  per-model : |p_gland_<model> - 0.5|
  hardvote  : |p_gland_ensemble - 0.5|   (ensemble mean prob)

Output structure:
    /app/Gland_Seg/results/<slide>/sample_patches/
    ├── virchow2/
    │   ├── gland_high_conf/        100 PNGs + manifest.csv
    │   ├── gland_boundary/         100 PNGs + manifest.csv
    │   ├── non-gland_high_conf/    100 PNGs + manifest.csv
    │   └── non-gland_boundary/     100 PNGs + manifest.csv
    ├── uni2/        (same 4 subfolders)
    ├── phikon-v2/   (same 4 subfolders)
    └── hardvote/    (same 4 subfolders)

Usage:
    python extract_sample_patches.py S14-2289-1-6 [--n 100]
"""

import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

from config import Config


SOURCES = ("virchow2", "uni2", "phikon-v2", "hardvote")
CLASSES = ("gland", "non-gland")


def conf_column(source):
    """Column to use as confidence proxy."""
    return "p_gland_ensemble" if source == "hardvote" else f"p_gland_{source}"


def select_patches(df_class, source, predicted_class, n_each):
    """Return (high_conf_df, boundary_df) for one (source, class)."""
    col = conf_column(source)
    # distance from 0.5 (higher = more confident in the predicted class)
    if predicted_class == "gland":
        score = df_class[col] - 0.5      # > 0 since predicted gland
    else:
        score = 0.5 - df_class[col]      # > 0 since predicted non-gland

    df_sorted = df_class.assign(_conf=score).sort_values("_conf", ascending=False)
    n = len(df_sorted)
    if n < 2 * n_each:
        # Not enough patches to give 100+100 disjoint; split in half
        cut = n // 2
        high = df_sorted.iloc[:cut]
        boundary = df_sorted.iloc[cut:]
        print(f"    [warn] only {n} patches in this class — split {len(high)}/{len(boundary)}")
    else:
        high = df_sorted.iloc[:n_each]
        boundary = df_sorted.iloc[-n_each:].iloc[::-1]  # least-confident first
    return high, boundary


def save_patches(slide_obj, sub_df, folder, patch_size, source):
    """Save patches in sub_df as PNGs in `folder` and write manifest.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    conf_col = conf_column(source)
    for rank, (_, row) in enumerate(sub_df.iterrows(), start=1):
        x, y = int(row["x"]), int(row["y"])
        img = slide_obj.read_region((x, y), 0, (patch_size, patch_size))
        fname = f"rank{rank:03d}_x{x}_y{y}.png"
        img.convert("RGB").save(folder / fname, optimize=True)
        rows.append({
            "rank": rank,
            "filename": fname,
            "x": x,
            "y": y,
            "p_gland_virchow2":  float(row["p_gland_virchow2"]),
            "p_gland_uni2":      float(row["p_gland_uni2"]),
            "p_gland_phikon-v2": float(row["p_gland_phikon-v2"]),
            "p_gland_ensemble":  float(row["p_gland_ensemble"]),
            "conf_proxy":        float(row[conf_col]),
            "pred_virchow2":  row["pred_virchow2"],
            "pred_uni2":      row["pred_uni2"],
            "pred_phikon-v2": row["pred_phikon-v2"],
            "pred_hardvote":  row["pred_hardvote"],
        })
    pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide")
    parser.add_argument("--n", type=int, default=100,
                        help="patches per (source, class, tag)")
    args = parser.parse_args()

    config = Config()
    results = Path(config.base_dir) / "results" / args.slide
    csv_path = results / "per_patch_predictions_with_hardvote.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    meta = np.load(results / "slide_meta.npy", allow_pickle=True).item()
    patch_size = meta["patch_size"]

    # Find SVS path via config
    info = config.external_test_slides.get(args.slide) or config.slides.get(args.slide)
    if info is None:
        raise KeyError(f"{args.slide} not in config.slides nor external_test_slides")
    svs_path = Path(config.svs_dir) / info["svs"]
    print(f"Slide: {svs_path}")
    slide_obj = openslide.OpenSlide(str(svs_path))

    out_root = results / "sample_patches"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}")

    pred_col_map = {"virchow2": "pred_virchow2", "uni2": "pred_uni2",
                    "phikon-v2": "pred_phikon-v2", "hardvote": "pred_hardvote"}

    summary = []
    for source in SOURCES:
        pred_col = pred_col_map[source]
        for cls in CLASSES:
            df_cls = df[df[pred_col] == cls]
            if len(df_cls) == 0:
                print(f"  [skip] {source}/{cls}: 0 patches")
                continue
            high, boundary = select_patches(df_cls, source, cls, args.n)
            print(f"  {source}/{cls}: total={len(df_cls)}  "
                  f"high_conf={len(high)}  boundary={len(boundary)}")
            for tag, sub in (("high_conf", high), ("boundary", boundary)):
                folder = out_root / source / f"{cls}_{tag}"
                save_patches(slide_obj, sub, folder, patch_size, source)
                summary.append({"source": source, "class": cls, "tag": tag,
                                "n": len(sub), "folder": str(folder.relative_to(results))})

    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    print(f"\nDone. Total folders: {len(summary)}")
    print(f"Manifest: {out_root / 'summary.csv'}")
    slide_obj.close()


if __name__ == "__main__":
    main()
