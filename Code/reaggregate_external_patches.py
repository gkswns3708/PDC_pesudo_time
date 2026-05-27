"""
Re-aggregate patches for already-inferred external slides into the new
4-subfolder-per-source layout (high_conf vs boundary), without re-running
inference. Uses cached per_patch_predictions_with_hardvote.csv.

Usage:
    python reaggregate_external_patches.py
        --slides S14-10234-2-3 S14-1069-1-6 S14-1253-1-3
        --out-dir /app/Gland_Seg/results/external3
        [--n-each 100]
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from config import Config
from run_external_batch import (
    aggregate_patches_for_slide, SOURCES, CLASSES,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", nargs="+", required=True)
    parser.add_argument("--out-dir", default="/app/Gland_Seg/results/external3")
    parser.add_argument("--n-each", type=int, default=100)
    parser.add_argument("--svs-dir", default=None,
                        help="defaults to config.svs_dir (= /app/Gland_Seg/Data/S14/SVS)")
    args = parser.parse_args()

    config = Config()
    out = Path(args.out_dir)
    pred_root = out / "prediction"
    if pred_root.exists():
        print(f"Clearing existing {pred_root} ...")
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True)

    svs_dir = Path(args.svs_dir or config.svs_dir)
    for slide in args.slides:
        results_dir = Path(config.base_dir) / "results" / slide
        if not results_dir.exists():
            raise FileNotFoundError(f"missing inference output for {slide}: {results_dir}")
        meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
        svs_path = svs_dir / f"{slide}.svs"
        if not svs_path.exists():
            raise FileNotFoundError(f"missing SVS: {svs_path}")
        print(f"\n[{slide}]")
        aggregate_patches_for_slide(
            slide, str(svs_path), results_dir, pred_root,
            patch_size=meta["patch_size"], n_each=args.n_each,
        )

    # ── Re-zip prediction (STORED) and rebuild annotation zip ──
    pred_zip = out / "external3_prediction.zip"
    print(f"\nWriting {pred_zip} (STORED) ...")
    if pred_zip.exists():
        pred_zip.unlink()
    with zipfile.ZipFile(pred_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in pred_root.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out))
    print(f"  size = {pred_zip.stat().st_size / 1e9:.2f} GB")

    # Counts
    print("\nFinal counts:")
    for src in SOURCES:
        for cls in CLASSES:
            for tag in ("high_conf", "boundary"):
                folder = pred_root / src / f"{cls}_{tag}"
                n = len(list(folder.glob("*.png"))) if folder.exists() else 0
                print(f"  {src}/{cls}_{tag}: {n}")


if __name__ == "__main__":
    main()
