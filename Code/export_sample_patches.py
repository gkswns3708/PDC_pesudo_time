"""
Export sample patches for the professor's manual re-classification.

Selects from the per-patch CSV produced by infer_external_slide.py and
extracts each patch image from the SVS at slide-level (level 0), with the
same Macenko stain normalization used during training.

Categories (50 each → 200 total):
  - predicted_gland_high_conf   (p_ensemble > 0.95)
  - predicted_gland_uncertain   (0.50 < p_ensemble < 0.70)
  - predicted_non_gland_high_conf  (p_ensemble < 0.05)
  - predicted_non_gland_uncertain  (0.30 < p_ensemble < 0.50)

Output:
    results/<slide>/sample_patches/
        predicted_gland/{high_conf,uncertain}/*.png
        predicted_non_gland/{high_conf,uncertain}/*.png
        metadata.csv

Usage:
    python export_sample_patches.py S14-2289-1-6 [--n-per-bucket 50]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd

from config import Config
from stain_normalizer import MacenkoNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--n-per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config()
    out_dir = Path(config.base_dir) / "results" / args.slide
    if not out_dir.exists():
        raise FileNotFoundError(f"No inference output at {out_dir}")

    csv_path = out_dir / "per_patch_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} patches from {csv_path}")
    print(f"P(gland) distribution: min={df.p_gland_ensemble.min():.3f}  "
          f"max={df.p_gland_ensemble.max():.3f}  mean={df.p_gland_ensemble.mean():.3f}")

    rng = np.random.default_rng(args.seed)

    # Define buckets
    p = df["p_gland_ensemble"]
    buckets = {
        ("predicted_gland", "high_conf"):     df[p >= 0.95],
        ("predicted_gland", "uncertain"):     df[(p > 0.50) & (p < 0.70)],
        ("predicted_non_gland", "high_conf"): df[p <= 0.05],
        ("predicted_non_gland", "uncertain"): df[(p > 0.30) & (p < 0.50)],
    }
    print("\nBucket sizes (before sampling):")
    for k, sub in buckets.items():
        print(f"  {k[0]:>20} / {k[1]:<10} : {len(sub)} candidates")

    # Sample
    sampled = {}
    for k, sub in buckets.items():
        n = min(args.n_per_bucket, len(sub))
        if n == 0:
            print(f"  [warn] empty bucket: {k}")
            sampled[k] = sub.iloc[0:0]
            continue
        idx = rng.choice(len(sub), size=n, replace=False)
        sampled[k] = sub.iloc[idx].reset_index(drop=True)

    # Open slide + Macenko (same target as training)
    info = config.external_test_slides[args.slide]
    svs_path = str(Path(config.svs_dir) / info["svs"])
    slide = openslide.OpenSlide(svs_path)
    print(f"\nOpened SVS: {svs_path}  dims={slide.level_dimensions[0]}")

    target_rgb = None
    if config.stain_normalize:
        t = cv2.imread(config.stain_target_path)
        target_rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        normalizer = MacenkoNormalizer()
        normalizer.fit(target_rgb)
    else:
        normalizer = None

    # Output dirs
    base = out_dir / "sample_patches"
    base.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for (cls, bucket), sub in sampled.items():
        d = base / cls / bucket
        d.mkdir(parents=True, exist_ok=True)
        print(f"\nExtracting {len(sub)} patches → {d}")
        for i, row in sub.iterrows():
            x, y = int(row["x"]), int(row["y"])
            patch = slide.read_region((x, y), 0, (config.patch_size, config.patch_size))
            patch_rgb = np.array(patch.convert("RGB"))
            if normalizer is not None:
                try:
                    patch_rgb = normalizer.transform(patch_rgb)
                except Exception:
                    pass  # skip normalization for failing patches but still save raw
            fname = f"{cls}_{bucket}_{i:03d}_x{x}_y{y}.png"
            cv2.imwrite(str(d / fname),
                        cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
            row_meta = row.to_dict()
            row_meta.update({
                "filename": fname,
                "class_predicted": cls,
                "confidence_bucket": bucket,
                "rel_path": f"{cls}/{bucket}/{fname}",
            })
            metadata_rows.append(row_meta)

    slide.close()

    # Combined metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = base / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nMetadata: {meta_path} ({len(meta_df)} rows)")

    # Brief README
    readme = base / "README.md"
    readme.write_text(f"""# Sample patches — {args.slide}

각 patch는 512×512 px, level 0 (40x), Macenko stain normalized.

## Categories ({args.n_per_bucket}장씩, 총 {len(meta_df)}장)

| 폴더 | 의미 |
|---|---|
| `predicted_gland/high_conf/`     | 모델이 gland로 매우 confident하게 예측 (p≥0.95) |
| `predicted_gland/uncertain/`     | gland로 예측했지만 애매 (0.50<p<0.70) |
| `predicted_non_gland/high_conf/` | non-gland로 매우 confident (p≤0.05) |
| `predicted_non_gland/uncertain/` | non-gland로 예측했지만 애매 (0.30<p<0.50) |

## 활용

- **High-conf 패치**: 모델 신뢰도 검증 — 이게 명백히 틀려있으면 모델 자체 문제
- **Uncertain 패치**: 모델이 헷갈려하는 case — 교수님 재분류로 추가 학습 신호 확보 가능

## 메타데이터

`metadata.csv`에 각 patch의 (x, y) 슬라이드 좌표와 모든 모델별 prob 포함.
- `p_gland_virchow2`, `p_gland_uni2`, `p_gland_phikon-v2`: 각 모델별 gland 확률
- `p_gland_ensemble`: 3개 모델 mean prob
- `pred_class_ensemble`: ensemble 기준 hard prediction
""", encoding="utf-8")
    print(f"README: {readme}")

    print(f"\n=== DONE — {len(meta_df)} sample patches at {base} ===")


if __name__ == "__main__":
    main()
