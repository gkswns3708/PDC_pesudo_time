"""
Export sample patches for the professor's manual re-classification, with
confidence buckets separated into folders (not just filename suffix).

Source: only patches that passed the Cancer mask (Kather TUM+STR ≥ threshold).
        i.e., excludes Normal/necrosis/Other patches.

Output structure (4 folders × 50 patches = 200 total + manifest):

    sample_patches_cancer_only/
    ├── high_confidence/
    │   ├── gland/          (50 PNG, p_gland_virchow2 > 0.95)
    │   └── non_gland/      (50 PNG, p_gland_virchow2 < 0.05)
    ├── boundary/
    │   ├── gland/          (50 PNG, 0.50 ≤ p ≤ 0.70)
    │   └── non_gland/      (50 PNG, 0.30 ≤ p < 0.50)
    ├── manifest.csv        (all 200 patches with metadata)
    └── README.md

Usage:
    python export_sample_patches_cancer_only.py S14-2289-1-6 [--n-per-bucket 50]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd

from config import Config
from stain_normalizer import MacenkoNormalizer


BUCKETS = {
    # (folder_class, folder_confidence): (lo, hi)
    ("gland", "high_confidence"):     (0.95, 1.01),
    ("gland", "boundary"):            (0.50, 0.70),
    ("non_gland", "high_confidence"): (-0.01, 0.05),
    ("non_gland", "boundary"):        (0.30, 0.50),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--n-per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cancer-threshold", type=float, default=0.5)
    parser.add_argument("--virchow2-col", type=str, default="p_gland_virchow2")
    parser.add_argument("--svs", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.svs:
        svs_path = args.svs
    elif args.slide in getattr(config, "external_test_slides", {}):
        info = config.external_test_slides[args.slide]
        svs_path = str(Path(config.svs_dir) / info["svs"])
    else:
        raise ValueError(f"{args.slide} not configured")

    out_dir = Path(config.base_dir) / "results" / args.slide
    eval_csv = out_dir / "evaluation_cancer_only.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"{eval_csv} not found — run evaluate_cancer_only.py first")
    df = pd.read_csv(eval_csv)
    print(f"Loaded {len(df)} evaluated patches")

    # Filter: cancer-mask passed only
    df_c = df[df["cancer_mask"] == True].copy()
    print(f"After Cancer-mask filter: {len(df_c)} patches")

    if args.virchow2_col not in df_c.columns:
        raise KeyError(f"{args.virchow2_col} not in CSV. "
                       f"Available: {[c for c in df_c.columns if 'p_gland' in c]}")

    rng = np.random.default_rng(args.seed)
    p = df_c[args.virchow2_col].values

    sample_dir = out_dir / "sample_patches_cancer_only"
    if sample_dir.exists():
        import shutil; shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Open SVS + Macenko (same target as training, so patches look as model saw them)
    slide = openslide.OpenSlide(svs_path)
    target_rgb = None
    normalizer = None
    if config.stain_normalize:
        t = cv2.imread(config.stain_target_path)
        if t is not None:
            target_rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
            normalizer = MacenkoNormalizer()
            normalizer.fit(target_rgb)
            print("Macenko stain normalization will be applied (same target as training)")

    manifest_rows = []
    for (cls, conf), (lo, hi) in BUCKETS.items():
        mask = (p >= lo) & (p < hi)
        subset = df_c[mask]
        print(f"\n[{cls:9} / {conf:15}] p in [{lo:.2f}, {hi:.2f}) → "
              f"{len(subset)} candidates", end="")
        if len(subset) == 0:
            print(" — skip (no candidates)")
            continue
        n_take = min(args.n_per_bucket, len(subset))
        idx = rng.choice(len(subset), size=n_take, replace=False)
        chosen = subset.iloc[idx].reset_index(drop=True)
        print(f" → {n_take} sampled")

        folder = sample_dir / conf / cls
        folder.mkdir(parents=True, exist_ok=True)

        for i, row in chosen.iterrows():
            x, y = int(row["x"]), int(row["y"])
            region = slide.read_region((x, y), 0, (config.patch_size, config.patch_size))
            patch_rgb = np.array(region.convert("RGB"))
            if normalizer is not None:
                try:
                    patch_rgb = normalizer.transform(patch_rgb)
                except Exception:
                    pass
            fname = f"{args.slide}_x{x}_y{y}.png"
            cv2.imwrite(str(folder / fname),
                        cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
            meta = {
                "filename": fname,
                "rel_path": f"{conf}/{cls}/{fname}",
                "confidence_bucket": conf,
                "predicted_class": cls,
                "x": x, "y": y,
                "p_gland_virchow2": float(row[args.virchow2_col]),
                "p_Cancer_kather": float(row.get("p_Cancer", float("nan"))),
                "p_TUM": float(row.get("p_TUM", float("nan"))),
                "p_STR": float(row.get("p_STR", float("nan"))),
                "gt_label": int(row["gt_label"]),
                "gt_class": "non-gland" if row["gt_label"] == 1 else "gland",
                "pred_label": int(row["pred_label"]),
                "correct_in_cancer_mask": bool(row["correct"]),
            }
            manifest_rows.append(meta)

    slide.close()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(sample_dir / "manifest.csv", index=False)
    print(f"\nManifest saved: {sample_dir}/manifest.csv ({len(manifest_df)} rows)")

    # README
    n_g_hi = sum(1 for r in manifest_rows if r["predicted_class"]=="gland" and r["confidence_bucket"]=="high_confidence")
    n_g_bd = sum(1 for r in manifest_rows if r["predicted_class"]=="gland" and r["confidence_bucket"]=="boundary")
    n_n_hi = sum(1 for r in manifest_rows if r["predicted_class"]=="non_gland" and r["confidence_bucket"]=="high_confidence")
    n_n_bd = sum(1 for r in manifest_rows if r["predicted_class"]=="non_gland" and r["confidence_bucket"]=="boundary")

    readme = sample_dir / "README.md"
    readme.write_text(f"""# Sample patches — {args.slide} (Cancer-only)

각 patch는 512×512 px, level 0 (40x, 0.2521 μm/px), Macenko stain normalized
(학습 시 사용된 동일 target).

대상: **Cancer mask (Kather TUM+STR ≥ {args.cancer_threshold})를 통과한 patch만**
       (normal·necrosis 제외)

## 폴더 구조

```
sample_patches_cancer_only/
├── high_confidence/
│   ├── gland/         ({n_g_hi}장)   ← Virchow2가 매우 confident하게 gland로 예측 (p>0.95)
│   └── non_gland/     ({n_n_hi}장)   ← Virchow2가 매우 confident하게 non-gland로 예측 (p<0.05)
├── boundary/
│   ├── gland/         ({n_g_bd}장)   ← gland로 예측했지만 애매 (0.50<p<0.70)
│   └── non_gland/     ({n_n_bd}장)   ← non-gland로 예측했지만 애매 (0.30<p<0.50)
└── manifest.csv       ({len(manifest_rows)}장 통합 메타데이터)
```

## 활용

- **`high_confidence/`** patches: 모델 신뢰도 검증 — 이게 명백히 틀려있으면 모델 자체 문제
- **`boundary/`** patches: 모델이 헷갈려하는 case — 교수님 재분류로 추가 학습 신호 확보 가능

## manifest.csv 컬럼

| 컬럼 | 의미 |
|---|---|
| filename | 파일명 |
| rel_path | 폴더 내 상대 경로 |
| confidence_bucket | high_confidence / boundary |
| predicted_class | gland / non_gland (Virchow2 결과) |
| x, y | slide level 0 좌표 |
| p_gland_virchow2 | Virchow2 P(gland) |
| p_Cancer_kather | Kather TUM+STR 확률 (cancer mask) |
| p_TUM, p_STR | Kather 개별 클래스 확률 |
| gt_label | GT label (1=non-gland, 0=gland) — XML 기준 |
| gt_class | gland / non-gland 문자열 |
| pred_label | prediction label (1=non-gland, 0=gland) |
| correct_in_cancer_mask | GT == prediction (cancer mask 안에서 평가) |

## 사용 가이드

각 patch는 파일명에 (x, y) slide-level 좌표가 포함되어 있어 ImageScope/QuPath에서 SVS에 위치 매핑 가능합니다.
""", encoding="utf-8")
    print(f"README: {readme}")

    print(f"\n=== Done. Total {len(manifest_rows)} patches in {sample_dir} ===")


if __name__ == "__main__":
    main()
