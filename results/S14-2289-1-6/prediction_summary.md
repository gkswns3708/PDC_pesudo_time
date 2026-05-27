# Prediction summary — S14-2289-1-6

- Total tissue patches scanned: **9,605**
- Patch size: 512 | stride: 512

## Per-model patch counts (decision threshold p_gland=0.5)

| Model | n_patches | n_gland | n_non_gland | frac_gland | mean p_gland | median p_gland |
|---|---:|---:|---:|---:|---:|---:|
| virchow2 | 9,605 | 7,529 | 2,076 | 0.7839 | 0.7571 | 0.9258 |
| uni2 | 9,605 | 8,986 | 619 | 0.9356 | 0.8759 | 0.9585 |
| phikon-v2 | 9,605 | 9,239 | 366 | 0.9619 | 0.8777 | 0.9395 |
| ensemble_mean_prob | 9,605 | 8,801 | 804 | 0.9163 | 0.8369 | 0.9143 |
| ensemble_hard_vote | 9,605 | 8,924 | 681 | 0.9291 |  |  |

## Hard-prediction agreement

- All-3 unanimous: **77.87%** of patches
- virchow2 vs uni2: 82.67%
- virchow2 vs phikon-v2: 80.36%
- uni2 vs phikon-v2: 92.70%

## S14-2289-1-6 GT 기준 성능

교수님 XML annotation을 GT로 사용 (parity 규칙: ROI 박스 안 단일 polygon → gland, 중첩 polygon 안 → non-gland).

- GT 가진 패치: **2,802** (gland 2,626 / non-gland 176). ROI 밖 6,803 패치는 평가 제외.

| Source | n_eval | accuracy | F1 (gland) | F1 (non-gland) | **macro-F1** |
|---|---:|---:|---:|---:|---:|
| virchow2 | 2,802 | 0.8041 | 0.8880 | 0.2191 | **0.5535** |
| uni2 | 2,802 | 0.9126 | 0.9538 | 0.1967 | **0.5752** |
| phikon-v2 | 2,802 | 0.9193 | 0.9572 | 0.3110 | **0.6341** |
| ensemble_mean_prob | 2,802 | 0.9036 | 0.9483 | 0.2932 | **0.6207** |
| hardvote | 2,802 | 0.9090 | 0.9514 | 0.2897 | **0.6205** |

Precision / recall 등 세부 지표는 `metrics_vs_gt.csv` 참고.
