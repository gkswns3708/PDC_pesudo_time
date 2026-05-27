# Prediction summary — S14-1253-1-3

- Total tissue patches scanned: **7,072**
- Patch size: 512 | stride: 512

## Per-model patch counts (decision threshold p_gland=0.5)

| Model | n_patches | n_gland | n_non_gland | frac_gland | mean p_gland | median p_gland |
|---|---:|---:|---:|---:|---:|---:|
| virchow2 | 7,072 | 6,793 | 279 | 0.9605 | 0.9113 | 0.9804 |
| uni2 | 7,072 | 5,964 | 1,108 | 0.8433 | 0.7646 | 0.8482 |
| phikon-v2 | 7,072 | 6,783 | 289 | 0.9591 | 0.8077 | 0.8360 |
| ensemble_mean_prob | 7,072 | 6,826 | 246 | 0.9652 | 0.8279 | 0.8632 |
| ensemble_hard_vote | 7,072 | 6,767 | 305 | 0.9569 |  |  |

## Hard-prediction agreement

- All-3 unanimous: **82.65%** of patches
- virchow2 vs uni2: 86.27%
- virchow2 vs phikon-v2: 94.51%
- uni2 vs phikon-v2: 84.52%
