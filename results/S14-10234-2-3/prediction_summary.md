# Prediction summary — S14-10234-2-3

- Total tissue patches scanned: **9,327**
- Patch size: 512 | stride: 512

## Per-model patch counts (decision threshold p_gland=0.5)

| Model | n_patches | n_gland | n_non_gland | frac_gland | mean p_gland | median p_gland |
|---|---:|---:|---:|---:|---:|---:|
| virchow2 | 9,327 | 6,371 | 2,956 | 0.6831 | 0.6445 | 0.7652 |
| uni2 | 9,327 | 4,667 | 4,660 | 0.5004 | 0.4938 | 0.5001 |
| phikon-v2 | 9,327 | 5,830 | 3,497 | 0.6251 | 0.6043 | 0.6384 |
| ensemble_mean_prob | 9,327 | 5,859 | 3,468 | 0.6282 | 0.5808 | 0.6162 |
| ensemble_hard_vote | 9,327 | 5,738 | 3,589 | 0.6152 |  |  |

## Hard-prediction agreement

- All-3 unanimous: **62.24%** of patches
- virchow2 vs uni2: 74.78%
- virchow2 vs phikon-v2: 77.20%
- uni2 vs phikon-v2: 72.50%
