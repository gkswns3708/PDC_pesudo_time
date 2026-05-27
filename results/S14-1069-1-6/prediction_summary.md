# Prediction summary — S14-1069-1-6

- Total tissue patches scanned: **4,541**
- Patch size: 512 | stride: 512

## Per-model patch counts (decision threshold p_gland=0.5)

| Model | n_patches | n_gland | n_non_gland | frac_gland | mean p_gland | median p_gland |
|---|---:|---:|---:|---:|---:|---:|
| virchow2 | 4,541 | 2,577 | 1,964 | 0.5675 | 0.5580 | 0.6016 |
| uni2 | 4,541 | 2,035 | 2,506 | 0.4481 | 0.4704 | 0.4481 |
| phikon-v2 | 4,541 | 2,244 | 2,297 | 0.4942 | 0.4844 | 0.4960 |
| ensemble_mean_prob | 4,541 | 2,307 | 2,234 | 0.5080 | 0.5043 | 0.5057 |
| ensemble_hard_vote | 4,541 | 2,278 | 2,263 | 0.5017 |  |  |

## Hard-prediction agreement

- All-3 unanimous: **56.75%** of patches
- virchow2 vs uni2: 74.19%
- virchow2 vs phikon-v2: 68.75%
- uni2 vs phikon-v2: 70.56%
