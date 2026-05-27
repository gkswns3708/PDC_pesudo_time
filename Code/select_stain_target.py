"""
Select a representative target patch for Macenko stain normalization.

Picks the patch whose tissue mean RGB is closest to the global mean across
all slides. Saves the chosen patch as `Gland_Seg/Data/stain_target.png`
and a preview PNG with stats.
"""

from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

from config import Config


N_CANDIDATES_PER_SLIDE = 50
RANDOM_SEED = 42


def tissue_mean_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    tissue = hsv[..., 1] > 20
    if tissue.sum() < 500:
        return None
    return rgb[tissue].mean(axis=0)


def main():
    config = Config()
    rng = np.random.default_rng(RANDOM_SEED)
    # Target selection uses the raw (non-normalized) patches — we pick a real tissue
    # sample as the reference to normalize toward. Hardcode the original patches dir.
    src_patches_dir = Path("/app/Gland_Seg/patches")
    df = pd.read_csv(src_patches_dir / "metadata.csv")

    # Sample candidates per slide
    candidates = []
    for slide, group in df.groupby("slide"):
        idx = rng.choice(len(group), size=min(N_CANDIDATES_PER_SLIDE, len(group)),
                         replace=False)
        subset = group.iloc[idx]
        for _, row in subset.iterrows():
            p = src_patches_dir / slide / row["class"] / row["filename"]
            mean = tissue_mean_rgb(p)
            if mean is not None:
                candidates.append((p, slide, mean))

    all_means = np.stack([m for _, _, m in candidates])
    global_mean = all_means.mean(axis=0)
    print(f"Global tissue RGB mean across {len(candidates)} candidates: "
          f"{global_mean.round(1)}")

    # Distance from global mean
    dists = np.linalg.norm(all_means - global_mean, axis=1)
    best_idx = int(np.argmin(dists))
    best_path, best_slide, best_mean = candidates[best_idx]
    print(f"Selected target patch:")
    print(f"  Path:  {best_path}")
    print(f"  Slide: {best_slide}")
    print(f"  Mean RGB: {best_mean.round(1)}")
    print(f"  Distance from global mean: {dists[best_idx]:.2f}")

    target_path = Path(config.data_dir) / "stain_target.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, target_path)
    print(f"\nSaved target patch to: {target_path}")


if __name__ == "__main__":
    main()
