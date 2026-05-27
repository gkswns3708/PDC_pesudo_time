"""
P0 diagnostic: per-slide color distribution.

Samples patches from each slide, computes RGB/HSV channel statistics,
and plots distributions to visually assess stain variation between slides.

If S14-248-1-3 (Fold 4 val) is a clear outlier in color space, the Fold 4
failure is largely explained by stain shift → stain normalization (P1) is
the right fix.

Output: Gland_Seg/Viz/slide_color_stats.png
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config


N_SAMPLES_PER_SLIDE = 300  # random patches per slide for stats
RANDOM_SEED = 42


def sample_patch_paths(metadata_df, patches_dir, n_per_slide, rng):
    """Return dict: slide_name -> list of patch paths."""
    sampled = {}
    for slide, group in metadata_df.groupby("slide"):
        paths = [Path(patches_dir) / slide / row["class"] / row["filename"]
                 for _, row in group.iterrows()]
        if len(paths) > n_per_slide:
            idx = rng.choice(len(paths), size=n_per_slide, replace=False)
            paths = [paths[i] for i in idx]
        sampled[slide] = paths
    return sampled


def compute_stats(paths):
    """Load patches, compute mean RGB and HSV per patch.

    Returns:
        rgb: (N, 3) mean RGB per patch
        hsv: (N, 3) mean HSV per patch
    """
    rgb_means = []
    hsv_means = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Tissue mask to exclude white background in the mean
        hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        tissue = hsv_img[..., 1] > 20
        if tissue.sum() < 100:
            continue
        rgb_means.append(rgb[tissue].mean(axis=0))
        hsv_means.append(hsv_img[tissue].mean(axis=0))
    return np.array(rgb_means), np.array(hsv_means)


def plot_distributions(slide_stats, config, save_path):
    """Plot per-slide histograms for each channel."""
    slides = list(slide_stats.keys())
    # Color code: gland=blue-ish, non-gland=red-ish, with distinct hues per slide
    palette = plt.get_cmap("tab10")
    slide_colors = {s: palette(i) for i, s in enumerate(slides)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    channels_rgb = ["R", "G", "B"]
    channels_hsv = ["H", "S", "V"]

    for ch_idx in range(3):
        ax = axes[0, ch_idx]
        for slide in slides:
            rgb, _ = slide_stats[slide]
            cls = config.slides[slide]["class"]
            linestyle = "-" if cls == "gland" else "--"
            ax.hist(rgb[:, ch_idx], bins=40, alpha=0.4,
                    color=slide_colors[slide], density=True,
                    label=f"{slide} ({cls})", histtype="stepfilled",
                    linestyle=linestyle, edgecolor=slide_colors[slide], linewidth=1.5)
        ax.set_title(f"RGB channel: {channels_rgb[ch_idx]}")
        ax.set_xlabel("mean value (0-255)")
        ax.set_ylabel("density")
        if ch_idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    for ch_idx in range(3):
        ax = axes[1, ch_idx]
        for slide in slides:
            _, hsv = slide_stats[slide]
            cls = config.slides[slide]["class"]
            linestyle = "-" if cls == "gland" else "--"
            ax.hist(hsv[:, ch_idx], bins=40, alpha=0.4,
                    color=slide_colors[slide], density=True,
                    label=f"{slide} ({cls})", histtype="stepfilled",
                    linestyle=linestyle, edgecolor=slide_colors[slide], linewidth=1.5)
        ax.set_title(f"HSV channel: {channels_hsv[ch_idx]}")
        ax.set_xlabel("mean value")
        ax.set_ylabel("density")

    plt.suptitle("Per-slide mean color distributions (tissue pixels only)\n"
                 "solid line = gland,  dashed = non-gland", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def plot_rgb_scatter(slide_stats, config, save_path):
    """2D scatter: mean R vs mean B per patch, colored per slide."""
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = plt.get_cmap("tab10")
    for i, (slide, (rgb, _)) in enumerate(slide_stats.items()):
        cls = config.slides[slide]["class"]
        marker = "o" if cls == "gland" else "x"
        ax.scatter(rgb[:, 0], rgb[:, 2], s=8, alpha=0.5,
                   c=[palette(i)], marker=marker, label=f"{slide} ({cls})")
    ax.set_xlabel("mean R (tissue)")
    ax.set_ylabel("mean B (tissue)")
    ax.set_title("Per-patch mean R vs B — slide clustering in color space\n"
                 "o = gland, x = non-gland")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def main():
    config = Config()
    rng = np.random.default_rng(RANDOM_SEED)

    metadata_path = Path(config.output_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} patches from {metadata_path}")

    print(f"Sampling up to {N_SAMPLES_PER_SLIDE} patches per slide...")
    sampled = sample_patch_paths(df, config.output_dir, N_SAMPLES_PER_SLIDE, rng)

    slide_stats = {}
    for slide, paths in sampled.items():
        print(f"  {slide}: computing stats on {len(paths)} patches...")
        rgb, hsv = compute_stats(paths)
        slide_stats[slide] = (rgb, hsv)
        print(f"    RGB mean = {rgb.mean(axis=0).round(1)}, "
              f"HSV mean = {hsv.mean(axis=0).round(1)}")

    # Print summary table
    print("\nPer-slide RGB/HSV means (tissue pixels only):")
    print(f"{'slide':<16} {'class':<10} {'R':>6} {'G':>6} {'B':>6} "
          f"{'H':>6} {'S':>6} {'V':>6}")
    print("-" * 70)
    for slide, (rgb, hsv) in slide_stats.items():
        cls = config.slides[slide]["class"]
        r, g, b = rgb.mean(axis=0)
        h, s, v = hsv.mean(axis=0)
        print(f"{slide:<16} {cls:<10} {r:>6.1f} {g:>6.1f} {b:>6.1f} "
              f"{h:>6.1f} {s:>6.1f} {v:>6.1f}")

    viz_dir = Path(config.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    hist_path = viz_dir / "slide_color_stats.png"
    plot_distributions(slide_stats, config, hist_path)
    print(f"\nSaved: {hist_path}")

    scatter_path = viz_dir / "slide_color_scatter.png"
    plot_rgb_scatter(slide_stats, config, scatter_path)
    print(f"Saved: {scatter_path}")


if __name__ == "__main__":
    main()
