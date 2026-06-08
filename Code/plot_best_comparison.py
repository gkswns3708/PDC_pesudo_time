"""Plot best-by-ext F1 values per backbone (bar chart, no training curves).

Output: /app/Gland_Seg/Viz/best_comparison_20x_alignment.png
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/app/Gland_Seg/Viz/best_comparison_20x_alignment.png")

backbones = ["phikon-v2", "uni2", "virchow2", "hibou-l"]
colors = {
    "phikon-v2": "#d62728",
    "uni2":      "#2ca02c",
    "virchow2":  "#1f77b4",
    "hibou-l":   "#ff7f0e",
}

# Best-by-ext per backbone (best setup chosen per model — see _setup_ in legend)
# Hibou-L uses Raw+CE (best); others use Macenko+CE (best so far)
best_224_20x = {
    "phikon-v2": dict(macro=0.6359, gland=0.9262, non_gland=0.3456, setup="Macenko+CE"),
    "uni2":      dict(macro=0.6082, gland=0.9201, non_gland=0.3098, setup="Macenko+CE"),
    "virchow2":  dict(macro=0.5486, gland=0.8664, non_gland=0.2387, setup="Macenko+CE"),
    # Hibou-L Raw+CE peak ep11 i200 macro=0.6390; per-class nearest captured at ep7 = (0.9188, 0.3466)
    "hibou-l":   dict(macro=0.6390, gland=0.9188, non_gland=0.3466, setup="Raw+CE (best)"),
}
# Base run (256/40x -> infer 512) reference
base_run = {
    "phikon-v2": dict(macro=0.634, gland=0.957, non_gland=0.311),
    "uni2":      dict(macro=0.575, gland=0.954, non_gland=0.197),
    "virchow2":  dict(macro=0.554, gland=0.888, non_gland=0.219),
}

metrics = [("macro", "macro-F1"), ("gland", "gland F1"), ("non_gland", "non-gland F1")]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(backbones))
bar_w = 0.36

for ax, (mkey, mlabel) in zip(axes, metrics):
    new_vals  = [best_224_20x[bb][mkey] for bb in backbones]
    base_vals = [base_run.get(bb, {}).get(mkey, np.nan) for bb in backbones]
    bar_colors = [colors[bb] for bb in backbones]

    # base (lighter / hatched) on the left, 224_20x (solid) on the right
    b1 = ax.bar(x - bar_w/2, base_vals, width=bar_w,
                color=bar_colors, alpha=0.35, edgecolor="gray",
                hatch="//", label="Base (256/40x → infer 512)")
    b2 = ax.bar(x + bar_w/2, new_vals, width=bar_w,
                color=bar_colors, edgecolor="black", linewidth=0.8,
                label="224_20x (aligned)")

    # value labels above bars
    for rect, v in zip(b1, base_vals):
        if not np.isnan(v):
            ax.text(rect.get_x() + rect.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="gray")
    for rect, v in zip(b2, new_vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # delta arrow for each model where base exists
    for i, bb in enumerate(backbones):
        if bb not in base_run:
            continue
        d = new_vals[i] - base_vals[i]
        sign = "+" if d >= 0 else ""
        ax.annotate(f"Δ {sign}{d:+.03f}".replace("++", "+"),
                    xy=(x[i], max(new_vals[i], base_vals[i]) + 0.04),
                    ha="center", fontsize=8,
                    color=("green" if d > 0 else "red"))

    tick_labels = [f"{bb}\n({best_224_20x[bb]['setup']})" for bb in backbones]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(f"External {mlabel}")
    ax.set_title(f"External {mlabel} — best-by-ext")
    ax.set_ylim(0, max(new_vals + base_vals + [0.0]) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("CRC gland/non-gland · 20× MPP-aligned · Best-by-ext per backbone "
             "(S14-2289-1-6 external; setup picked per backbone)",
             fontsize=12, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"saved: {OUT}")
