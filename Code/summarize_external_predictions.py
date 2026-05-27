"""
Summarize per-patch predictions from `infer_external_slide.py` for an external
slide (e.g., S14-2289-1-6).

Inputs (from /app/Gland_Seg/results/<slide>/):
  per_patch_predictions.csv  — x,y,p_gland_<model>...,p_gland_ensemble
  prob_map_<model>.npy       — pixel-level p_gland at thumbnail resolution
  prob_map_ensemble.npy      — mean-prob ensemble map (already in CSV too)
  thumbnail.npy              — RGB thumbnail
  slide_meta.npy             — dims, scale, models list
  annotation.npz             — (positive, negative) polygons in level-0 px

Outputs (under /app/Gland_Seg/results/<slide>/):
  prediction_summary.csv     — per-model + hardvote counts/ratios
  prediction_summary.md      — same as a markdown table for easy reading
  per_patch_predictions_with_hardvote.csv   — original CSV + each model's hard
                                              label + hardvote consensus
  prediction_overlay.png     — 5-panel viz: thumbnail+annot, per-model maps,
                               and hard-vote map; red=non-gland, blue=gland.

Usage:
    python summarize_external_predictions.py S14-2289-1-6
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLAND_LABEL = "gland"
NONGLAND_LABEL = "non-gland"
GLAND_COLOR = (0.15, 0.40, 0.95)       # blue
NONGLAND_COLOR = (0.90, 0.25, 0.20)    # red
ANNOT_COLOR = (1.00, 0.85, 0.10)       # yellow outline for annotation polygons


def hard_vote(pred_matrix):
    """pred_matrix: (N, M) of 0/1 ints, 0=gland 1=non-gland.
    Returns (N,) of 0/1 majority vote. Tie-broken toward gland (0)."""
    s = pred_matrix.sum(axis=1)
    n_models = pred_matrix.shape[1]
    # >n/2 means majority non-gland → 1, else 0
    return (s > n_models / 2).astype(np.int8)


def build_summary(df, models):
    rows = []
    n_total = len(df)
    for m in models:
        pred = (df[f"p_gland_{m}"] < 0.5).astype(np.int8)  # 1 = non-gland
        n_gland = int((pred == 0).sum())
        n_nong = int((pred == 1).sum())
        rows.append({
            "model": m,
            "n_patches": n_total,
            "n_gland": n_gland,
            "n_non_gland": n_nong,
            "frac_gland": n_gland / n_total,
            "mean_p_gland": float(df[f"p_gland_{m}"].mean()),
            "median_p_gland": float(df[f"p_gland_{m}"].median()),
        })
    # Mean-prob ensemble (already in CSV)
    if "p_gland_ensemble" in df.columns:
        pred = (df["p_gland_ensemble"] < 0.5).astype(np.int8)
        rows.append({
            "model": "ensemble_mean_prob",
            "n_patches": n_total,
            "n_gland": int((pred == 0).sum()),
            "n_non_gland": int((pred == 1).sum()),
            "frac_gland": float((pred == 0).mean()),
            "mean_p_gland": float(df["p_gland_ensemble"].mean()),
            "median_p_gland": float(df["p_gland_ensemble"].median()),
        })
    # Hard-voting ensemble
    pred_mat = np.stack([(df[f"p_gland_{m}"] < 0.5).astype(np.int8).values for m in models], axis=1)
    hv = hard_vote(pred_mat)
    df["pred_hardvote"] = np.where(hv == 0, GLAND_LABEL, NONGLAND_LABEL)
    rows.append({
        "model": "ensemble_hard_vote",
        "n_patches": n_total,
        "n_gland": int((hv == 0).sum()),
        "n_non_gland": int((hv == 1).sum()),
        "frac_gland": float((hv == 0).mean()),
        "mean_p_gland": np.nan,
        "median_p_gland": np.nan,
    })
    summary = pd.DataFrame(rows)

    # Pairwise agreement (on hard predictions)
    agree = {}
    pred_dict = {m: (df[f"p_gland_{m}"] < 0.5).astype(np.int8).values for m in models}
    for i, a in enumerate(models):
        for b in models[i+1:]:
            agree[f"{a} vs {b}"] = float((pred_dict[a] == pred_dict[b]).mean())
    all_agree = float(np.all(pred_mat == pred_mat[:, [0]], axis=1).mean())

    return summary, df, agree, all_agree, hv, pred_dict


def render_polys(ax, polys, scale, edge=ANNOT_COLOR, lw=0.6, ls="-"):
    for poly in polys:
        xs = poly[:, 0] / scale
        ys = poly[:, 1] / scale
        ax.plot(xs, ys, color=edge, linewidth=lw, linestyle=ls, alpha=0.9)


def draw_heatmap(ax, thumb_rgb, prob_map, valid, polys_pos, polys_neg, scale, title):
    ax.imshow(thumb_rgb)
    # 0=non-gland (red) … 1=gland (blue) — use RdBu_r with center at 0.5
    masked = np.ma.array(prob_map, mask=~valid)
    ax.imshow(masked, cmap="RdBu", vmin=0.0, vmax=1.0, alpha=0.55)
    render_polys(ax, polys_pos, scale, edge=ANNOT_COLOR, lw=0.6, ls="-")
    render_polys(ax, polys_neg, scale, edge=(1, 0.4, 0.7), lw=0.6, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def draw_hardvote(ax, thumb_rgb, hard_map, valid, polys_pos, polys_neg, scale, title):
    """hard_map: float32 same shape as thumbnail; 0=gland, 1=non-gland, NaN=invalid."""
    ax.imshow(thumb_rgb)
    masked = np.ma.array(hard_map, mask=~valid)
    # Use the same RdBu colormap (now binary): 0→blue, 1→red
    ax.imshow(masked, cmap="RdBu", vmin=0.0, vmax=1.0, alpha=0.55)
    render_polys(ax, polys_pos, scale, edge=ANNOT_COLOR, lw=0.6, ls="-")
    render_polys(ax, polys_neg, scale, edge=(1, 0.4, 0.7), lw=0.6, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def build_hardvote_pixel_map(xs, ys, hv_labels, H, W, patch_size, scale):
    """Same accumulator as build_pixel_prob_map but for binary hard-vote labels."""
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for x, y, lab in zip(xs, ys, hv_labels):
        tx0 = int(x / scale); ty0 = int(y / scale)
        tx1 = min(int((x + patch_size) / scale), W)
        ty1 = min(int((y + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += float(lab)  # 0 or 1
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    # avg ∈ [0,1]: fraction of overlapping patches voted non-gland.
    # Threshold 0.5 → final pixel hard-vote label.
    return avg, count > 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_external_predictions.py <slide_name>")
        sys.exit(1)
    slide = sys.argv[1]
    base = Path("/app/Gland_Seg") / "results" / slide
    if not base.exists():
        sys.exit(f"results dir not found: {base}")

    csv_path = base / "per_patch_predictions.csv"
    df = pd.read_csv(csv_path)
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    models = list(meta["models"])
    print(f"Models: {models}")
    print(f"Patches: {len(df)}")

    summary, df_aug, pairwise, all_agree, hv, pred_dict = build_summary(df, models)

    # Save augmented CSV (with each model's hard label + hardvote)
    for m in models:
        df_aug[f"pred_{m}"] = np.where((df_aug[f"p_gland_{m}"] >= 0.5), GLAND_LABEL, NONGLAND_LABEL)
    out_csv = base / "per_patch_predictions_with_hardvote.csv"
    df_aug.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")

    # Save summary CSV + Markdown
    summary_path = base / "prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  → {summary_path}")

    md_lines = [f"# Prediction summary — {slide}", "",
                f"- Total tissue patches scanned: **{len(df):,}**",
                f"- Patch size: {meta.get('patch_size', '?')} | stride: {meta.get('stride', '?')}",
                "", "## Per-model patch counts (decision threshold p_gland=0.5)", ""]
    md_lines.append("| Model | n_patches | n_gland | n_non_gland | frac_gland | mean p_gland | median p_gland |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        mp = "" if pd.isna(r["mean_p_gland"]) else f"{r['mean_p_gland']:.4f}"
        md = "" if pd.isna(r["median_p_gland"]) else f"{r['median_p_gland']:.4f}"
        md_lines.append(f"| {r['model']} | {r['n_patches']:,} | {r['n_gland']:,} | "
                        f"{r['n_non_gland']:,} | {r['frac_gland']:.4f} | {mp} | {md} |")
    md_lines += ["", "## Hard-prediction agreement", ""]
    md_lines.append(f"- All-3 unanimous: **{all_agree*100:.2f}%** of patches")
    for k, v in pairwise.items():
        md_lines.append(f"- {k}: {v*100:.2f}%")
    md_path = base / "prediction_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  → {md_path}")
    print()
    print("\n".join(md_lines))

    # ── Visualization ──
    thumb_rgb = np.load(base / "thumbnail.npy")
    scale = meta["scale"]
    H, W = thumb_rgb.shape[:2]
    ann = np.load(base / "annotation.npz", allow_pickle=True)
    polys_pos = list(ann["positive"])
    polys_neg = list(ann["negative"])

    # Per-model pixel maps
    panels = []
    for m in models:
        pm = np.load(base / f"prob_map_{m}.npy")
        vm = np.load(base / f"valid_mask_{m}.npy")
        panels.append((m, pm, vm))

    # Hard-vote pixel map (built from per-patch hardvote labels)
    xs = df_aug["x"].values
    ys = df_aug["y"].values
    hv_pixel, hv_valid = build_hardvote_pixel_map(
        xs, ys, hv, H, W, meta["patch_size"], scale)
    np.save(base / "hardvote_pixel_map.npy", hv_pixel)
    np.save(base / "hardvote_valid_mask.npy", hv_valid)

    # Mean-prob ensemble map already saved by infer script
    ens_map = np.load(base / "prob_map_ensemble.npy")
    ens_valid = np.load(base / "valid_mask_ensemble.npy")

    n_models = len(models)
    n_cols = 3
    n_rows = 2 + max(0, (n_models - 1)) // n_cols  # we want: thumb, models..., mean-ens, hardvote
    panels_to_draw = []
    panels_to_draw.append(("thumb_annot", None, None))
    for m, pm, vm in panels:
        panels_to_draw.append((f"model:{m}", pm, vm))
    panels_to_draw.append(("ensemble_mean", ens_map, ens_valid))
    panels_to_draw.append(("hardvote", hv_pixel, hv_valid))

    n = len(panels_to_draw)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (kind, pm, vm) in zip(axes, panels_to_draw):
        if kind == "thumb_annot":
            ax.imshow(thumb_rgb)
            render_polys(ax, polys_pos, scale, edge=ANNOT_COLOR, lw=0.6, ls="-")
            render_polys(ax, polys_neg, scale, edge=(1, 0.4, 0.7), lw=0.6, ls="--")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{slide} — thumbnail + annotation\n"
                         f"{len(polys_pos)} positive, {len(polys_neg)} negative polygons",
                         fontsize=10)
        elif kind.startswith("model:"):
            m = kind.split(":", 1)[1]
            row = summary[summary["model"] == m].iloc[0]
            draw_heatmap(ax, thumb_rgb, pm, vm, polys_pos, polys_neg, scale,
                         f"{m} — p(gland)\n"
                         f"gland={row['n_gland']:,}  non-gland={row['n_non_gland']:,}  "
                         f"frac_gland={row['frac_gland']:.3f}")
        elif kind == "ensemble_mean":
            row = summary[summary["model"] == "ensemble_mean_prob"].iloc[0]
            draw_heatmap(ax, thumb_rgb, pm, vm, polys_pos, polys_neg, scale,
                         f"Ensemble (mean prob)\n"
                         f"gland={row['n_gland']:,}  non-gland={row['n_non_gland']:,}  "
                         f"frac_gland={row['frac_gland']:.3f}")
        elif kind == "hardvote":
            row = summary[summary["model"] == "ensemble_hard_vote"].iloc[0]
            draw_hardvote(ax, thumb_rgb, pm, vm, polys_pos, polys_neg, scale,
                          f"Hard voting (3 models)\n"
                          f"gland={row['n_gland']:,}  non-gland={row['n_non_gland']:,}  "
                          f"frac_gland={row['frac_gland']:.3f}")

    # Hide leftover axes
    for j in range(len(panels_to_draw), len(axes)):
        axes[j].axis("off")

    # Legend on the first axis
    handles = [
        mpatches.Patch(color=GLAND_COLOR, label="gland (high p)"),
        mpatches.Patch(color=NONGLAND_COLOR, label="non-gland (low p)"),
        mpatches.Patch(facecolor="none", edgecolor=ANNOT_COLOR,
                       label="annotation polygon (positive)"),
        mpatches.Patch(facecolor="none", edgecolor=(1, 0.4, 0.7),
                       label="annotation polygon (negative ROA)", linestyle="--"),
    ]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"{slide} — per-model predictions & hard-voting ensemble",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = base / "prediction_overlay.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved viz → {out_png}")


if __name__ == "__main__":
    main()
