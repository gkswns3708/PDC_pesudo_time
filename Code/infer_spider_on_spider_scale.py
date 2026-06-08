"""
Stage 2 — SPIDER inference on the SPIDER-FOV-scale grid built in Stage 1.

Reads per_patch_grid_spider_scale.csv, runs histai/SPIDER-colorectal-model on
every tissue-positive patch (regardless of gt label, so we keep "skip" rows for
visualization later), and saves predictions.

Reuses patch_to_spider_input and spider_preprocess from infer_spider_on_eval,
overriding the patch size to 2240 (level-0) instead of the old 512.

Usage:
    HF_TOKEN=... /root/miniconda3/envs/tiatoolbox/bin/python \
        infer_spider_on_spider_scale.py S14-2289-1-6 --batch_size 4 --device cuda:0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from infer_spider_on_eval import (
    SPIDER_REPO,
    patch_to_spider_input,
    spider_preprocess,
)

PATCH_SIZE_LEVEL0 = 2240  # SPIDER-scale grid (same as Stage 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    cfg = Config()
    results_dir = Path(cfg.base_dir) / "results" / args.slide
    grid_csv = results_dir / "per_patch_grid_spider_scale.csv"
    if not grid_csv.exists():
        sys.exit(f"missing {grid_csv}  (run build_spider_scale_grid.py first)")
    df = pd.read_csv(grid_csv)
    df = df[df.is_tissue == 1].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"[input] {len(df):,} tissue patches from {grid_csv.name}")

    svs_path = Path(cfg.svs_dir) / f"{args.slide}.svs"
    if not svs_path.exists():
        ext = cfg.external_test_slides.get(args.slide, {})
        if ext.get("svs"):
            svs_path = Path(cfg.svs_dir) / ext["svs"]
    if not svs_path.exists():
        sys.exit(f"missing svs: {svs_path}")
    slide = openslide.OpenSlide(str(svs_path))
    print(f"[slide] {svs_path.name}  dim={slide.level_dimensions[0]}  "
          f"mpp={slide.properties.get('openslide.mpp-x')}")

    print(f"[load] {SPIDER_REPO}  (device={args.device})")
    t0 = time.time()
    model = AutoModel.from_pretrained(SPIDER_REPO, trust_remote_code=True, token=args.token)
    model.eval().to(args.device)
    class_names = list(model.config.class_names)
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}
    idx_high = label2id.get("Adenocarcinoma high grade")
    idx_low = label2id.get("Adenocarcinoma low grade")
    print(f"[load] done {time.time()-t0:.1f}s.  {len(id2label)} classes  "
          f"idx_high={idx_high}  idx_low={idx_low}")

    half = PATCH_SIZE_LEVEL0 // 2
    all_probs = np.zeros((len(df), len(id2label)), dtype=np.float32)
    top1_idx = np.zeros(len(df), dtype=np.int32)

    batch_imgs, batch_rows = [], []
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        cx = int(row["x"]) + half
        cy = int(row["y"]) + half
        pil = patch_to_spider_input(slide, cx, cy,
                                    level0_size=PATCH_SIZE_LEVEL0,
                                    out_size=1120)
        batch_imgs.append(pil)
        batch_rows.append(i)
        if len(batch_imgs) >= args.batch_size or i == len(df) - 1:
            pixel_values = spider_preprocess(batch_imgs).to(args.device)
            with torch.inference_mode():
                out = model(pixel_values=pixel_values)
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, dict):
                logits = out.get("logits")
            if logits is None:
                logits = out[0] if isinstance(out, (tuple, list)) else None
            if logits is None:
                sys.exit(f"could not extract logits: {type(out)}")
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for r, p in zip(batch_rows, probs):
                all_probs[r] = p
                top1_idx[r] = int(p.argmax())
            batch_imgs.clear()
            batch_rows.clear()
            if (i + 1) % 100 == 0 or i == len(df) - 1:
                rate = (i + 1) / (time.time() - t_start)
                eta = (len(df) - i - 1) / max(rate, 1e-9)
                print(f"  [{i+1:>4,}/{len(df):,}]  {rate:.1f} p/s  ETA {eta/60:.1f} min",
                      flush=True)

    slide.close()

    out_df = pd.DataFrame({
        "x": df["x"].values,
        "y": df["y"].values,
        "gt_label_thr050": df["gt_label_thr050"].values,
        "gt_label_thr025": df["gt_label_thr025"].values,
        "pct_inROI": df["pct_inROI"].values,
        "pct_nested": df["pct_nested"].values,
        "top1_idx": top1_idx,
        "top1_class": [id2label[i] for i in top1_idx],
        "p_high_grade": all_probs[:, idx_high] if idx_high is not None else np.nan,
        "p_low_grade": all_probs[:, idx_low] if idx_low is not None else np.nan,
    })
    if idx_high is not None:
        out_df["pred_binary"] = (out_df["top1_class"] == "Adenocarcinoma high grade").astype(int)
        out_df["pred_binary_softrenorm"] = (out_df["p_high_grade"] > out_df["p_low_grade"]).astype(int)
    out_path = results_dir / "per_patch_predictions_spider_scale.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")

    full = pd.DataFrame(all_probs, columns=[f"p_{id2label[i].replace(' ','_')}"
                                            for i in range(len(id2label))])
    full.insert(0, "y", df["y"].values)
    full.insert(0, "x", df["x"].values)
    full_path = results_dir / "per_patch_predictions_spider_scale_full.csv"
    full.to_csv(full_path, index=False)
    print(f"[save] {full_path}")


if __name__ == "__main__":
    main()
