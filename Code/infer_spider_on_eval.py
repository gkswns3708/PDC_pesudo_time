"""
Run HistAI SPIDER-colorectal-model inference on our external eval slide
(default S14-2289-1-6) and produce per-patch predictions aligned with the
existing per_patch_predictions_with_hardvote.csv so compute_gt_metrics-style
F1 is directly comparable.

Pipeline per patch:
  1. Read (x, y) = level-0 top-left of the 512px training-time patch
  2. Compute center = (x + 256, y + 256) at level 0 (40x, 0.252 µm/px)
  3. Read 2240x2240 RGB region from level 0 centered on the patch
     (boundary-cropped + padded if near edge)
  4. Downsample to 1120x1120 (bilinear) -> simulates 20x @ ~0.504 µm/px which
     is what SPIDER center+context grid expects
  5. Feed to SPIDER processor + model -> 14 softmax probs + top1
  6. Save:
       per_patch_predictions_spider.csv
         columns: x, y, top1_idx, top1_class, p_high_grade, p_low_grade,
                  pred_binary  (1=non-gland if top1==Adeno_high_grade else 0)
       per_patch_predictions_spider_full.csv
         + all 14 class probs

Then run compute_gt_metrics_spider.py to score against ROI parity GT.

Usage:
    HF_TOKEN=... \
    /root/miniconda3/envs/tiatoolbox/bin/python infer_spider_on_eval.py \
        S14-2289-1-6 --batch_size 4 --device cuda:0
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
from PIL import Image
from transformers import AutoModel

from config import Config

SPIDER_REPO = "histai/SPIDER-colorectal-model"

# preprocessor_config.json on the HF repo (pinned to avoid transformers-version drift):
# do_resize=false, do_center_crop with crop_size 1120, rescale 1/255, normalize with
# Hibou-style H&E mean/std. We re-implement these 4 steps manually so we don't depend on
# the ViTImageProcessorFast class (transformers >= 4.47).
SPIDER_IMG_MEAN = np.array([0.7068, 0.5755, 0.722], dtype=np.float32)
SPIDER_IMG_STD = np.array([0.195, 0.2316, 0.1816], dtype=np.float32)


def spider_preprocess(pil_imgs):
    """List[PIL.Image] (1120x1120 RGB) -> Tensor (N, 3, 1120, 1120) normalized."""
    arr = np.stack([np.asarray(p, dtype=np.float32) for p in pil_imgs])  # (N, 1120, 1120, 3)
    arr /= 255.0
    arr = (arr - SPIDER_IMG_MEAN) / SPIDER_IMG_STD
    arr = np.transpose(arr, (0, 3, 1, 2))  # (N, 3, H, W)
    return torch.from_numpy(arr)


def patch_to_spider_input(slide, cx, cy, level0_size=2240, out_size=1120):
    """Read level-0 RGB region centered on (cx, cy), pad if at boundary, resize to out_size."""
    w, h = slide.level_dimensions[0]
    half = level0_size // 2
    x0 = cx - half
    y0 = cy - half
    # bound the read
    rx0 = max(0, x0)
    ry0 = max(0, y0)
    rx1 = min(w, x0 + level0_size)
    ry1 = min(h, y0 + level0_size)
    read_w = rx1 - rx0
    read_h = ry1 - ry0
    rgba = slide.read_region((rx0, ry0), 0, (read_w, read_h)).convert("RGB")
    arr = np.asarray(rgba, dtype=np.uint8)
    # pad to (level0_size, level0_size) with white if cropped
    if (read_w, read_h) != (level0_size, level0_size):
        pad = np.full((level0_size, level0_size, 3), 255, dtype=np.uint8)
        py0 = ry0 - y0
        px0 = rx0 - x0
        pad[py0:py0 + read_h, px0:px0 + read_w] = arr
        arr = pad
    # resize 2240 -> 1120 (PIL bilinear)
    pil = Image.fromarray(arr).resize((out_size, out_size), Image.BILINEAR)
    return pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slide", help="slide name (e.g. S14-2289-1-6)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="evaluate first N patches only (debug)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    config = Config()
    results_dir = Path(config.base_dir) / "results" / args.slide
    src_csv = results_dir / "per_patch_predictions_with_hardvote.csv"
    if not src_csv.exists():
        sys.exit(f"Missing source CSV: {src_csv}")
    df = pd.read_csv(src_csv)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"[input] {len(df):,} patches from {src_csv}")

    # slide
    svs_path = Path(config.svs_dir) / f"{args.slide}.svs"
    if not svs_path.exists():
        # external slide may be elsewhere — try external_test_slides config
        ext = config.external_test_slides.get(args.slide, {})
        if ext.get("svs"):
            svs_path = Path(config.svs_dir) / ext["svs"]
    if not svs_path.exists():
        sys.exit(f"Missing svs: {svs_path}")
    slide = openslide.OpenSlide(str(svs_path))
    print(f"[slide] {svs_path.name}  dim={slide.level_dimensions[0]}  mpp={slide.properties.get('openslide.mpp-x')}")

    # model + processor
    print(f"[load] {SPIDER_REPO} (device={args.device})")
    t0 = time.time()
    model = AutoModel.from_pretrained(SPIDER_REPO, trust_remote_code=True, token=args.token)
    model.eval().to(args.device)
    # SPIDER stores real class names in config.class_names; id2label is just LABEL_n dummies
    class_names = list(model.config.class_names)
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}
    idx_high = label2id.get("Adenocarcinoma high grade")
    idx_low = label2id.get("Adenocarcinoma low grade")
    if idx_high is None or idx_low is None:
        print(f"[warn] could not find adenocarcinoma high/low classes; available: {class_names}")
    print(f"[load] done in {time.time()-t0:.1f}s. {len(id2label)} classes")
    print(f"[load] idx_high={idx_high} ({id2label.get(idx_high)}), idx_low={idx_low} ({id2label.get(idx_low)})")

    # patch center derivation — uses old eval's 512 patch_size
    # (per_patch_predictions_with_hardvote.csv stores level-0 top-left of 512-patch)
    PATCH_SIZE_LEVEL0 = 512
    half_patch = PATCH_SIZE_LEVEL0 // 2

    all_probs = np.zeros((len(df), len(id2label)), dtype=np.float32)
    top1_idx = np.zeros(len(df), dtype=np.int32)

    batch_imgs = []
    batch_rows = []
    t_start = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        cx = int(row["x"]) + half_patch
        cy = int(row["y"]) + half_patch
        pil = patch_to_spider_input(slide, cx, cy)
        batch_imgs.append(pil)
        batch_rows.append(i)
        if len(batch_imgs) >= args.batch_size or i == len(df) - 1:
            pixel_values = spider_preprocess(batch_imgs).to(args.device)
            with torch.inference_mode():
                out = model(pixel_values=pixel_values)
            # SPIDER model returns either logits or a dict with "logits"
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, dict):
                logits = out.get("logits")
            if logits is None:
                # Try predicted_class_names path + fallback to raw model.classifier output
                logits = out[0] if isinstance(out, (tuple, list)) else None
            if logits is None:
                sys.exit(f"Could not extract logits from SPIDER output: {type(out)} keys={getattr(out,'keys',lambda:[])()}")
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for r, p in zip(batch_rows, probs):
                all_probs[r] = p
                top1_idx[r] = int(p.argmax())
            batch_imgs.clear()
            batch_rows.clear()
            if (i + 1) % 200 == 0 or i == len(df) - 1:
                rate = (i + 1) / (time.time() - t_start)
                eta = (len(df) - i - 1) / max(rate, 1e-9)
                print(f"  [{i+1:>5,}/{len(df):,}]  {rate:.1f} patch/s  ETA {eta/60:.1f} min", flush=True)

    slide.close()

    # write outputs
    out_df = pd.DataFrame({
        "x": df["x"].values,
        "y": df["y"].values,
        "top1_idx": top1_idx,
        "top1_class": [id2label[i] for i in top1_idx],
        "p_high_grade": all_probs[:, idx_high] if idx_high is not None else np.nan,
        "p_low_grade": all_probs[:, idx_low] if idx_low is not None else np.nan,
    })
    if idx_high is not None:
        # binary prediction: 1 = non-gland (high grade), 0 = gland
        out_df["pred_binary"] = (out_df["top1_class"] == "Adenocarcinoma high grade").astype(int).values
        # softer binary: 1 if p_high > p_low (restricted to the two adenocarcinoma classes)
        out_df["pred_binary_softrenorm"] = (out_df["p_high_grade"] > out_df["p_low_grade"]).astype(int)
    out_path = results_dir / "per_patch_predictions_spider.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[save] {out_path}")

    # full probs
    full_df = pd.DataFrame(all_probs, columns=[f"p_{id2label[i].replace(' ','_')}" for i in range(len(id2label))])
    full_df.insert(0, "y", df["y"].values)
    full_df.insert(0, "x", df["x"].values)
    full_path = results_dir / "per_patch_predictions_spider_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"[save] {full_path}")


if __name__ == "__main__":
    main()
