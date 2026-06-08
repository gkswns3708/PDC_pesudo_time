"""
Step 1 sanity check: run SPIDER colorectal model on SPIDER's OWN known
"Adenocarcinoma high grade" patches (pre-stitched 1120×1120). If pipeline
is correct, top1 should be HG on most samples with high p_high_grade.

Decision rule (from /app/plan/glistening-launching-wave.md):
  PASS if top1=HG ratio >= 0.85 AND mean p_high >= 0.6  ->  pipeline OK, go to Step 2
  FAIL -> debug preprocess (mean/std, stitching order)

Usage:
    HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 \
      /root/miniconda3/envs/tiatoolbox/bin/python sanity_spider_on_own_hg.py \
        --n 500 --batch_size 8
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel

from infer_spider_on_eval import SPIDER_REPO, spider_preprocess

STITCHED_DIR = Path("/app/spider_samples/high_grade_extract/stitched_1120")
OUT_DIR = Path("/app/Gland_Seg/results/_spider_sanity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of samples")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pngs = sorted(STITCHED_DIR.glob("*.png"))
    if not all_pngs:
        sys.exit(f"No PNGs under {STITCHED_DIR}")
    rng = np.random.default_rng(args.seed)
    if len(all_pngs) > args.n:
        idx = rng.choice(len(all_pngs), args.n, replace=False)
        pngs = [all_pngs[i] for i in sorted(idx)]
    else:
        pngs = all_pngs
    print(f"[input] {len(pngs):,} stitched HG patches from {STITCHED_DIR}")

    print(f"[load] {SPIDER_REPO} (device={args.device})")
    t0 = time.time()
    model = AutoModel.from_pretrained(SPIDER_REPO, trust_remote_code=True, token=args.token)
    model.eval().to(args.device)
    class_names = list(model.config.class_names)
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}
    idx_high = label2id.get("Adenocarcinoma high grade")
    idx_low = label2id.get("Adenocarcinoma low grade")
    print(f"[load] done in {time.time()-t0:.1f}s. {len(id2label)} classes; "
          f"idx_high={idx_high} idx_low={idx_low}")

    all_probs = np.zeros((len(pngs), len(id2label)), dtype=np.float32)
    top1 = np.zeros(len(pngs), dtype=np.int32)

    batch_imgs, batch_idx = [], []
    t_start = time.time()
    for i, p in enumerate(pngs):
        pil = Image.open(p).convert("RGB")
        if pil.size != (1120, 1120):
            pil = pil.resize((1120, 1120), Image.BILINEAR)
        batch_imgs.append(pil)
        batch_idx.append(i)
        if len(batch_imgs) >= args.batch_size or i == len(pngs) - 1:
            x = spider_preprocess(batch_imgs).to(args.device)
            with torch.inference_mode():
                out = model(pixel_values=x)
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, dict):
                logits = out.get("logits")
            if logits is None and isinstance(out, (tuple, list)):
                logits = out[0]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for r, pr in zip(batch_idx, probs):
                all_probs[r] = pr
                top1[r] = int(pr.argmax())
            batch_imgs.clear(); batch_idx.clear()
            if (i + 1) % 50 == 0 or i == len(pngs) - 1:
                rate = (i + 1) / (time.time() - t_start)
                eta = (len(pngs) - i - 1) / max(rate, 1e-9)
                print(f"  [{i+1:>4}/{len(pngs)}]  {rate:.1f}/s  ETA {eta:.0f}s", flush=True)

    # build dataframe
    df = pd.DataFrame({
        "file": [p.name for p in pngs],
        "top1_idx": top1,
        "top1_class": [id2label[i] for i in top1],
    })
    if idx_high is not None:
        df["p_high_grade"] = all_probs[:, idx_high]
    if idx_low is not None:
        df["p_low_grade"] = all_probs[:, idx_low]
    out_csv = OUT_DIR / "sanity_spider_on_own_hg.csv"
    df.to_csv(out_csv, index=False)

    # full probs
    full = pd.DataFrame(all_probs, columns=[f"p_{id2label[i].replace(' ', '_')}" for i in range(len(id2label))])
    full.insert(0, "file", [p.name for p in pngs])
    full_csv = OUT_DIR / "sanity_spider_on_own_hg_full.csv"
    full.to_csv(full_csv, index=False)

    # report
    print()
    print("=" * 60)
    print("SANITY CHECK REPORT")
    print("=" * 60)
    print(f"input dir            : {STITCHED_DIR}")
    print(f"n samples            : {len(pngs)}")
    print(f"top1 class histogram :")
    for cls, n in df["top1_class"].value_counts().items():
        print(f"  {n:>4}  ({n/len(df)*100:5.1f}%)  {cls}")
    if idx_high is not None:
        hg_ratio = (df["top1_idx"] == idx_high).mean()
        p_high_mean = df["p_high_grade"].mean()
        p_high_med = df["p_high_grade"].median()
        print()
        print(f"top1 == Adeno HG     : {hg_ratio*100:.1f}%  (target ≥ 85.0%)")
        print(f"mean p_high_grade    : {p_high_mean:.3f}    (target ≥ 0.60)")
        print(f"median p_high_grade  : {p_high_med:.3f}")
        verdict = "PASS" if (hg_ratio >= 0.85 and p_high_mean >= 0.6) else "FAIL"
        print(f"VERDICT              : {verdict}")
    print(f"\n[save] {out_csv}")
    print(f"[save] {full_csv}")


if __name__ == "__main__":
    main()
