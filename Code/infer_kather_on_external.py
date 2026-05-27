"""
Run Kather-100K (resnet18) 9-class tissue inference on the ROI region of an
external slide, using the same patch grid as our Virchow2/UNI2/Phikon-v2
results in per_patch_predictions.csv.

Pipeline:
  1. Load per_patch_predictions.csv (our patch positions across whole slide)
  2. Filter patches whose center lies INSIDE the ROI big box(es) from XML
  3. For each ROI patch, read 448×448 region at level 0 (matches Kather 0.5 mpp)
     and let tiatoolbox PatchPredictor handle resize/normalize/inference
  4. Save patch-level 9-class probabilities + 3-group derived columns

Output:
  results/<slide>/kather_per_patch.csv
    columns: x, y,
             p_BACK, p_NORM, p_DEB, p_TUM, p_ADI, p_MUC, p_MUS, p_STR, p_LYM,
             p_Cancer (=TUM+STR), p_Normal (=NORM+ADI+MUS+LYM),
             p_Others (=BACK+DEB+MUC),
             group_argmax (Cancer/Normal/Others),
             in_roi (True), kather_topclass

Usage:
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    python infer_kather_on_external.py S14-2289-1-6
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from lxml import etree
from matplotlib.path import Path as MplPath
from tqdm import tqdm

from config import Config


KATHER_CLASSES = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
NORMAL_IDX = [1, 4, 6, 8]   # NORM, ADI, MUS, LYM
CANCER_IDX = [3, 7]         # TUM, STR
OTHERS_IDX = [0, 2, 5]      # BACK, DEB, MUC


def parse_roi_boxes(xml_path, area_threshold_factor=10.0):
    """Return list of (N,2) ndarrays for the BIG ROI boxes (large area, low vertices).

    Heuristic: area > median * threshold AND vertices < 30 → ROI box.
    """
    tree = etree.parse(xml_path)
    polys = []
    for r in tree.getroot().findall(".//Region"):
        verts = [(float(v.get("X")), float(v.get("Y"))) for v in r.findall(".//Vertex")]
        if len(verts) < 3:
            continue
        polys.append(np.array(verts, dtype=np.float64))
    if not polys:
        return []
    areas = []
    for poly in polys:
        x, y = poly[:, 0], poly[:, 1]
        a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas.append(a)
    areas = np.array(areas)
    med = np.median(areas)
    rois = [p for p, a in zip(polys, areas)
            if (a > med * area_threshold_factor) and (len(p) < 30)]
    return rois


def filter_patches_in_roi(df, roi_polys, patch_size):
    """Keep only patches whose CENTER lies inside any ROI box."""
    if not roi_polys:
        return df.copy()
    cx = df["x"].values + patch_size / 2.0
    cy = df["y"].values + patch_size / 2.0
    pts = np.column_stack([cx, cy])
    in_any = np.zeros(len(df), dtype=bool)
    for poly in roi_polys:
        path = MplPath(poly)
        in_any |= path.contains_points(pts)
    return df[in_any].copy().reset_index(drop=True)


def read_patches(svs_path, df, patch_size_level0, kather_input_px=448):
    """Read kather_input_px x kather_input_px regions centered on our patch centers,
    at level 0. Returns list of (H, W, 3) uint8 RGB arrays."""
    slide = openslide.OpenSlide(svs_path)
    # offset so we center on each patch
    off = (patch_size_level0 - kather_input_px) // 2
    patches = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Read patches"):
        x = int(r["x"]) + off
        y = int(r["y"]) + off
        region = slide.read_region((x, y), 0, (kather_input_px, kather_input_px))
        arr = np.array(region.convert("RGB"))
        patches.append(arr)
    slide.close()
    return patches


def run_kather_inference(patches, batch_size=64):
    """Use tiatoolbox PatchPredictor in patch_mode for our pre-extracted patches.

    Returns (probs (N,9), preds (N,))
    """
    from tiatoolbox.models.engine.patch_predictor import PatchPredictor
    predictor = PatchPredictor(model="resnet18-kather100k",
                                batch_size=batch_size)
    tmpdir = Path(tempfile.mkdtemp(prefix="kather_"))

    # tiatoolbox expects exact (224, 224) input for resnet18-kather100k
    print(f"  Resizing {len(patches)} patches to 224×224 for Kather...")
    patches_224 = [
        cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA) if p.shape[:2] != (224, 224) else p
        for p in patches
    ]

    try:
        # Try patch_mode=True first
        try:
            output = predictor.run(
                images=patches_224,
                patch_mode=True,
                save_dir=tmpdir,
                overwrite=True,
                output_type="zarr",
                return_probabilities=True,
            )
        except Exception as e:
            print(f"  patch_mode=True failed ({e}); falling back to manual inference")
            return _manual_kather_inference(patches_224, batch_size)

        # tiatoolbox output format depends on output_type
        probs = None
        preds = None
        if isinstance(output, dict):
            if "probabilities" in output:
                probs = np.array(output["probabilities"])
            if "predictions" in output:
                preds = np.array(output["predictions"])
        else:
            # output_type="zarr" → output is a Path to .zarr directory
            import zarr
            zarr_path = Path(output)
            print(f"  Loading Kather output from zarr: {zarr_path}")
            store = zarr.open(str(zarr_path), mode="r")
            keys = list(store.keys()) if hasattr(store, "keys") else []
            print(f"  zarr keys: {keys}")
            if "probabilities" in keys:
                probs = np.array(store["probabilities"])
            if "predictions" in keys:
                preds = np.array(store["predictions"])

        if probs is None:
            raise RuntimeError("Kather output did not include probabilities")
        if probs.ndim == 3:
            probs = probs[:, 0, :]
        if preds is None:
            preds = probs.argmax(axis=1)
        return probs, preds
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _manual_kather_inference(patches, batch_size=64):
    """Fallback: load weights directly with timm and infer manually."""
    import torch
    import torch.nn as nn
    import timm

    # tiatoolbox stores resnet18 with custom architecture — try standard first
    state_path = "/root/.tiatoolbox/models/resnet18-kather100k.pth"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = timm.create_model("resnet18", pretrained=False, num_classes=9)
    # adjust possible key prefixes
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."): nk = nk[len("module."):]
        if nk.startswith("model."): nk = nk[len("model."):]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 5:
        raise RuntimeError(f"Unexpected key mismatch: {unexpected[:5]}")
    print(f"  manual load: missing={len(missing)} unexpected={len(unexpected)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    probs_all = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Kather infer"):
            batch = patches[i:i+batch_size]
            # resize to 224
            resized = np.stack([cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA)
                                 for p in batch])
            x = (resized.astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(p)
    probs = np.concatenate(probs_all, axis=0)
    preds = probs.argmax(axis=1)
    return probs, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str)
    parser.add_argument("--svs", type=str, default=None)
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kather-input-px", type=int, default=448,
                        help="Read this px window at level 0; tiatoolbox internally resizes to 224 (≈0.5mpp)")
    args = parser.parse_args()

    config = Config()
    if args.svs:
        svs_path = args.svs
        xml_path = args.xml
    elif args.slide in getattr(config, "external_test_slides", {}):
        info = config.external_test_slides[args.slide]
        svs_path = str(Path(config.svs_dir) / info["svs"])
        xml_path = str(Path(config.xml_dir) / info["xml"])
    else:
        raise ValueError(f"{args.slide} not configured; pass --svs/--xml")

    out_dir = Path(config.base_dir) / "results" / args.slide
    csv_in = out_dir / "per_patch_predictions.csv"
    if not csv_in.exists():
        raise FileNotFoundError(
            f"{csv_in} not found. Run infer_external_slide.py first to get our patch grid.")
    df = pd.read_csv(csv_in)
    print(f"Loaded {len(df)} patches from {csv_in}")

    # ── Filter to ROI box only ──
    rois = parse_roi_boxes(xml_path)
    print(f"Found {len(rois)} ROI big-boxes in {xml_path}")
    df_roi = filter_patches_in_roi(df, rois, config.patch_size)
    print(f"Patches inside ROI: {len(df_roi)} (of {len(df)})")
    if len(df_roi) == 0:
        raise RuntimeError("No patches found inside ROI — check XML / box detection")

    # ── Read patches ──
    patches = read_patches(svs_path, df_roi, config.patch_size,
                            kather_input_px=args.kather_input_px)
    print(f"Read {len(patches)} patches at {args.kather_input_px}×{args.kather_input_px} level 0")

    # ── Run Kather ──
    print(f"\nRunning resnet18-kather100k inference (batch_size={args.batch_size})...")
    probs, preds = run_kather_inference(patches, batch_size=args.batch_size)
    print(f"Done. probs shape={probs.shape}")

    # ── Build output DataFrame ──
    out_df = df_roi[["x", "y"]].copy()
    for i, name in enumerate(KATHER_CLASSES):
        out_df[f"p_{name}"] = probs[:, i]
    out_df["p_Cancer"] = probs[:, CANCER_IDX].sum(axis=1)
    out_df["p_Normal"] = probs[:, NORMAL_IDX].sum(axis=1)
    out_df["p_Others"] = probs[:, OTHERS_IDX].sum(axis=1)

    group_argmax = np.argmax(
        np.stack([out_df["p_Cancer"], out_df["p_Normal"], out_df["p_Others"]], axis=1),
        axis=1,
    )
    group_names = ["Cancer", "Normal", "Others"]
    out_df["group_argmax"] = [group_names[g] for g in group_argmax]
    out_df["kather_topclass"] = [KATHER_CLASSES[p] for p in preds]
    out_df["in_roi"] = True

    csv_out = out_dir / "kather_per_patch.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"\nSaved Kather predictions to {csv_out}")
    print(f"  group distribution:")
    print(out_df["group_argmax"].value_counts().to_string())
    print(f"\n  Cancer mask (p_Cancer >= 0.5):  {(out_df['p_Cancer']>=0.5).sum()} / {len(out_df)} patches")


if __name__ == "__main__":
    main()
