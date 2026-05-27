"""
Inference on an external held-out slide (e.g., S14-2289-1-6) with ensemble of
multiple foundation models.

For each tissue patch on the entire WSI:
    1. Macenko stain normalize (same target as training)
    2. Forward through each loaded checkpoint
    3. Save per-patch (x, y, p_gland_<model>...) + ensemble (mean prob)

Output:
    /app/Gland_Seg/results/<slide>/per_patch_predictions.csv
    /app/Gland_Seg/results/<slide>/prob_map_<model>.npy   (pixel-level, thumb resolution)
    /app/Gland_Seg/results/<slide>/prob_map_ensemble.npy

Usage:
    CUDA_VISIBLE_DEVICES=0 python infer_external_slide.py S14-2289-1-6 \
        --models virchow2 uni2 phikon-v2 \
        --stride 256

If --models is omitted: uses all foundation backbones for which a
`best_model_<bb>_full.pth` checkpoint exists.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
import torch

from config import Config
from model import create_model
from stain_normalizer import MacenkoNormalizer
from visualize_prediction_wsi import (
    _is_tissue, get_thumbnail, parse_aperio_xml,
)
import visualize_prediction_wsi as vw


THUMB_MAX_DIM = 4000


def imagenet_normalize(batch_uint8):
    x = batch_uint8.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.transpose(x, (0, 3, 1, 2))


def _worker_init(svs_path, target_rgb, params):
    vw._worker_init(svs_path, target_rgb, params)


def _process_position(ix_iy):
    return vw._process_position(ix_iy)


def scan_slide_patches(svs_path, target_rgb, config, stride, workers, verbose=True):
    """Scan whole slide → return list of (abs_x, abs_y, patch_uint8(224,224,3))."""
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.level_dimensions[0]
    slide.close()
    patch_size = config.patch_size
    n_x = (w - patch_size) // stride + 1
    n_y = (h - patch_size) // stride + 1
    positions = [(ix, iy) for iy in range(n_y) for ix in range(n_x)]
    if verbose:
        print(f"  Grid {n_x}×{n_y} = {len(positions):,} positions "
              f"(slide {w}x{h}, stride {stride})", flush=True)

    params = {
        "patch_size": patch_size,
        "stride": stride,
        "slide_w": w, "slide_h": h,
        "level": config.extraction_level,
        "tissue_threshold": config.tissue_threshold,
        "input_size": config.input_size,
    }

    ctx = mp.get_context("spawn")
    results = []
    t0 = time.time()
    with ctx.Pool(
        processes=max(1, workers),
        initializer=_worker_init,
        initargs=(svs_path, target_rgb, params),
    ) as pool:
        chunksize = max(1, len(positions) // (max(1, workers) * 16))
        for r in pool.imap_unordered(_process_position, positions, chunksize=chunksize):
            if r is not None:
                results.append(r)
    if verbose:
        print(f"  Tissue patches: {len(results):,}/{len(positions):,} "
              f"({len(results)/max(1,len(positions))*100:.1f}%) in {time.time()-t0:.1f}s",
              flush=True)
    return results, (w, h)


@torch.no_grad()
def run_model_inference(model, patches_uint8, device, batch_size=128, amp_dtype=torch.bfloat16):
    """Returns list of prob_gland values aligned with patches order."""
    probs = []
    use_amp = amp_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
    for i in range(0, len(patches_uint8), batch_size):
        batch = np.stack(patches_uint8[i:i+batch_size])
        x = torch.from_numpy(imagenet_normalize(batch)).to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 0].cpu().numpy()  # prob_gland
        probs.append(p)
    return np.concatenate(probs) if probs else np.array([])


def build_pixel_prob_map(xs, ys, probs, H, W, patch_size, scale):
    score = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for x, y, p in zip(xs, ys, probs):
        tx0 = int(x / scale); ty0 = int(y / scale)
        tx1 = min(int((x + patch_size) / scale), W)
        ty1 = min(int((y + patch_size) / scale), H)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        score[ty0:ty1, tx0:tx1] += p
        count[ty0:ty1, tx0:tx1] += 1
    avg = np.divide(score, count, out=np.zeros_like(score), where=count > 0)
    return avg, count > 0  # value, valid_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slide", type=str, help="slide stem (key into config.external_test_slides) or arbitrary name when --svs-path is given")
    parser.add_argument("--svs-path", type=str, default=None,
                        help="absolute path to SVS — overrides config lookup; allows external slides not in config")
    parser.add_argument("--xml-path", type=str, default=None,
                        help="absolute path to Aperio XML annotation. Optional; if missing, an empty annotation.npz is written")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: <config.base_dir>/results/<slide>)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="backbones to use for ensemble (default: all with _full.pth ckpt)")
    parser.add_argument("--stride", type=int, default=256, help="WSI sliding stride")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=24, help="CPU workers for patch reading")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.svs_path is not None:
        svs_path = str(Path(args.svs_path).resolve())
        xml_path = str(Path(args.xml_path).resolve()) if args.xml_path else None
    elif args.slide in getattr(config, "external_test_slides", {}):
        info = config.external_test_slides[args.slide]
        svs_path = str(Path(config.svs_dir) / info["svs"])
        xml_path = str(Path(config.xml_dir) / info["xml"])
    else:
        raise ValueError(
            f"{args.slide} not in config.external_test_slides — pass --svs-path explicitly")
    print(f"Slide: {svs_path}")
    print(f"Annotation: {xml_path or '(none)'}")

    # Resolve which models to use
    ckpt_dir = Path(config.checkpoint_dir)
    if args.models:
        models = args.models
    else:
        models = []
        for ckpt in sorted(ckpt_dir.glob("best_model_*_full.pth")):
            bb = ckpt.stem.replace("best_model_", "").replace("_full", "")
            models.append(bb)
    print(f"Models (ensemble): {models}")
    if not models:
        raise RuntimeError("No full-train checkpoints found.")

    # Stain target
    target_rgb = None
    if config.stain_normalize:
        t = cv2.imread(config.stain_target_path)
        if t is None:
            raise FileNotFoundError(f"stain target not found: {config.stain_target_path}")
        target_rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)

    # ── Scan slide once, share patches across models ──
    print("\n=== Scanning WSI for tissue patches (parallel) ===")
    results, (slide_w, slide_h) = scan_slide_patches(
        svs_path, target_rgb, config, args.stride, args.workers,
    )
    if not results:
        raise RuntimeError("No tissue patches!")
    xs = np.array([r[0] for r in results])
    ys = np.array([r[1] for r in results])
    patches = [r[2] for r in results]
    print(f"  Patches collected: {len(patches)}")

    # ── Run each model ──
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(config.base_dir) / "results" / args.slide
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_probs = {}
    for bb in models:
        ckpt_path = ckpt_dir / f"best_model_{bb}_full.pth"
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} not found")
            continue
        print(f"\n=== Inference: {bb} ===")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = create_model(num_classes=config.num_classes, pretrained=False,
                             backbone=bb, head_type=getattr(config, "head_type", "linear"))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        t0 = time.time()
        probs = run_model_inference(model, patches, device,
                                     batch_size=args.batch_size)
        print(f"  {bb}: {len(probs)} patches inferred in {time.time()-t0:.1f}s")
        per_model_probs[bb] = probs
        del model
        torch.cuda.empty_cache()

    # ── Ensemble (mean of probs) ──
    ens_probs = np.mean(np.stack(list(per_model_probs.values())), axis=0)
    print(f"\nEnsemble across {len(per_model_probs)} models")
    print(f"  Mean p_gland over slide: {ens_probs.mean():.3f}")
    print(f"  fraction predicted gland (>0.5): {(ens_probs>=0.5).mean():.3f}")

    # ── Save per-patch CSV ──
    df = pd.DataFrame({
        "x": xs, "y": ys,
        **{f"p_gland_{bb}": v for bb, v in per_model_probs.items()},
        "p_gland_ensemble": ens_probs,
        "pred_class_ensemble": np.where(ens_probs >= 0.5, "gland", "non-gland"),
    })
    csv_path = out_dir / "per_patch_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-patch CSV: {csv_path} ({len(df)} rows)")

    # ── Pixel-level prob maps (thumbnail resolution) ──
    slide = openslide.OpenSlide(svs_path)
    thumb_rgb, scale = get_thumbnail(slide, THUMB_MAX_DIM)
    slide.close()
    H, W = thumb_rgb.shape[:2]
    print(f"\nBuilding pixel-level prob maps at thumb {W}x{H} (scale={scale:.2f})")

    np.save(out_dir / "thumbnail.npy", thumb_rgb)
    for bb, probs in per_model_probs.items():
        prob_map, valid = build_pixel_prob_map(xs, ys, probs, H, W,
                                                config.patch_size, scale)
        np.save(out_dir / f"prob_map_{bb}.npy", prob_map)
        np.save(out_dir / f"valid_mask_{bb}.npy", valid)
        print(f"  saved prob_map_{bb}.npy  shape={prob_map.shape}")

    ens_map, ens_valid = build_pixel_prob_map(xs, ys, ens_probs, H, W,
                                              config.patch_size, scale)
    np.save(out_dir / "prob_map_ensemble.npy", ens_map)
    np.save(out_dir / "valid_mask_ensemble.npy", ens_valid)
    print(f"  saved prob_map_ensemble.npy")

    # ── Save annotation polygons too (for downstream viz) ──
    if xml_path and Path(xml_path).exists():
        pos, neg = parse_aperio_xml(xml_path)
    else:
        pos, neg = [], []
    np.savez(out_dir / "annotation.npz",
             positive=np.array([p for p in pos], dtype=object),
             negative=np.array([p for p in neg], dtype=object),
             allow_pickle=True)
    print(f"  saved annotation.npz ({len(pos)} pos, {len(neg)} neg polygons)")

    # ── Slide meta ──
    np.save(out_dir / "slide_meta.npy", np.array({
        "slide_w": slide_w, "slide_h": slide_h,
        "thumb_W": W, "thumb_H": H, "scale": scale,
        "patch_size": config.patch_size, "stride": args.stride,
        "models": list(per_model_probs.keys()),
    }, dtype=object))

    print(f"\n=== Done. Output dir: {out_dir} ===")


if __name__ == "__main__":
    main()
