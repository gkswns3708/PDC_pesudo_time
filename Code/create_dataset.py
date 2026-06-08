"""
Create patch dataset from SVS + Aperio XML annotations.

Usage:
    python create_dataset.py

Pipeline:
    1. Parse XML → positive/negative polygons
    2. Create binary mask (positive - negative) within bounding box
    3. Extract patches from annotated regions (parallelized via multiprocessing)
    4. Filter by tissue content and mask coverage
    5. (Optional) Macenko stain normalization
    6. Save per-slide for leave-one-slide-out CV
"""

import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from lxml import etree
from tqdm import tqdm

from config import Config
from stain_normalizer import MacenkoNormalizer


# ──────────────────────────────────────────────────────────
# Worker state (set by _worker_init)
# ──────────────────────────────────────────────────────────
_W_SLIDE = None
_W_MASK = None
_W_NORMALIZER = None
_W_PARAMS = None  # dict: patch_size, mask_ds, x_off, y_off, save_dir, slide_name, class_name, class_label, extraction_level, tissue_threshold, mask_threshold


def _worker_init(svs_path, mask, target_rgb, params):
    global _W_SLIDE, _W_MASK, _W_NORMALIZER, _W_PARAMS
    _W_SLIDE = openslide.OpenSlide(svs_path)
    _W_MASK = mask
    _W_PARAMS = params
    if target_rgb is not None:
        norm = MacenkoNormalizer()
        norm.fit(target_rgb)
        _W_NORMALIZER = norm
    else:
        _W_NORMALIZER = None


def _process_position(ix_iy):
    """Process one (ix, iy) grid cell. Returns record dict or None."""
    ix, iy = ix_iy
    p = _W_PARAMS
    patch_size = p["patch_size"]
    stride = p["stride"]
    mask_ds = p["mask_ds"]
    x_off = p["x_off"]
    y_off = p["y_off"]
    mask = _W_MASK

    local_x = ix * stride
    local_y = iy * stride

    mx0 = int(local_x / mask_ds)
    my0 = int(local_y / mask_ds)
    mx1 = min(int((local_x + patch_size) / mask_ds), mask.shape[1])
    my1 = min(int((local_y + patch_size) / mask_ds), mask.shape[0])
    if mx0 >= mx1 or my0 >= my1:
        return None

    mask_region = mask[my0:my1, mx0:mx1]
    if mask_region.size == 0:
        return None
    mask_ratio = (mask_region > 0).sum() / mask_region.size
    if mask_ratio < p["mask_threshold"]:
        return None

    abs_x = x_off + local_x
    abs_y = y_off + local_y

    patch = _W_SLIDE.read_region(
        (abs_x, abs_y), p["extraction_level"], (patch_size, patch_size)
    )
    patch_rgb = np.array(patch.convert("RGB"))

    if not is_tissue(patch_rgb, p["tissue_threshold"]):
        return None

    if _W_NORMALIZER is not None:
        try:
            patch_rgb = _W_NORMALIZER.transform(patch_rgb)
        except Exception:
            return None

    # Downsample native patch (e.g. 448) → input_size (e.g. 224) for disk save.
    # INTER_AREA is the correct choice for downsampling (anti-aliased mean).
    input_size = p["input_size"]
    if patch_rgb.shape[0] != input_size:
        patch_rgb = cv2.resize(patch_rgb, (input_size, input_size),
                               interpolation=cv2.INTER_AREA)

    fname = f"{p['slide_name']}_{abs_x}_{abs_y}.png"
    cv2.imwrite(
        str(Path(p["save_dir"]) / fname),
        cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR),
    )
    return {
        "filename": fname,
        "slide": p["slide_name"],
        "class": p["class_name"],
        "label": p["class_label"],
        "x": abs_x,
        "y": abs_y,
    }


# ─────────────────────────────────────────────
# 1. XML Parsing
# ─────────────────────────────────────────────

def parse_aperio_xml(xml_path):
    """Parse Aperio XML annotation file.

    Returns:
        positive_polys: list of Nx2 np.ndarray (outer boundaries)
        negative_polys: list of Nx2 np.ndarray (excluded regions)
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()
    positive_polys = []
    negative_polys = []

    for annotation in root.findall(".//Annotation"):
        for region in annotation.findall(".//Region"):
            vertices = []
            for vertex in region.findall(".//Vertex"):
                x = float(vertex.get("X"))
                y = float(vertex.get("Y"))
                vertices.append([x, y])
            if not vertices:
                continue

            poly = np.array(vertices, dtype=np.float64)
            if region.get("NegativeROA", "0") == "1":
                negative_polys.append(poly)
            else:
                positive_polys.append(poly)

    return positive_polys, negative_polys


# ─────────────────────────────────────────────
# 2. Mask Generation
# ─────────────────────────────────────────────

def get_annotation_bbox(positive_polys, patch_size):
    """Compute bounding box around all positive polygons with padding."""
    all_pts = np.concatenate(positive_polys, axis=0)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    x_min = int(max(0, x_min - patch_size))
    y_min = int(max(0, y_min - patch_size))
    x_max = int(x_max + patch_size)
    y_max = int(y_max + patch_size)

    return x_min, y_min, x_max - x_min, y_max - y_min


def create_binary_mask(positive_polys, negative_polys, bbox, downsample=1.0):
    """Create binary mask within bounding box."""
    x_off, y_off, w, h = bbox
    mask_w = int(w / downsample)
    mask_h = int(h / downsample)
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    for poly in positive_polys:
        pts = ((poly - np.array([x_off, y_off])) / downsample).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    for poly in negative_polys:
        pts = ((poly - np.array([x_off, y_off])) / downsample).astype(np.int32)
        cv2.fillPoly(mask, [pts], 0)

    return mask


# ─────────────────────────────────────────────
# 3. Tissue Detection
# ─────────────────────────────────────────────

def is_tissue(patch_rgb, threshold=0.7):
    """Check if patch contains enough tissue (not background)."""
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    tissue_mask = sat > 20
    tissue_ratio = tissue_mask.sum() / tissue_mask.size
    return tissue_ratio >= threshold


# ─────────────────────────────────────────────
# 4. Patch Extraction
# ─────────────────────────────────────────────

def prepare_slide(svs_path, xml_path, slide_name, class_name, class_label, config):
    """Pre-scan: parse XML, build mask, filter positions by mask_threshold.

    Returns a dict with everything needed for extraction, or None if empty.
    The returned 'candidate_positions' is a list of (ix, iy) that pass the
    mask filter — these are the positions that will actually be read from SVS.
    """
    positive_polys, negative_polys = parse_aperio_xml(xml_path)
    if not positive_polys:
        print(f"  [{slide_name}] No positive polygons. Skipping.")
        return None

    bbox = get_annotation_bbox(positive_polys, config.patch_size)
    x_off, y_off, bbox_w, bbox_h = bbox

    mask_ds = 4.0
    mask = create_binary_mask(positive_polys, negative_polys, bbox, downsample=mask_ds)

    patch_size = config.patch_size
    stride = config.stride
    n_x = (bbox_w - patch_size) // stride + 1
    n_y = (bbox_h - patch_size) // stride + 1

    # Pre-filter by mask_threshold to estimate real workload
    candidate_positions = []
    for iy in range(n_y):
        for ix in range(n_x):
            local_x = ix * stride
            local_y = iy * stride
            mx0 = int(local_x / mask_ds)
            my0 = int(local_y / mask_ds)
            mx1 = min(int((local_x + patch_size) / mask_ds), mask.shape[1])
            my1 = min(int((local_y + patch_size) / mask_ds), mask.shape[0])
            if mx0 >= mx1 or my0 >= my1:
                continue
            region = mask[my0:my1, mx0:mx1]
            if region.size == 0:
                continue
            if (region > 0).sum() / region.size >= config.mask_threshold:
                candidate_positions.append((ix, iy))

    params = {
        "patch_size": patch_size,
        "stride": stride,
        "mask_ds": mask_ds,
        "x_off": x_off,
        "y_off": y_off,
        "save_dir": str(Path(config.output_dir) / slide_name / class_name),
        "slide_name": slide_name,
        "class_name": class_name,
        "class_label": class_label,
        "extraction_level": config.extraction_level,
        "tissue_threshold": config.tissue_threshold,
        "mask_threshold": config.mask_threshold,
        "input_size": config.input_size,
    }

    return {
        "svs_path": svs_path,
        "mask": mask,
        "params": params,
        "candidate_positions": candidate_positions,
        "grid_total": n_x * n_y,
        "bbox": bbox,
    }


def extract_patches_from_slide(prep, config, target_rgb=None, global_bar=None):
    """Run the actual SVS reads + Macenko + save, using a pre-built prep dict.

    If `global_bar` is provided, increment it for each candidate position
    processed (whether kept or dropped by tissue filter). Returns list of
    saved-record dicts.
    """
    svs_path = prep["svs_path"]
    mask = prep["mask"]
    params = prep["params"]
    positions = prep["candidate_positions"]
    slide_name = params["slide_name"]

    Path(params["save_dir"]).mkdir(parents=True, exist_ok=True)

    workers = max(1, int(config.extract_workers))
    records = []

    if workers == 1:
        _worker_init(svs_path, mask, target_rgb, params)
        for pos in positions:
            rec = _process_position(pos)
            if rec is not None:
                records.append(rec)
            if global_bar is not None:
                global_bar.update(1)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(svs_path, mask, target_rgb, params),
        ) as pool:
            chunksize = max(1, len(positions) // (workers * 8))
            for rec in pool.imap_unordered(_process_position, positions, chunksize=chunksize):
                if rec is not None:
                    records.append(rec)
                if global_bar is not None:
                    global_bar.update(1)

    return records


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", nargs="+", default=None,
                        help="extract only these slide stems (default: all in config.slides)")
    args = parser.parse_args()

    config = Config()
    config.ensure_dirs()
    if args.slides:
        wanted = set(args.slides)
        missing = wanted - set(config.slides.keys())
        if missing:
            raise SystemExit(f"--slides not in config.slides: {sorted(missing)}")
        config.slides = {k: v for k, v in config.slides.items() if k in wanted}
        print(f"Filtered to {len(config.slides)} slide(s): {list(config.slides.keys())}")

    target_rgb = None
    if config.stain_normalize:
        target = cv2.imread(config.stain_target_path)
        if target is None:
            raise FileNotFoundError(
                f"stain_target_path not found: {config.stain_target_path}. "
                "Run select_stain_target.py first."
            )
        target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        print(f"Stain normalization ON. Target: {config.stain_target_path}")
    print(f"Output → {config.output_dir}")
    print(f"Extraction workers per slide: {config.extract_workers}")

    # ── Pass 1: pre-scan all slides (XML → mask → candidate count) ──
    print(f"\n{'='*60}")
    print("Pre-scanning slides to compute total workload...")
    print(f"{'='*60}")
    preps = []
    for slide_name, info in config.slides.items():
        svs_path = str(Path(config.svs_dir) / info["svs"])
        xml_path = str(Path(config.xml_dir) / info["xml"])
        prep = prepare_slide(svs_path, xml_path, slide_name,
                             info["class"], info["label"], config)
        if prep is None:
            continue
        n_cand = len(prep["candidate_positions"])
        print(f"  {slide_name:<16} ({info['class']:<10}): "
              f"grid {prep['grid_total']:>7}  →  mask-pass {n_cand:>6}")
        preps.append((slide_name, info, prep))

    total_candidates = sum(len(p["candidate_positions"]) for _, _, p in preps)
    print(f"\nTotal mask-passing positions across {len(preps)} slides: "
          f"{total_candidates:,}")
    print("(Final saved patch count is <= this after tissue-threshold filter.)")

    # ── Pass 2: extract with a single global tqdm bar ──
    all_records = []
    global_bar = tqdm(total=total_candidates, desc="Overall extract",
                      unit="pos", dynamic_ncols=True)

    for slide_name, info, prep in preps:
        global_bar.set_postfix_str(f"slide={slide_name}", refresh=True)
        records = extract_patches_from_slide(
            prep, config, target_rgb=target_rgb, global_bar=global_bar,
        )
        kept = len(records)
        total_cand_slide = len(prep["candidate_positions"])
        global_bar.write(f"  [{slide_name}] saved {kept}/{total_cand_slide} "
                         f"({100*kept/max(total_cand_slide,1):.1f}% pass tissue)")
        all_records.extend(records)

    global_bar.close()

    df = pd.DataFrame(all_records)
    print(f"\n{'='*60}")
    print(f"Total patches extracted: {len(df)}")
    print(f"\nPer slide:")
    print(df.groupby(["slide", "class"]).size().to_string())

    df.to_csv(Path(config.output_dir) / "metadata.csv", index=False)
    print(f"\nMetadata saved to {config.output_dir}/metadata.csv")
    print("Done!")


if __name__ == "__main__":
    main()
