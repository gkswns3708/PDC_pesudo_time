"""
Re-render METU-CCTGS overlays at full original resolution (no matplotlib
downsampling). For each annotated image produce 2 files:
  - <name>__raw.png      : raw H&E at original size
  - <name>__overlay.png  : RGB overlay (raw 55% + colorized mask 45%) at original size

Slides are grouped into per-Grade folders by class-presence:
  Grade-1/  : slides whose annotation mask contains class id 1
  Grade-2/  : slides whose annotation mask contains class id 2
  Grade-3/  : slides whose annotation mask contains class id 3 (PDC, our target)

A slide containing multiple grades is copied into each matching grade folder
(symlink to avoid disk bloat — overlay files are written once into a shared
"_full" dir and symlinked from per-grade folders).

Output root: /app/overlays_fullres/
  _full/<split>/<name>__raw.png         (one copy)
  _full/<split>/<name>__overlay.png     (one copy)
  Grade-1/<split>/<name>__overlay.png   (symlink to _full)
  Grade-1/<split>/<name>__raw.png       (symlink to _full)
  Grade-2/...
  Grade-3/...
  legend.png                            (color legend reference)
  class_distribution.csv                (split, file, has_<class>)
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/app/metucctgs_ds")
OUT = Path("/app/overlays_fullres")

CLASS_RGB = {
    0: ((0, 0, 0), "Others"),
    1: ((0, 192, 0), "Tumor Grade-1 (well diff.)"),
    2: ((255, 224, 32), "Tumor Grade-2 (mod. diff.)"),
    3: ((255, 0, 0), "Tumor Grade-3 (poorly diff. / PDC)"),
    4: ((0, 32, 255), "Normal Mucosa"),
}


def colorize(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, (rgb, _) in CLASS_RGB.items():
        out[mask == cid] = rgb
    return out


def render_one(img_path, ann_path, raw_out, overlay_out):
    img = np.asarray(Image.open(img_path).convert("RGB"))
    mask = np.asarray(Image.open(ann_path))
    color = colorize(mask)
    blend = (0.55 * img.astype(np.float32) + 0.45 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(raw_out, optimize=True)
    Image.fromarray(blend).save(overlay_out, optimize=True)
    return sorted(int(c) for c in np.unique(mask))


def write_legend(path):
    rows = list(CLASS_RGB.items())
    sw, sh = 60, 60
    pad = 12
    text_w = 520
    H = (sh + pad) * len(rows) + pad
    W = pad + sw + pad + text_w + pad
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    y = pad
    for cid, (rgb, name) in rows:
        d.rectangle([pad, y, pad + sw, y + sh], fill=rgb, outline="black", width=2)
        d.text((pad + sw + pad, y + sh // 4), f"[{cid}] {name}", fill="black", font=font)
        y += sh + pad
    img.save(path, optimize=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    full_root = OUT / "_full"
    full_root.mkdir(parents=True)

    # per-grade output dirs
    grade_dirs = {g: OUT / f"Grade-{g}" for g in (1, 2, 3)}
    for d in grade_dirs.values():
        d.mkdir(parents=True)

    rows = []
    for split in ("train", "validation"):
        img_dir = ROOT / "images" / split
        ann_dir = ROOT / "annotations" / split
        full_split = full_root / split
        full_split.mkdir(parents=True, exist_ok=True)
        for g in (1, 2, 3):
            (grade_dirs[g] / split).mkdir(parents=True, exist_ok=True)
        files = sorted(os.listdir(ann_dir))
        print(f"[{split}] {len(files)} images")
        for f in files:
            ip = img_dir / f
            ap = ann_dir / f
            stem = Path(f).stem
            raw_out = full_split / f"{stem}__raw.png"
            ovl_out = full_split / f"{stem}__overlay.png"
            classes = render_one(ip, ap, raw_out, ovl_out)

            row = {"split": split, "file": f, "classes_present": ";".join(map(str, classes))}
            row.update({f"has_{cid}_{CLASS_RGB[cid][1].split(' (')[0].replace(' ','_')}": int(cid in classes)
                        for cid in CLASS_RGB})
            rows.append(row)

            # symlink into per-grade folders
            for g in (1, 2, 3):
                if g in classes:
                    for src in (raw_out, ovl_out):
                        dst = grade_dirs[g] / split / src.name
                        if dst.exists() or dst.is_symlink():
                            dst.unlink()
                        # relative symlink so the archive is portable
                        rel = os.path.relpath(src, dst.parent)
                        os.symlink(rel, dst)
            print(f"  {split}/{f}  classes={classes}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "class_distribution.csv", index=False)
    write_legend(OUT / "legend.png")

    # summary counts
    n_per_grade = {g: int(df[f"has_{g}_Tumor_Grade-{g}"].sum()) for g in (1, 2, 3)}
    with open(OUT / "README.txt", "w") as fp:
        fp.write("METU-CCTGS full-resolution overlays\n")
        fp.write("====================================\n\n")
        fp.write(f"Total slides:        {len(df)}\n")
        fp.write(f"Grade-1 (well):      {n_per_grade[1]} slides\n")
        fp.write(f"Grade-2 (mod):       {n_per_grade[2]} slides\n")
        fp.write(f"Grade-3 (PDC):       {n_per_grade[3]} slides\n\n")
        fp.write("Layout:\n")
        fp.write("  _full/<split>/<name>__raw.png       (raw H&E, original resolution)\n")
        fp.write("  _full/<split>/<name>__overlay.png   (raw + colorized mask 45% blend)\n")
        fp.write("  Grade-{1,2,3}/<split>/...           (symlinks to _full, grouped by class presence)\n")
        fp.write("  legend.png                          (color reference)\n")
        fp.write("  class_distribution.csv              (per-slide class presence flags)\n\n")
        fp.write("Class colors (in overlays):\n")
        for cid, (rgb, name) in CLASS_RGB.items():
            fp.write(f"  [{cid}] {name}  RGB={rgb}\n")
    print(f"\n[done] {n_per_grade}  -> {OUT}")


if __name__ == "__main__":
    main()
