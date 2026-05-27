"""
Repackage external3 deliverable into a slide-centric self-contained layout:

    external3/
    ├── README.txt
    ├── selected_slides.txt
    ├── manifest.csv               # slide,source,class,tag,filename,x,y,p_gland_*
    ├── <slide>/
    │   ├── prediction_hardvote.xml
    │   ├── prediction_virchow2.xml
    │   ├── hardvote/
    │   │   ├── gland/         (50 high + 50 boundary = 100 patches)
    │   │   ├── non-gland/     (100)
    │   │   └── wrong/         (empty — drag wrong patches here)
    │   └── virchow2/
    │       └── (same 3 subfolders)
    └── ... (3 slides)

Filenames embed both coords AND confidence tag for full traceability:
    <slide>_x{X}_y{Y}_<hi|bd>.png

Output: single external3.zip (STORED) at out_dir parent.

Run from /app/Gland_Seg/Code:
    python repackage_external3.py
"""

import shutil
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

from config import Config
from extract_sample_patches import select_patches


SLIDES = ("S14-10234-2-3", "S14-1069-1-6", "S14-1253-1-3")
SOURCES = ("virchow2", "hardvote")
CLASSES = ("gland", "non-gland")
PRED_COL = {"virchow2": "pred_virchow2", "hardvote": "pred_hardvote"}
N_EACH = 50  # 50 high + 50 boundary = 100 per (slide, source, class)


def main():
    config = Config()
    out_dir = Path(config.base_dir) / "results" / "external3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe per-slide subfolders + old zips, keep selected_slides.txt + xmls/ + prediction/ for now
    for s in SLIDES:
        slide_dir = out_dir / s
        if slide_dir.exists():
            shutil.rmtree(slide_dir)

    manifest_rows = []
    for slide in SLIDES:
        results_dir = Path(config.base_dir) / "results" / slide
        if not results_dir.exists():
            raise FileNotFoundError(f"missing inference output for {slide}")
        df = pd.read_csv(results_dir / "per_patch_predictions_with_hardvote.csv")
        meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
        ps = meta["patch_size"]

        slide_dir = out_dir / slide
        slide_dir.mkdir(parents=True)

        # Copy XMLs (renamed for cleaner display)
        for src in SOURCES:
            xml_in = results_dir / f"{slide}_prediction_{src}.xml"
            xml_out = slide_dir / f"prediction_{src}.xml"
            shutil.copy2(xml_in, xml_out)

        svs_path = Path(config.svs_dir) / f"{slide}.svs"
        so = openslide.OpenSlide(str(svs_path))
        try:
            for src in SOURCES:
                # Make 3 leaf folders per source (gland, non-gland, wrong)
                for cls in CLASSES:
                    (slide_dir / src / cls).mkdir(parents=True, exist_ok=True)
                (slide_dir / src / "wrong").mkdir(parents=True, exist_ok=True)

                for cls in CLASSES:
                    df_cls = df[df[PRED_COL[src]] == cls]
                    if len(df_cls) == 0:
                        print(f"  [{slide}] {src}/{cls}: 0 patches")
                        continue
                    high, boundary = select_patches(df_cls, src, cls, N_EACH)
                    for tag, sub in (("hi", high), ("bd", boundary)):
                        for _, row in sub.iterrows():
                            x, y = int(row["x"]), int(row["y"])
                            fname = f"{slide}_x{x}_y{y}_{tag}.png"
                            img = so.read_region((x, y), 0, (ps, ps))
                            img.convert("RGB").save(
                                slide_dir / src / cls / fname, optimize=True)
                            manifest_rows.append({
                                "slide": slide, "source": src, "class": cls,
                                "tag": "high_conf" if tag == "hi" else "boundary",
                                "filename": fname, "x": x, "y": y,
                                "p_gland_virchow2":  float(row["p_gland_virchow2"]),
                                "p_gland_uni2":      float(row["p_gland_uni2"]),
                                "p_gland_phikon-v2": float(row["p_gland_phikon-v2"]),
                                "p_gland_ensemble":  float(row["p_gland_ensemble"]),
                            })
                    print(f"  [{slide}] {src}/{cls}: 50 high + 50 boundary saved")
        finally:
            so.close()

    # Write manifest
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nWrote manifest.csv ({len(manifest_rows)} rows)")

    # README
    (out_dir / "README.txt").write_text(
        """external3 — AI 예측 결과 (annotation 없는 외부 슬라이드 3장)

대상 슬라이드: S14-10234-2-3, S14-1069-1-6, S14-1253-1-3

폴더 구조 (슬라이드별 self-contained):
    <slide>/
    ├── prediction_hardvote.xml   ← ImageScope에서 SVS와 함께 띄워 보시는 용도
    ├── prediction_virchow2.xml      (녹색=gland, 빨강=non-gland)
    ├── hardvote/                    ← 교수님 작업은 이 폴더에서만!
    │   ├── gland/        (100장: 확신 50 + 경계 50 — 파일명 끝의 _hi/_bd로 구분)
    │   ├── non-gland/    (100장)
    │   └── wrong/        (빈 폴더 — 잘못 분류된 패치를 여기로 이동)
    └── virchow2/                    (참고용 — 동일 3 하위폴더)
        ├── gland/
        ├── non-gland/
        └── wrong/

작업 방식:
1. <slide>/hardvote/gland/, <slide>/hardvote/non-gland/ 만 보시면 됩니다.
2. AI가 잘못 분류했다고 판단되는 패치만 같은 슬라이드의 hardvote/wrong/ 폴더로 이동.
   (파일명 변경 없이 — 좌표·tag·소스 모두 파일명에 박혀 있음)
3. binary classification이므로 wrong에 들어온 파일은 자동으로 라벨 뒤집어 처리.
4. 작업 끝나시면 external3 폴더 통째로 압축해 보내주세요.

파일명 규칙: <slide>_x{X}_y{Y}_<hi|bd>.png
  hi = high_conf (모델 확신 높음)
  bd = boundary  (결정 경계 p≈0.5)
manifest.csv 에 모든 패치의 모델별 확률·tag 기록돼 있습니다.
""", encoding="utf-8")
    print("Wrote README.txt")

    # selected_slides.txt
    (out_dir / "selected_slides.txt").write_text("\n".join(SLIDES) + "\n")

    # ── Build single zip (slide folders + meta files; STORED) ──
    # Old artifacts (xmls/, prediction/, *.zip) excluded
    zip_out = out_dir.parent / "external3.zip"
    if zip_out.exists():
        zip_out.unlink()
    print(f"\nWriting {zip_out} (STORED) ...")
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        # Top-level files
        for top in ("README.txt", "selected_slides.txt", "manifest.csv"):
            p = out_dir / top
            z.write(p, f"external3/{top}")
        # Per-slide folders (recursive, including empty wrong/)
        for slide in SLIDES:
            base = out_dir / slide
            for p in base.rglob("*"):
                if p.is_dir() and p.name == "wrong" and not any(p.iterdir()):
                    arc = f"external3/{p.relative_to(out_dir)}/"
                    z.writestr(arc, "")
                elif p.is_file():
                    arc = f"external3/{p.relative_to(out_dir)}"
                    z.write(p, arc)
    print(f"  size = {zip_out.stat().st_size / 1e9:.2f} GB")
    print(f"\nFinal location: {zip_out}")

    # Counts
    print("\nFinal patch counts:")
    for slide in SLIDES:
        for src in SOURCES:
            for cls in CLASSES:
                n = len(list((out_dir / slide / src / cls).glob("*.png")))
                print(f"  {slide}/{src}/{cls}: {n}")


if __name__ == "__main__":
    main()
