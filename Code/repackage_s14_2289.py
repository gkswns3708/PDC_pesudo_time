"""
Repackage S14-2289-1-6 deliverable in the same per-slide self-contained layout
as external3 (single-slide). Adds: patches (50 high + 50 boundary per source-class),
prediction_overlay.png, manifest, and all 4 XMLs.

Output:
    /app/Gland_Seg/results/S14-2289-1-6.zip
    layout (when unzipped):
        S14-2289-1-6/
        ├── README.txt
        ├── manifest.csv
        ├── prediction_overlay.png
        ├── prediction_hardvote.xml
        ├── prediction_virchow2.xml
        ├── prediction_uni2.xml
        ├── prediction_phikon-v2.xml
        ├── hardvote/
        │   ├── gland/        (100: 50 hi + 50 bd)
        │   ├── non-gland/    (100)
        │   └── wrong/        (empty)
        └── virchow2/
            ├── gland/, non-gland/, wrong/
"""

import shutil
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

from config import Config
from extract_sample_patches import select_patches


SLIDE = "S14-2289-1-6"
PATCH_SOURCES = ("virchow2", "hardvote")        # patches in 2 source folders
ALL_XML_SOURCES = ("virchow2", "uni2", "phikon-v2", "hardvote")  # 4 XMLs included
CLASSES = ("gland", "non-gland")
PRED_COL = {"virchow2": "pred_virchow2", "hardvote": "pred_hardvote"}
N_EACH = 50


def main():
    config = Config()
    results_dir = Path(config.base_dir) / "results" / SLIDE
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    df = pd.read_csv(results_dir / "per_patch_predictions_with_hardvote.csv")
    meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
    ps = meta["patch_size"]

    out_root = Path(config.base_dir) / "results" / "_pkg_s14_2289"
    if out_root.exists():
        shutil.rmtree(out_root)
    slide_dir = out_root / SLIDE
    slide_dir.mkdir(parents=True)

    # Copy XMLs (all 4) and the 6-panel viz
    for src in ALL_XML_SOURCES:
        shutil.copy2(results_dir / f"{SLIDE}_prediction_{src}.xml",
                     slide_dir / f"prediction_{src}.xml")
    shutil.copy2(results_dir / "prediction_overlay.png",
                 slide_dir / "prediction_overlay.png")

    svs_path = Path(config.svs_dir) / f"{SLIDE}.svs"
    so = openslide.OpenSlide(str(svs_path))
    manifest = []
    try:
        for src in PATCH_SOURCES:
            for cls in CLASSES:
                (slide_dir / src / cls).mkdir(parents=True, exist_ok=True)
            (slide_dir / src / "wrong").mkdir(parents=True, exist_ok=True)

            for cls in CLASSES:
                df_cls = df[df[PRED_COL[src]] == cls]
                high, boundary = select_patches(df_cls, src, cls, N_EACH)
                for tag, sub in (("hi", high), ("bd", boundary)):
                    for _, row in sub.iterrows():
                        x, y = int(row["x"]), int(row["y"])
                        fname = f"{SLIDE}_x{x}_y{y}_{tag}.png"
                        img = so.read_region((x, y), 0, (ps, ps))
                        img.convert("RGB").save(
                            slide_dir / src / cls / fname, optimize=True)
                        manifest.append({
                            "slide": SLIDE, "source": src, "class": cls,
                            "tag": "high_conf" if tag == "hi" else "boundary",
                            "filename": fname, "x": x, "y": y,
                            "p_gland_virchow2":  float(row["p_gland_virchow2"]),
                            "p_gland_uni2":      float(row["p_gland_uni2"]),
                            "p_gland_phikon-v2": float(row["p_gland_phikon-v2"]),
                            "p_gland_ensemble":  float(row["p_gland_ensemble"]),
                        })
                print(f"  {src}/{cls}: 50 high + 50 boundary saved")
    finally:
        so.close()

    pd.DataFrame(manifest).to_csv(slide_dir / "manifest.csv", index=False)
    (slide_dir / "README.txt").write_text(
        f"""{SLIDE} — AI 예측 결과 (교수님 XML annotation으로 GT 평가 가능한 슬라이드)

폴더 구조 (self-contained):
    {SLIDE}/
    ├── prediction_hardvote.xml      ← ImageScope에서 SVS와 함께 띄우시는 용도
    ├── prediction_virchow2.xml         (녹색=gland, 빨강=non-gland)
    ├── prediction_uni2.xml
    ├── prediction_phikon-v2.xml
    ├── prediction_overlay.png       ← 빠른 전체 흐름 파악용 6패널 figure
    ├── hardvote/                    ← 작업은 이 폴더에서만!
    │   ├── gland/        (100장: 확신 50 + 경계 50 — 파일명 끝 _hi/_bd로 구분)
    │   ├── non-gland/    (100장)
    │   └── wrong/        (빈 폴더 — 잘못 분류된 패치를 여기로 이동)
    └── virchow2/                    (참고용 — 동일 3 하위폴더)

작업 방식:
1. hardvote/gland/, hardvote/non-gland/ 만 보시면 됩니다 (총 200장).
2. AI가 잘못 분류했다고 판단되는 patch는 hardvote/wrong/ 폴더로 이동.
   (파일명은 그대로 — 좌표·tag·소스 모두 파일명에 박혀 있음)
3. binary classification이므로 wrong에 들어온 파일은 자동으로 라벨이 뒤집힙니다.
4. 작업 끝나시면 {SLIDE} 폴더 통째로 압축해 보내주세요.

파일명 규칙: {SLIDE}_x{{X}}_y{{Y}}_<hi|bd>.png
  hi = high_conf (모델 확신 높음)
  bd = boundary  (결정 경계 p≈0.5)
manifest.csv 에 모든 패치의 모델별 확률·tag 기록돼 있습니다.
""", encoding="utf-8")

    # ── single zip ──
    zip_out = Path(config.base_dir) / "results" / f"{SLIDE}.zip"
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in slide_dir.rglob("*"):
            if p.is_dir() and p.name == "wrong" and not any(p.iterdir()):
                z.writestr(f"{SLIDE}/{p.relative_to(slide_dir)}/", "")
            elif p.is_file():
                z.write(p, f"{SLIDE}/{p.relative_to(slide_dir)}")
    print(f"\nWrote {zip_out} ({zip_out.stat().st_size / 1e9:.2f} GB)")

    # Counts
    print("\nFinal counts:")
    for src in PATCH_SOURCES:
        for cls in CLASSES:
            n = len(list((slide_dir / src / cls).glob("*.png")))
            print(f"  {src}/{cls}: {n}")


if __name__ == "__main__":
    main()
