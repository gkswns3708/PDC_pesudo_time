"""
Bundle the prediction artifacts for a held-out slide into a delivery folder.

Structure (slim — 3 categories only):
    /app/Gland_Seg/results/<slide>/delivery/
    ├── README.md                         # 슬라이드/모델/F1 요약 + 작업 안내
    ├── annotation/                       # 4 Aperio XML files (3 모델 + hardvote)
    └── visualization/                    # prediction_overlay.png (6-panel)

Patches are delivered separately as `sample_patches.zip` (4 model folders).

Usage:
    python build_delivery_package.py S14-2289-1-6
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


SOURCES = ("virchow2", "uni2", "phikon-v2", "hardvote")


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied  {src.name}  →  {dst.relative_to(dst.parent.parent.parent)}")


def load_train_val_metrics(checkpoint_dir):
    """Read val_f1 / val_acc / epoch from each `_full.pth` checkpoint."""
    import torch
    rows = []
    for bb in ("virchow2", "uni2", "phikon-v2"):
        ck_path = Path(checkpoint_dir) / f"best_model_{bb}_full.pth"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        rows.append({
            "model": bb,
            "val_f1": float(ck.get("val_f1", float("nan"))),
            "val_acc": float(ck.get("val_acc", float("nan"))),
            "epoch": int(ck.get("epoch", -1)),
        })
    return rows


def build_readme(slide, results_dir, dest):
    meta = np.load(results_dir / "slide_meta.npy", allow_pickle=True).item()
    summary = pd.read_csv(results_dir / "prediction_summary.csv")
    summary_md = (results_dir / "prediction_summary.md").read_text(encoding="utf-8")

    lines = [
        f"# {slide} — Prediction delivery package",
        "",
        "## 슬라이드 정보",
        "",
        f"- 슬라이드명: **{slide}** (외부 held-out, 학습에는 사용하지 않음)",
        f"- Level-0 dimensions: {meta['slide_w']:,} × {meta['slide_h']:,} px",
        f"- Thumbnail: {meta['thumb_W']:,} × {meta['thumb_H']:,} px (scale={meta['scale']:.2f}×)",
        f"- 패치 크기: {meta['patch_size']} px @ stride {meta['stride']} px",
        f"- 분류된 tissue 패치 수: {int(summary.iloc[0]['n_patches']):,}",
        "",
        "## 학습 설정",
        "",
        "- 학습 데이터: 8장 슬라이드(S14-1255-1-3, 1382-4, 1639-1-7, 1720-6, 177-1-5, 2162-1-5, 248-1-3, 252-3) 전수 학습 (LOSO 아님)",
        "- 모델 3개 (모두 `_full.pth` 체크포인트):",
        "    - **virchow2** (Paige, ViT-H/14, 3.1M WSI 사전학습)",
        "    - **uni2** (MahmoodLab UNI2-h, ViT-H/14)",
        "    - **phikon-v2** (Owkin, ViT-L/16)",
        "- Stain normalization: Macenko (target = `Data/stain_target.png`)",
        "- 분류 결정 임계값: P(gland) ≥ 0.5",
        "",
        "## 학습 시 validation 성능 (10% patch holdout, macro-F1)",
        "",
        "학습 슬라이드 8장에서 무작위 추출한 10% 패치 holdout 기준입니다 — 외부 슬라이드 평가는 별도 (아래 GT 섹션 참고).",
        "",
        "| Model | val F1 (macro) | val acc | best epoch |",
        "|---|---:|---:|---:|",
    ]
    for r in load_train_val_metrics(Config().checkpoint_dir):
        lines.append(
            f"| {r['model']} | {r['val_f1']:.4f} | {r['val_acc']:.4f} | {r['epoch']} |"
        )
    lines += [
        "",
        "## 전달 파일 구성 (3 가지)",
        "",
        "**① delivery/annotation/ — Aperio XML 4종**",
        "",
        "```",
        "annotation/",
        "├── S14-2289-1-6_prediction_virchow2.xml",
        "├── S14-2289-1-6_prediction_uni2.xml",
        "├── S14-2289-1-6_prediction_phikon-v2.xml",
        "└── S14-2289-1-6_prediction_hardvote.xml   # 3-모델 다수결 앙상블",
        "```",
        "",
        "ImageScope에서 원본 SVS와 함께 띄워 모델 예측을 polygon으로 보실 수 있습니다. **hardvote XML이 가장 우선**입니다.",
        "",
        "**② delivery/visualization/prediction_overlay.png**",
        "",
        "썸네일+annotation, 모델 3개 heatmap, mean-prob 앙상블, hardvote 앙상블의 6패널 figure. 빠른 전체 흐름 파악용.",
        "",
        "**③ sample_patches/ (별도 zip, 약 3.3 GB)**",
        "",
        "```",
        "sample_patches/",
        "├── hardvote/                            # 교수님 작업은 이 폴더에서만 진행하시면 됩니다",
        "│   ├── gland_high_conf/      (100장)    # AI: gland (확신, |p-0.5| 가장 큼)",
        "│   ├── gland_boundary/       (100장)    # AI: gland (경계, p≈0.5)",
        "│   ├── non-gland_high_conf/  (100장)",
        "│   └── non-gland_boundary/   (100장)",
        "├── virchow2/    (동일 4개 하위폴더 — 참고용)",
        "├── uni2/        (동일 4개 하위폴더 — 참고용)",
        "└── phikon-v2/   (동일 4개 하위폴더 — 참고용)",
        "```",
        "",
        "파일명 형식: `rank###_x{X}_y{Y}.png` — 좌표가 박혀 있어 어느 폴더에 있어도 WSI 위 위치를 자동 복원 가능. 각 하위폴더에 `manifest.csv`(좌표 + 모델별 확률).",
        "",
        "## 교수님 작업 방식",
        "",
        "1. `sample_patches/hardvote/` 안의 4개 하위폴더만 보시면 됩니다 (총 400장).",
        "2. 각 폴더 안에 **`wrong`** 라는 빈 폴더를 새로 만들어 주세요.",
        "3. AI가 잘못 분류했다고 판단되는 패치만 `wrong/` 폴더로 *이동*해 주시면 됩니다 (이름 변경 없이).",
        "4. binary classification 이라 `wrong/`에 들어온 파일은 라벨을 뒤집기만 하면 됩니다 — 별도 라벨 입력 불필요.",
        "5. 작업 끝나시면 sample_patches 폴더 통째로 압축해 보내주세요.",
        "",
        "## XML 색상 규약 (ImageScope)",
        "",
        "각 XML 파일에 두 개의 `<Annotation>` 그룹이 들어 있습니다:",
        "",
        "| Annotation Id | LineColor | 의미 |",
        "|---|---|---|",
        "| 1 | 녹색 (LineColor=65280) | predicted **gland** |",
        "| 2 | 빨강 (LineColor=255) | predicted **non-gland** |",
        "",
        "Polygon은 thumbnail 마스크에서 OpenCV 외곽선 추출 → 5,000 px(level-0) 미만 영역 제거 → Douglas-Peucker(ε=8 px)로 단순화한 결과입니다.",
        "",
        "## 예측 요약",
        "",
        summary_md.split("# Prediction summary —")[-1].split("\n", 1)[1].strip(),
        "",
        "## 추론 파이프라인 재현",
        "",
        "```bash",
        "cd /app/Gland_Seg/Code",
        "# 1) 추론 (3 모델) — ~5분 (L40 1장, stride=512)",
        f"python infer_external_slide.py {slide} --models virchow2 uni2 phikon-v2 --stride 512",
        "# 2) 패치 표·시각화·하드보팅",
        f"python summarize_external_predictions.py {slide}",
        "# 3) Aperio XML 4개",
        f"python prediction_to_xml.py {slide} --all",
        "# 4) 본 패키지 빌드",
        f"python build_delivery_package.py {slide}",
        "```",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote   README.md")


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_delivery_package.py <slide_name>")
        sys.exit(1)
    slide = sys.argv[1]
    config = Config()
    results_dir = Path(config.base_dir) / "results" / slide
    if not results_dir.exists():
        sys.exit(f"results dir not found: {results_dir}")

    delivery = results_dir / "delivery"
    if delivery.exists():
        shutil.rmtree(delivery)
    (delivery / "annotation").mkdir(parents=True)
    (delivery / "visualization").mkdir(parents=True)

    print(f"Building delivery package → {delivery}")

    # ── annotation/ ──
    for src in SOURCES:
        xml = results_dir / f"{slide}_prediction_{src}.xml"
        if not xml.exists():
            sys.exit(f"missing XML: {xml} — run prediction_to_xml.py --all first")
        copy(xml, delivery / "annotation" / xml.name)

    # ── visualization/ ── (only the 6-panel prediction overlay)
    pred_overlay = results_dir / "prediction_overlay.png"
    if not pred_overlay.exists():
        sys.exit(f"missing: {pred_overlay} — run summarize_external_predictions.py first")
    copy(pred_overlay, delivery / "visualization" / pred_overlay.name)

    # ── README ──
    build_readme(slide, results_dir, delivery / "README.md")

    # ── Final tree ──
    print(f"\nDelivery package ready: {delivery}")
    for p in sorted(delivery.rglob("*")):
        if p.is_file():
            rel = p.relative_to(delivery)
            size = p.stat().st_size
            print(f"  {size:>12,} bytes  {rel}")


if __name__ == "__main__":
    main()
