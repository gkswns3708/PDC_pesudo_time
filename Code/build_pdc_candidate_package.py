"""
Build a candidate package for professor review:

  1) Top-K patches from user's slides where SPIDER's p_high_grade is highest
     (relative rank, even though absolute values are low) — each rendered as a
     1120×1120 SPIDER-input view AND a thumbnail with a red box showing WSI
     location.
  2) Reference: 20 random SPIDER-confirmed Adenocarcinoma-High-Grade patches
     (1120×1120 stitched composites from the SPIDER training set).

Output: /app/Gland_Seg/results/_spider_pdc_candidates/
  README.md, candidates_top30.csv,
  candidates/{rank:03d}__{slide}__x{x}_y{y}__phigh{p}.png      (1120x1120 patch)
  candidates/{rank:03d}__{slide}__x{x}_y{y}__phigh{p}_loc.png  (thumbnail+marker)
  spider_reference_hg/ref_{i:03d}__{src}.png                   (1120x1120)
  contact_sheet_candidates.png, contact_sheet_reference.png
"""
import shutil
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import Config
from infer_spider_on_eval import patch_to_spider_input

TOP_K = 30
N_REF = 20

OUT_DIR = Path('/app/Gland_Seg/results/_spider_pdc_candidates')
CAND_DIR = OUT_DIR / 'candidates'
REF_DIR = OUT_DIR / 'spider_reference_hg'
SPIDER_HG_SRC = Path('/app/spider_samples/high_grade_extract/stitched_1120')


def aggregate_topk():
    rows = []
    for sd in Path('/app/Gland_Seg/results').glob('S14-*'):
        csv = sd / 'per_patch_predictions_spider_scale.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df['slide'] = sd.name
        rows.append(df)
    big = pd.concat(rows, ignore_index=True)
    top = big.nlargest(TOP_K, 'p_high_grade').reset_index(drop=True)
    top.insert(0, 'rank', np.arange(1, len(top) + 1))
    return top


def render_thumbnail_marker(slide, cx, cy, fov=2240, thumb_max=1200, marker_color=(255, 0, 0)):
    """Return PIL thumbnail with a red box around the patch FOV."""
    W0, H0 = slide.level_dimensions[0]
    scale = max(W0, H0) / thumb_max
    tlevel = 0
    # pick a coarse level for fast thumbnail
    for i, d in enumerate(slide.level_downsamples):
        if d <= scale * 1.1:
            tlevel = i
    td = slide.level_downsamples[tlevel]
    tw = int(round(W0 / td))
    th = int(round(H0 / td))
    thumb = slide.read_region((0, 0), tlevel, (tw, th)).convert('RGB')
    # rescale to thumb_max
    final_scale = max(tw, th) / thumb_max
    if final_scale > 1.0:
        new_w = int(round(tw / final_scale))
        new_h = int(round(th / final_scale))
        thumb = thumb.resize((new_w, new_h), Image.BILINEAR)
    actual_scale = max(W0, H0) / max(thumb.size)

    half = fov // 2
    x0 = (cx - half) / actual_scale
    y0 = (cy - half) / actual_scale
    x1 = (cx + half) / actual_scale
    y1 = (cy + half) / actual_scale
    draw = ImageDraw.Draw(thumb)
    # thick rectangle for visibility
    for w in range(4):
        draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w], outline=marker_color, width=1)
    return thumb


def make_contact_sheet(image_paths, labels, out_path, cols=5, tile=300):
    rows = (len(image_paths) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile, rows * (tile + 28)), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except Exception:
        font = ImageFont.load_default()
    for i, (p, lab) in enumerate(zip(image_paths, labels)):
        r, c = divmod(i, cols)
        img = Image.open(p).convert('RGB').resize((tile, tile), Image.BILINEAR)
        sheet.paste(img, (c * tile, r * (tile + 28)))
        draw.text((c * tile + 4, r * (tile + 28) + tile + 4), lab, fill='black', font=font)
    sheet.save(out_path)


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    CAND_DIR.mkdir(parents=True)
    REF_DIR.mkdir(parents=True)

    cfg = Config()

    # 1) Candidates from user's slides
    top = aggregate_topk()
    print(f"[top-{TOP_K}] across {top['slide'].nunique()} slides; "
          f"p_high range [{top['p_high_grade'].min():.4f}, {top['p_high_grade'].max():.4f}]")

    slide_cache = {}
    cand_metadata = []
    cand_thumbs_for_sheet = []
    cand_labels = []
    for _, row in top.iterrows():
        rank = int(row['rank'])
        slide_id = row['slide']
        x = int(row['x'])
        y = int(row['y'])
        p_high = float(row['p_high_grade'])
        p_low = float(row['p_low_grade'])
        top1 = row['top1_class']
        gt_label = int(row.get('gt_label_thr050', -9)) if not pd.isna(row.get('gt_label_thr050', np.nan)) else -9

        if slide_id not in slide_cache:
            svs_path = Path(cfg.svs_dir) / f"{slide_id}.svs"
            if not svs_path.exists():
                ext = cfg.external_test_slides.get(slide_id, {})
                if ext.get('svs'):
                    svs_path = Path(cfg.svs_dir) / ext['svs']
            slide_cache[slide_id] = openslide.OpenSlide(str(svs_path))
        slide = slide_cache[slide_id]

        # FOV: 2240×2240 @ L0 centered on (x + 1120, y + 1120)
        # (the grid CSV stores top-left of the 2240 FOV)
        cx = x + 1120
        cy = y + 1120
        patch = patch_to_spider_input(slide, cx, cy, level0_size=2240, out_size=1120)
        thumb = render_thumbnail_marker(slide, cx, cy, fov=2240, thumb_max=1200)

        gt_str = {1: 'GT=non-gland', 0: 'GT=gland', -1: 'GT=skip', -9: 'GT=n/a'}.get(gt_label, str(gt_label))
        base = f"{rank:03d}__{slide_id}__x{x}_y{y}__phigh{p_high:.4f}"
        patch_p = CAND_DIR / f"{base}.png"
        thumb_p = CAND_DIR / f"{base}_loc.png"
        patch.save(patch_p)
        thumb.save(thumb_p)

        cand_metadata.append({
            'rank': rank, 'slide': slide_id, 'x_topleft_L0': x, 'y_topleft_L0': y,
            'cx_L0': cx, 'cy_L0': cy, 'fov_L0_px': 2240, 'patch_um': '~564',
            'top1_class': top1, 'p_high_grade': p_high, 'p_low_grade': p_low,
            'gt_label_thr050': gt_label, 'gt_label_meaning': gt_str,
            'patch_file': patch_p.name, 'location_file': thumb_p.name,
        })
        cand_thumbs_for_sheet.append(patch_p)
        cand_labels.append(f"#{rank} p_HG={p_high:.3f}\n{slide_id}\n{gt_str}")
        print(f"  [{rank:03d}] {slide_id}  ({cx},{cy})  p_high={p_high:.4f}  {gt_str}")

    for slide in slide_cache.values():
        slide.close()

    meta_df = pd.DataFrame(cand_metadata)
    meta_df.to_csv(OUT_DIR / 'candidates_top30.csv', index=False)
    print(f"[save] {OUT_DIR/'candidates_top30.csv'}")

    # 2) SPIDER reference HG patches
    all_ref = sorted(SPIDER_HG_SRC.glob('*.png'))
    rng = np.random.default_rng(42)
    pick = rng.choice(len(all_ref), N_REF, replace=False)
    ref_meta = []
    ref_thumbs_for_sheet = []
    ref_labels = []
    for i, idx in enumerate(sorted(pick), start=1):
        src = all_ref[int(idx)]
        dst = REF_DIR / f"ref_{i:03d}__{src.name}"
        shutil.copy(src, dst)
        ref_meta.append({'idx': i, 'source': str(src), 'file': dst.name})
        ref_thumbs_for_sheet.append(dst)
        ref_labels.append(f"REF #{i}\n(SPIDER train HG)")
    pd.DataFrame(ref_meta).to_csv(OUT_DIR / 'spider_reference_hg.csv', index=False)
    print(f"[save] {OUT_DIR/'spider_reference_hg.csv'}  ({N_REF} reference patches)")

    # 3) Contact sheets
    make_contact_sheet(cand_thumbs_for_sheet, cand_labels,
                       OUT_DIR / 'contact_sheet_candidates.png', cols=6, tile=240)
    make_contact_sheet(ref_thumbs_for_sheet, ref_labels,
                       OUT_DIR / 'contact_sheet_reference.png', cols=5, tile=240)
    print(f"[save] contact sheets")

    # 4) README
    readme = f"""# SPIDER PDC Candidate Package — for professor review

작성: 2026-05-31
대상: 우리 CRC 슬라이드에서 SPIDER colorectal model의 "Adenocarcinoma High Grade" probability가
       높은 patch들이 실제로 poorly-differentiated 영역인지 확인 요청.

## 배경

- HistAI의 SPIDER colorectal model (Hibou-L + BERT attention head, 13-class)을 우리 슬라이드에
  적용한 결과, SPIDER의 "Adeno HG" 클래스는 거의 fire하지 않음 (max p_high ≈ 0.013).
- 그러나 그 안에서 상대적으로 p_high가 가장 높은 top-{TOP_K} patch를 추출.
- 비교용으로 SPIDER 자체 학습 데이터의 confirmed HG 샘플 {N_REF}장 동봉.

## 확인 부탁드리고 싶은 것

1. **REFERENCE/spider_reference_hg/** 의 {N_REF}장이 SPIDER가 "HG"라고 부르는 형태학.
   교수님 기준으로도 이게 poorly-differentiated CRC에 부합하는지?
2. **candidates/** 의 top-{TOP_K} 후보가 위 reference와 morphology가 유사한지?
   (= 우리 슬라이드에서 SPIDER가 약하게나마 HG라고 본 영역들)
3. 만약 reference와 candidates 모두 PDC에 일치한다면, p_high threshold를 매우 낮게 (>0.005?)
   잡고 SPIDER를 weak PDC detector로 활용 가능할 수 있음.

## 파일 구조

```
candidates/
    NNN__<slide>__x<X>_y<Y>__phigh<P>.png       # 1120x1120 SPIDER input view (~564 µm FoV)
    NNN__<slide>__x<X>_y<Y>__phigh<P>_loc.png   # WSI thumbnail with red box at patch location

spider_reference_hg/
    ref_NNN__<src>.png                          # SPIDER train HG 1120x1120 composites

candidates_top30.csv      # rank, slide, coords, p_high, p_low, top1_class, GT label
spider_reference_hg.csv   # reference list
contact_sheet_candidates.png   # 6×5 grid of all candidate patches
contact_sheet_reference.png    # 5×4 grid of all reference HG patches
```

## 주요 metric

- patch_um ≈ 564 µm (= 2240 px @ L0 0.252 µm/px = 1120 px @ 0.504 µm/px ≈ 20x)
- top-{TOP_K} p_high 범위: [{top['p_high_grade'].min():.4f}, {top['p_high_grade'].max():.4f}]
- top-{TOP_K} 슬라이드 분포: {dict(top['slide'].value_counts())}
- top-{TOP_K} 현재 GT 라벨: {dict(top['gt_label_thr050'].value_counts())}
  (1 = 우리가 non-gland로 어노테이션, 0 = 우리가 gland로 어노테이션, -1 = annotation 없음)

## 검증 결과 (지금까지)

| 메트릭 | 값 |
|---|---|
| SPIDER 자체 HG sanity (500장) | top1=HG 94.8%, mean p_high=0.940 (정상) |
| 우리 슬라이드 적용 시 max p_high | 0.013 (≈ uniform random) |
| 우리 non-gland GT 35장에서 top1 | 모두 "Adeno LG" (p_low=0.88) |
| 13 클래스 중 최고 AUC | 0.645 (Adenoma HG); Adeno HG = 0.443 |
| Logistic combo 5-fold CV AUC | 0.637 (useful threshold 0.7 미달) |

→ SPIDER classification head는 우리 task에 직접 사용 불가지만, 위 reference vs candidate 비교를 통해
   "왜 그런지"의 답을 얻을 수 있을 것으로 기대.
"""
    (OUT_DIR / 'README.md').write_text(readme)
    print(f"[save] {OUT_DIR/'README.md'}")

    # Zip
    shutil.make_archive(str(OUT_DIR), 'zip', root_dir=str(OUT_DIR.parent), base_dir=OUT_DIR.name)
    print(f"\n[zip] {OUT_DIR}.zip")
    print(f"\n=== Done. Total: {len(meta_df)} candidates + {N_REF} reference patches ===")


if __name__ == '__main__':
    main()
