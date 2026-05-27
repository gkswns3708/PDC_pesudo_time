"""
Pipeline scenario explainer — PPT-style figures with zoom-in callouts.

Each figure shows a big WSI thumbnail on the left and several zoomed-in
patch crops on the right, with explicit colored rectangles on the WSI
and connection lines pointing to the zoom panel ("이 부분이 확대된 겁니다").

Outputs in /app/Gland_Seg/results/pipeline_scenario/:
  fig1_training_pipeline.png   : 학습 데이터를 어떻게 만드는가
  fig2_inference_flow.png      : inference 가 어떻게 동작하는가
  fig3_gt_parity_rule.png      : S14-2289-1-6 GT (parity) + 예측 비교
  SCENARIO.md                  : 위 3개 figure 참조 + 설명
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from lxml import etree
from matplotlib.patches import ConnectionPatch, Rectangle


# Korean font
_KOR_FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
fm.fontManager.addfont(_KOR_FONT)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


OUT = Path("/app/Gland_Seg/results/pipeline_scenario")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_SLIDE = "S14-177-1-5"
TRAIN_CLASS = "non-gland"
EVAL_SLIDE = "S14-2289-1-6"
THUMB_MAX = 2200
PATCH = 512
STRIDE_TRAIN = 256

GLAND = (0.15, 0.40, 0.95)
NONGLAND = (0.90, 0.25, 0.20)
ANNOT = (1.00, 0.85, 0.10)

# Distinct callout colors (4-6 callouts per figure)
CALLOUT_COLORS = [
    (0.95, 0.30, 0.30),  # red
    (0.30, 0.65, 0.95),  # cyan
    (0.20, 0.75, 0.35),  # green
    (1.00, 0.65, 0.10),  # orange
    (0.65, 0.30, 0.80),  # purple
    (0.30, 0.30, 0.30),  # dark grey
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def parse_polygons(xml_path):
    tree = etree.parse(str(xml_path))
    pos, neg = [], []
    for ann in tree.getroot().findall(".//Annotation"):
        for reg in ann.findall(".//Region"):
            verts = [(float(v.get("X")), float(v.get("Y")))
                     for v in reg.findall(".//Vertex")]
            if len(verts) < 3:
                continue
            arr = np.array(verts)
            (neg if reg.get("NegativeROA", "0") == "1" else pos).append(arr)
    return pos, neg


def get_thumb(slide_path, max_dim=THUMB_MAX):
    s = openslide.OpenSlide(str(slide_path))
    w, h = s.level_dimensions[0]
    scale = max(w, h) / max_dim
    thumb = np.array(s.get_thumbnail((int(w / scale), int(h / scale))).convert("RGB"))
    s.close()
    return thumb, scale, (w, h)


def render_polys(ax, polys, scale, edge, lw=0.7, ls="-", fill=False, alpha=0.25):
    for p in polys:
        xs = p[:, 0] / scale
        ys = p[:, 1] / scale
        if fill:
            ax.fill(xs, ys, color=edge, alpha=alpha, linewidth=0)
        ax.plot(xs, ys, color=edge, linewidth=lw, linestyle=ls, alpha=0.9)


def draw_patch_squares(ax, xs_thumb, ys_thumb, sz_thumb, colors, alpha=0.55):
    from matplotlib.collections import PatchCollection
    rects = [Rectangle((x, y), sz_thumb, sz_thumb)
             for x, y in zip(xs_thumb, ys_thumb)]
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor='none',
                                       alpha=alpha))


def read_patch_from_svs(svs_path, x, y, patch_size=PATCH):
    s = openslide.OpenSlide(str(svs_path))
    img = np.array(s.read_region((int(x), int(y)), 0,
                                  (patch_size, patch_size)).convert("RGB"))
    s.close()
    return img


def draw_zoom_callout(fig, wsi_ax, zoom_ax, x_thumb, y_thumb, sz_thumb,
                      color, lw_rect=2.5, lw_line=1.0):
    """
    WSI ax 의 (x, y, sz, sz) 사각형 + zoom_ax 의 두 모서리(왼쪽 위·아래)로 연결선.
    더 큰 사각형은 zoom 안에 있는 patch가 어느 위치인지 명시.
    """
    rect = Rectangle((x_thumb, y_thumb), sz_thumb, sz_thumb,
                     fill=False, edgecolor=color, linewidth=lw_rect)
    wsi_ax.add_patch(rect)

    # zoom ax data limits (after imshow): xlim=(0,W), ylim=(H,0)
    zoom_img = zoom_ax.images[0].get_array()
    zh, zw = zoom_img.shape[:2]

    # Two connecting lines: top-right WSI rect → top-left zoom, bottom-right → bottom-left
    pairs = [
        ((x_thumb + sz_thumb, y_thumb), (0, 0)),
        ((x_thumb + sz_thumb, y_thumb + sz_thumb), (0, zh)),
    ]
    for (wxy, zxy) in pairs:
        con = ConnectionPatch(
            xyA=zxy, coordsA=zoom_ax.transData,
            xyB=wxy, coordsB=wsi_ax.transData,
            color=color, lw=lw_line, alpha=0.6,
        )
        fig.add_artist(con)


def make_figure_with_callouts(wsi_thumb, callouts, suptitle, out_path,
                              wsi_overlay_fn=None, figsize=(20, 11),
                              wsi_cols=3, callout_cols=2):
    """
    Layout: wsi on left (wsi_cols), 2 columns of small zoom axes on right.
    callouts = list of dict (length ≤ 6):
        x_abs, y_abs   — level-0 patch top-left coords
        scale          — thumb scale for plotting rect
        zoom_img       — HxWx3 RGB patch image
        title          — title for zoom ax
        color          — callout color
    wsi_overlay_fn(ax) optional — additional drawing on WSI before callouts.
    """
    n = len(callouts)
    n_rows = (n + callout_cols - 1) // callout_cols
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, wsi_cols + callout_cols, figure=fig,
                            wspace=0.10, hspace=0.18,
                            width_ratios=[1] * wsi_cols + [0.85] * callout_cols)

    wsi_ax = fig.add_subplot(gs[:, :wsi_cols])
    wsi_ax.imshow(wsi_thumb)
    wsi_ax.set_xticks([]); wsi_ax.set_yticks([])

    if wsi_overlay_fn is not None:
        wsi_overlay_fn(wsi_ax)

    for i, c in enumerate(callouts):
        r = i // callout_cols
        col = wsi_cols + (i % callout_cols)
        zax = fig.add_subplot(gs[r, col])
        zax.imshow(c["zoom_img"])
        zax.set_xticks([]); zax.set_yticks([])
        for sp in zax.spines.values():
            sp.set_edgecolor(c["color"]); sp.set_linewidth(3.0)
        zax.set_title(c["title"], fontsize=10, color=c["color"], pad=4)

        x_thumb = c["x_abs"] / c["scale"]
        y_thumb = c["y_abs"] / c["scale"]
        sz_thumb = PATCH / c["scale"]
        # Inflate visibly (×3 for visibility on big thumb)
        infl = 3.0
        cx = x_thumb + sz_thumb / 2
        cy = y_thumb + sz_thumb / 2
        s2 = sz_thumb * infl
        x_draw = cx - s2 / 2
        y_draw = cy - s2 / 2
        draw_zoom_callout(fig, wsi_ax, zax, x_draw, y_draw, s2, c["color"])

    fig.suptitle(suptitle, fontsize=13)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Picking helpers
# ─────────────────────────────────────────────
def pick_spread_coords(coords, n=4, seed=0):
    """k-means-like simple spread: divide bbox into n cells and pick one from each."""
    coords = np.asarray(coords)
    if len(coords) <= n:
        return coords
    rng = np.random.default_rng(seed)
    x_min, y_min = coords.min(0); x_max, y_max = coords.max(0)
    # split into roughly sqrt(n) x sqrt(n) cells (or n x 1)
    nx = int(np.ceil(np.sqrt(n))); ny = int(np.ceil(n / nx))
    picks = []
    for ix in range(nx):
        for iy in range(ny):
            xl = x_min + (x_max - x_min) * ix / nx
            xh = x_min + (x_max - x_min) * (ix + 1) / nx
            yl = y_min + (y_max - y_min) * iy / ny
            yh = y_min + (y_max - y_min) * (iy + 1) / ny
            mask = ((coords[:, 0] >= xl) & (coords[:, 0] < xh)
                    & (coords[:, 1] >= yl) & (coords[:, 1] < yh))
            cands = coords[mask]
            if len(cands):
                picks.append(cands[rng.integers(0, len(cands))])
            if len(picks) >= n:
                break
        if len(picks) >= n:
            break
    return np.array(picks[:n])


# ─────────────────────────────────────────────
# Figure 1: training pipeline
# ─────────────────────────────────────────────
def fig1_training_pipeline():
    print(f"\n=== Fig 1: training pipeline ({TRAIN_SLIDE}) ===")
    svs = Path("/app/Gland_Seg/Data/S14/SVS") / f"{TRAIN_SLIDE}.svs"
    xml = Path("/app/Gland_Seg/Data/S14/Annotation") / f"{TRAIN_SLIDE}_S.xml"
    patches_dir = Path("/app/Gland_Seg/patches_stainnorm") / TRAIN_SLIDE / TRAIN_CLASS

    thumb, scale, _ = get_thumb(svs)
    pos_polys, neg_polys = parse_polygons(xml)

    coords = []
    for f in sorted(patches_dir.glob(f"{TRAIN_SLIDE}_*.png")):
        try:
            x, y = int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])
            coords.append([x, y])
        except Exception:
            pass
    coords = np.array(coords)
    print(f"  extracted patches: {len(coords)}, polys: pos={len(pos_polys)}, neg={len(neg_polys)}")

    picks = pick_spread_coords(coords, n=4, seed=7)
    print(f"  picked {len(picks)} spread patches for callouts")

    callouts = []
    for i, (x, y) in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(patches_dir / f"{TRAIN_SLIDE}_{int(x)}_{int(y)}.png")),
                            cv2.COLOR_BGR2RGB)
        callouts.append({
            "x_abs": float(x), "y_abs": float(y), "scale": scale,
            "zoom_img": img,
            "title": f"label = {TRAIN_CLASS}\n(x={int(x)}, y={int(y)})",
            "color": CALLOUT_COLORS[i],
        })

    def overlay(ax):
        # 1) positive polygon filled + outlined
        render_polys(ax, pos_polys, scale, edge=ANNOT, fill=True, alpha=0.20)
        render_polys(ax, pos_polys, scale, edge=ANNOT, lw=1.2)
        # 2) negative polygons outline only
        render_polys(ax, neg_polys, scale, edge=(1, 0.4, 0.7), lw=0.8, ls="--")
        # 3) extracted patch positions as small red squares (background context)
        sz = PATCH / scale
        draw_patch_squares(ax,
                           coords[:, 0] / scale, coords[:, 1] / scale,
                           sz, [NONGLAND] * len(coords), alpha=0.30)
        ax.set_title(
            f"{TRAIN_SLIDE} ({TRAIN_CLASS})  ·  "
            f"노란선·면=positive polygon  ·  분홍점선=negative ROA\n"
            f"빨강 점들={len(coords):,} 개 추출 patch (전부 label={TRAIN_CLASS})",
            fontsize=11,
        )
        # Legend
        handles = [
            mpatches.Patch(facecolor=ANNOT, alpha=0.30, edgecolor=ANNOT, label="positive polygon (학습 영역)"),
            mpatches.Patch(facecolor="none", edgecolor=(1, 0.4, 0.7), linestyle="--", label="negative ROA (제외)"),
            mpatches.Patch(facecolor=NONGLAND, alpha=0.45, label="추출된 patch 위치"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

    make_figure_with_callouts(
        thumb, callouts,
        suptitle=("학습 데이터 구성 — WSI 의 positive polygon 안쪽에서만 patch 추출. "
                  "옆 4개는 실제 잘린 patch (Macenko 정규화 후) — 박스↔연결선으로 위치 표시"),
        out_path=OUT / "fig1_training_pipeline.png",
        wsi_overlay_fn=overlay,
    )
    print(f"  saved {OUT / 'fig1_training_pipeline.png'}")


# ─────────────────────────────────────────────
# Figure 2: inference flow
# ─────────────────────────────────────────────
def fig2_inference_flow():
    print(f"\n=== Fig 2: inference flow ({EVAL_SLIDE}) ===")
    base = Path("/app/Gland_Seg/results") / EVAL_SLIDE
    thumb = np.load(base / "thumbnail.npy")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    scale = meta["scale"]
    df = pd.read_csv(base / "per_patch_predictions_with_hardvote.csv")
    svs_path = Path("/app/Gland_Seg/Data/S14/SVS") / f"{EVAL_SLIDE}.svs"

    # Pick 4 callouts: 2 confident gland (high p), 2 confident non-gland (low p)
    df_sorted = df.sort_values("p_gland_ensemble", ascending=False)
    pick_g = df_sorted.head(2).sample(2, random_state=1)
    pick_n = df_sorted.tail(2).sample(2, random_state=2)
    picks = pd.concat([pick_g, pick_n], ignore_index=True)

    callouts = []
    for i, row in picks.iterrows():
        img = read_patch_from_svs(svs_path, int(row.x), int(row.y))
        pred = row["pred_hardvote"]
        col = GLAND if pred == "gland" else NONGLAND
        # use callout-specific color for visibility, but mark border w/ class color
        cc = CALLOUT_COLORS[i % len(CALLOUT_COLORS)]
        callouts.append({
            "x_abs": float(row.x), "y_abs": float(row.y), "scale": scale,
            "zoom_img": img,
            "title": (f"pred = {pred}  ·  p_gland_ens = {row.p_gland_ensemble:.2f}\n"
                      f"(x={int(row.x)}, y={int(row.y)})"),
            "color": cc,
        })

    def overlay(ax):
        # Light prediction overlay (all patches colored by hardvote)
        sz = PATCH / scale
        colors = [GLAND if p == "gland" else NONGLAND for p in df.pred_hardvote.values]
        draw_patch_squares(ax, df.x.values / scale, df.y.values / scale,
                           sz, colors, alpha=0.40)
        n_g = int((df.pred_hardvote == "gland").sum())
        n_n = int((df.pred_hardvote == "non-gland").sum())
        ax.set_title(
            f"{EVAL_SLIDE}  ·  3-모델 hardvote 예측 (모든 tissue patch)\n"
            f"파랑={n_g:,} (gland) · 빨강={n_n:,} (non-gland) · 총 {len(df):,}",
            fontsize=11,
        )
        handles = [
            mpatches.Patch(color=GLAND, label="predicted gland"),
            mpatches.Patch(color=NONGLAND, label="predicted non-gland"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

    make_figure_with_callouts(
        thumb, callouts,
        suptitle=(f"Inference — 모든 tissue patch 가 강제로 binary 분류됨. "
                  f"옆 4개는 모델이 가장 확신한 양 클래스 patch (위 2 = gland, 아래 2 = non-gland)"),
        out_path=OUT / "fig2_inference_flow.png",
        wsi_overlay_fn=overlay,
    )
    print(f"  saved {OUT / 'fig2_inference_flow.png'}")


# ─────────────────────────────────────────────
# Figure 3: GT parity rule + comparison
# ─────────────────────────────────────────────
def fig3_gt_parity_rule():
    print(f"\n=== Fig 3: GT parity rule + comparison ({EVAL_SLIDE}) ===")
    base = Path("/app/Gland_Seg/results") / EVAL_SLIDE
    thumb = np.load(base / "thumbnail.npy")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()
    scale = meta["scale"]
    df = pd.read_csv(base / "per_patch_predictions_with_hardvote.csv")
    ann = np.load(base / "annotation.npz", allow_pickle=True)
    polys = list(ann["positive"]) + list(ann["negative"])
    svs_path = Path("/app/Gland_Seg/Data/S14/SVS") / f"{EVAL_SLIDE}.svs"

    H, W = thumb.shape[:2]
    counter = np.zeros((H, W), dtype=np.int16)
    for p in polys:
        pts = (p / scale).round().astype(np.int32)
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        counter += m

    cx = ((df.x + PATCH / 2) / scale).astype(int).clip(0, W - 1)
    cy = ((df.y + PATCH / 2) / scale).astype(int).clip(0, H - 1)
    n_in = counter[cy, cx]
    gt = np.where(n_in == 0, -1, np.where(n_in == 1, 0, 1))
    pred = (df.pred_hardvote.values == "non-gland").astype(int)

    # Pick callouts: correct-gland, correct-nongland, false-positive (gt gland, pred nongland), false-negative (gt nongland, pred gland)
    df2 = df.assign(_gt=gt, _pred=pred)
    cands = {
        "correct gland\n(GT=gland, pred=gland)":
            df2[(df2._gt == 0) & (df2._pred == 0)].sort_values("p_gland_ensemble", ascending=False).head(20),
        "correct non-gland\n(GT=non-gland, pred=non-gland)":
            df2[(df2._gt == 1) & (df2._pred == 1)].sort_values("p_gland_ensemble", ascending=True).head(20),
        "오분류: gland→non-gland\n(GT=gland, pred=non-gland)":
            df2[(df2._gt == 0) & (df2._pred == 1)].sort_values("p_gland_ensemble", ascending=True).head(20),
        "오분류: non-gland→gland\n(GT=non-gland, pred=gland)":
            df2[(df2._gt == 1) & (df2._pred == 0)].sort_values("p_gland_ensemble", ascending=False).head(20),
    }
    callouts = []
    for i, (title, sub) in enumerate(cands.items()):
        if len(sub) == 0:
            print(f"  [warn] no candidate for: {title}")
            continue
        row = sub.iloc[0]  # most representative
        img = read_patch_from_svs(svs_path, int(row.x), int(row.y))
        callouts.append({
            "x_abs": float(row.x), "y_abs": float(row.y), "scale": scale,
            "zoom_img": img,
            "title": f"{title}\np_g_ens={row.p_gland_ensemble:.2f}",
            "color": CALLOUT_COLORS[i],
        })

    def overlay(ax):
        # Show parity-based GT mask (depth: 0=gray, 1=blue, 2+=red)
        cmap = matplotlib.colors.ListedColormap([
            (0.85, 0.85, 0.85),  # 0
            GLAND,               # 1
            NONGLAND,            # 2+
        ])
        depth_vis = np.clip(counter, 0, 2)
        ax.imshow(depth_vis, cmap=cmap, alpha=0.50, vmin=0, vmax=2,
                  interpolation="nearest")
        render_polys(ax, polys, scale, edge=ANNOT, lw=0.5)
        n_g = int((gt == 0).sum()); n_n = int((gt == 1).sum()); n_skip = int((gt == -1).sum())
        ax.set_title(
            f"{EVAL_SLIDE}  ·  parity 규칙으로 만든 GT  "
            f"(파랑=gland {n_g:,} · 빨강=non-gland {n_n:,} · 회색=평가제외 {n_skip:,})\n"
            f"노란선 = 교수님 polygon 149 개",
            fontsize=11,
        )
        handles = [
            mpatches.Patch(color=GLAND, label="GT = gland (ROI 박스 안)"),
            mpatches.Patch(color=NONGLAND, label="GT = non-gland (inner polygon)"),
            mpatches.Patch(color=(0.85, 0.85, 0.85), label="평가 제외 (ROI 밖)"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

    make_figure_with_callouts(
        thumb, callouts,
        suptitle=(f"{EVAL_SLIDE} 평가 — parity 규칙 GT vs 모델 hardvote 예측. "
                  f"옆 4개는 4가지 케이스 (정답 2종 + 오류 2종) 실제 patch"),
        out_path=OUT / "fig3_gt_parity_rule.png",
        wsi_overlay_fn=overlay,
        figsize=(20, 12),
    )
    print(f"  saved {OUT / 'fig3_gt_parity_rule.png'}")


# ─────────────────────────────────────────────
# SCENARIO.md
# ─────────────────────────────────────────────
def write_md():
    base = Path("/app/Gland_Seg/results") / EVAL_SLIDE
    df = pd.read_csv(base / "per_patch_predictions_with_hardvote.csv")
    meta = np.load(base / "slide_meta.npy", allow_pickle=True).item()

    md = f"""# Pipeline scenario — gland vs non-gland binary classification

각 figure 는 PPT 스타일 zoom-in: 왼쪽 WSI 위에 색상 박스(확대된 위치)를 그리고,
오른쪽에 그 위치의 실제 patch 가 동일 색상 테두리로 표시됩니다. 박스↔patch 가 연결선으로 매핑.

---

## 1. 학습 데이터 구성 — `fig1_training_pipeline.png`

![fig1](fig1_training_pipeline.png)

- 학습 슬라이드 1장 = **한 가지 class** (파일명 `_S` = non-gland, `_G` = gland).
- 교수님이 그 class 영역만 polygon(positive, 노란선)으로 표시. 안의 작은 negative ROA (분홍 점선)는 제외.
- 노란 영역 안에서 sliding window (patch {PATCH}, stride {STRIDE_TRAIN}) → mask_ratio ≥ 0.5 인 곳만 채택.
- 영역 밖 normal tissue 는 **학습 데이터에서 빠짐**.
- 4개 색상 박스 = 임의로 뽑은 patch 위치, 옆에 실제 잘린 patch (Macenko 정규화 후) 보여줌.

---

## 2. Inference — `fig2_inference_flow.png`

![fig2](fig2_inference_flow.png)

- 외부 슬라이드 ({EVAL_SLIDE}, {meta['slide_w']:,}×{meta['slide_h']:,} px) 전체에 stride 512 sliding.
- tissue 가 있는 패치만 후보 — {len(df):,} 개.
- 각 patch → Macenko → 224 resize → 3 모델 forward → hardvote → **gland/non-gland 둘 중 하나로 강제 분류**.
- 4개 callout = 가장 확신한 gland 2개 + 가장 확신한 non-gland 2개.

---

## 3. {EVAL_SLIDE} 평가 — `fig3_gt_parity_rule.png`

![fig3](fig3_gt_parity_rule.png)

교수님 안내: *"큰 네모 박스로 영역을 지정했고요, 그 안에서 annotation 한 것이 non-gland cancer, 나머지는 gland cancer + 일부 normal."*

XML 안 149 polygon 에서 parity 규칙 ([compute_gt_metrics.py:78](Gland_Seg/Code/compute_gt_metrics.py#L78)):

```
patch 중심이 들어 있는 polygon 수 →
  0개 (ROI 밖)        → 평가 제외 (회색)
  1개 (ROI 박스 안만) → GT = gland (파랑)  ← *normal tissue 도 여기 포함됨*
  2개 이상 (inner)    → GT = non-gland (빨강)
```

4개 callout = 4가지 분류 결과 (정답 2종 + 오분류 2종) 실제 patch.

---

## 자주 헷갈리는 포인트

| 질문 | 답 |
|---|---|
| normal 도 분류하나? | 아니요 — 모델은 항상 binary. normal tissue 도 gland/non-gland 중 하나로 예측됨 |
| 학습은 무엇을 보고 하나? | 슬라이드의 positive polygon 안쪽 패치만. 영역 밖 normal 은 학습 안 함 |
| GT 의 normal 처리? | S14-2289-1-6 GT 에서 normal 은 gland 와 같이 label 0 으로 묶임 — gland F1 가 약간 너그러움 |
| 왜 평가 patch 가 9,605 보다 적나? | parity 규칙으로 GT 부여된 patch (ROI 안) 만 평가 → ~2,800 |
| non-gland F1 낮은 이유? | (i) 학습 클래스 불균형, (ii) non-gland 패턴 다양성 부족 |

생성 스크립트: [viz_pipeline_scenario.py](../../Code/viz_pipeline_scenario.py)
"""
    (OUT / "SCENARIO.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {OUT / 'SCENARIO.md'}")


def main():
    fig1_training_pipeline()
    fig2_inference_flow()
    fig3_gt_parity_rule()
    write_md()
    print(f"\nAll outputs in: {OUT}")


if __name__ == "__main__":
    main()
