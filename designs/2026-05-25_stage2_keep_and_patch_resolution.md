# Design — Stage 2 유지 + patch 크기·stride 실험 (해상도 ↑, 경계 mixing ↓)

작성일: **2026-05-25**
관련 stage: Stage 2 (gland vs non-gland, patch-level binary classification)
연결 사이클: [2026-05-24 6장 non-gland 추가 학습](#) 결과 (macro-F1 0.6020 → 0.6205) 의 후속 실험

---

## Context — 왜 이 실험을 하는가

이전 사이클 (S14-2289-1-6 외부 평가):
- 14-slide virchow2 단독: macro-F1 0.5535 (오히려 약간 후퇴, 클래스 편향)
- 3-모델 hardvote 앙상블: macro-F1 0.6205 (소폭 향상)
- 핵심 약점: **non-gland F1 0.29** (gland F1 0.95 대비 매우 낮음)

근본 원인 가설 두 줄기:
1. **데이터 다양성** — non-gland 패턴 수 부족 (이미 6장 추가했으나 한계 확인)
2. **공간 해상도** — 512×512 patch 가 한 클래스만 담는다는 가정이 약함. 경계 영역에서 한 patch 안에 gland + non-gland 가 섞여 학습/평가가 모두 흐릿해짐

이번 사이클은 **(2) 해상도 가설**을 직접 검증:
> patch 크기를 줄이고 stride 도 비례 축소하면, 한 patch 내 mixing 이 줄어 분류 신뢰도가 올라간다.

또한 **Stage 2 자체 (foundation model + binary patch classification) 는 그대로 유지** — 구조 변경 없이 입력 스케일만 조정해 *효과의 한계* 부터 측정.

---

## 가설과 기대 효과

| 변경 전 (baseline) | 변경 후 (실험) | 기대 |
|---|---|---|
| patch 512px → resize 224 | patch 256px → resize 224 | 모델 시야 ×2 zoom, 세포·glandular 구조가 더 선명 |
| stride 256 (학습) / 512 (추론) | stride 128 (학습/추론) | 추론 해상도 ×4, 픽셀 prob map 부드러움 |
| 물리적 시야 ~129×129 μm @ 0.252 μm/px | 물리적 시야 ~64×64 μm | 한 클래스만 들어 있을 확률 ↑ |

예상 시나리오:
- **호전 시 (macro-F1 ≥ 0.70)**: 해상도가 원인이었음 → 256/128 을 default 로 채택
- **소폭 호전 (0.62 → 0.65 ~ 0.69)**: 일부 기여, 다음 사이클에 segmentation head 또는 다른 PEFT 검토
- **변화 없음 또는 후퇴**: 데이터 다양성이 더 큰 병목 → patch size 가 아니라 class-balanced sampler / partial FT / 추가 annotation 으로 전환

---

## Plan — Two Steps

### Step 1 — Baseline 박제 (실행 0분, 단순 복사)

이미 끝난 14-slide / 512px 결과를 비교용으로 보존:

```bash
mkdir -p /app/Gland_Seg/results/_baseline_512px_14slide
cp /app/Gland_Seg/results/S14-2289-1-6/metrics_vs_gt.csv \
   /app/Gland_Seg/results/_baseline_512px_14slide/
cp /app/Gland_Seg/results/S14-2289-1-6/prediction_summary.md \
   /app/Gland_Seg/results/_baseline_512px_14slide/
cp /app/Gland_Seg/checkpoints/best_model_virchow2_full.pth \
   /app/Gland_Seg/checkpoints/best_model_virchow2_full_512px_14slide.pth
```

→ 이후 S14-2289 추론·평가가 256px 결과로 덮어쓰여도 baseline 보존.

### Step 2 — Patch 256 / stride 128 실험

#### 2-1. 설정 변경 (config.py)

기존 값 → 새 값:
```python
patch_size:  int = 512  →  256
stride:      int = 256  →  128
output_dir:  str = "/app/Gland_Seg/patches_stainnorm"
          →  "/app/Gland_Seg/patches_stainnorm_256"
```
backbone, head_type, unfreeze_epoch 등은 그대로 (virchow2 / linear / 999 = LP).

#### 2-2. Patch 재추출 (14 슬라이드 전부, ~30-60분)

```bash
cd /app/Gland_Seg/Code
/root/miniconda3/envs/tiatoolbox/bin/python create_dataset.py \
    2>&1 | tee /app/Gland_Seg/logs/extract_256px_14slide.log
```
(이번에는 `--slides` 없이 14장 전부)

예상 patch 수: 기존 73,999 의 ~4배 ≈ **약 280-300k**
디스크: 패치 PNG ~40-60 GB (현재 /app 1.6 TB 여유 → OK)

#### 2-3. virchow2 재학습 (~8-12시간 추정, LP)

```bash
cd /app/Gland_Seg/Code
PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
  /root/miniconda3/envs/tiatoolbox/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 train_full.py \
  2>&1 | tee /app/Gland_Seg/logs/train_full_virchow2_256px_14slide.log
```

체크포인트: `best_model_virchow2_full.pth` (자동) → 학습 후 즉시 rename:
```bash
mv /app/Gland_Seg/checkpoints/best_model_virchow2_full.pth \
   /app/Gland_Seg/checkpoints/best_model_virchow2_full_256px_14slide.pth
```
그리고 비교 baseline (`_full_512px_14slide.pth`) 도 보존돼 있는 상태.

#### 2-4. 외부 평가 (S14-2289-1-6, stride 128)

inference 스크립트는 `config.patch_size` 와 CLI `--stride` 사용. config 가 이미 256 이라 patch는 자동, stride 만 명시:

```bash
cd /app/Gland_Seg/Code
# 이 추론은 256px 체크포인트를 다시 가져와야 하므로 임시 rename
cp /app/Gland_Seg/checkpoints/best_model_virchow2_full_256px_14slide.pth \
   /app/Gland_Seg/checkpoints/best_model_virchow2_full.pth

/root/miniconda3/envs/tiatoolbox/bin/python infer_external_slide.py S14-2289-1-6 \
    --models virchow2 uni2 phikon-v2 --stride 128 \
    2>&1 | tee /app/Gland_Seg/logs/infer_2289_256px.log

# 결과 별도 폴더로
mv /app/Gland_Seg/results/S14-2289-1-6 \
   /app/Gland_Seg/results/S14-2289-1-6_256px
```

**주의 — uni2/phikon-v2 체크포인트는 512px 로 학습돼 있어** 이번 256px 추론이 분포-mismatch 일 수 있음. 두 가지 선택:
- (a) virchow2 단독으로만 평가 (uni2/phikon-v2 결과는 참고만, 앙상블 skip)
- (b) uni2/phikon-v2 도 256px 로 재학습 (시간·디스크 추가 비용) — 이번 사이클 범위 밖 권장

→ 이 design 은 **(a) 채택**. 앙상블 비교는 다음 사이클로.

이후 summarize + GT F1:
```bash
/root/miniconda3/envs/tiatoolbox/bin/python summarize_external_predictions.py S14-2289-1-6_256px
/root/miniconda3/envs/tiatoolbox/bin/python compute_gt_metrics.py S14-2289-1-6_256px
```
→ 두 스크립트가 slide 이름을 그대로 폴더 명으로 쓰니 `S14-2289-1-6_256px` 로 분리 보존.

#### 2-5. 결과 비교 표 작성

`/app/Gland_Seg/designs/2026-05-25_results.md` 또는 prediction_summary.md 에 추가:

| Setting | n_train_patches | val_f1 (10% holdout) | external acc | external macro-F1 | non-gland F1 | gland F1 |
|---|---:|---:|---:|---:|---:|---:|
| 512 / 256 baseline (virchow2 단독) | 73,999 | 0.9897 | 0.804 | 0.5535 | 0.2191 | 0.8880 |
| 256 / 128 (virchow2 단독) | TBD | TBD | TBD | TBD | TBD | TBD |
| 변화 | — | — | — | — | — | — |

---

## 위험·주의

1. **Foundation model scale mismatch**
   - virchow2 는 보통 ~256-512 μm 시야로 사전학습됨. 64μm 패치 (256 @ 0.252 μm/px) 는 더 zoom-in 된 분포 → 사전학습 표현이 일부 부적합할 수 있음
   - 대안: 같은 시야를 유지하면서 stride 만 줄임 (`patch=512 stride=128`) → "보는 건 같으나 해상도만 ↑". 다만 이건 한 patch 안 mixing 문제는 해결 못 함

2. **학습 시간 + 디스크**
   - 약 10시간, 40-60 GB. 다른 학습/추론과 GPU 충돌 주의

3. **uni2 / phikon-v2 mismatch**
   - 256px 추론을 그들 가중치로 돌리면 분포 mismatch → 비교 일관성 깨짐
   - virchow2 단독 비교에 한정

4. **결과 해석 trap**
   - val_f1 (10% holdout) 은 거의 항상 ~0.99 로 saturate → train/test gap 본질 안 보임
   - 평가 기준은 **항상 외부 S14-2289 macro-F1 / non-gland F1**

---

## 결정 트리 (실험 후)

| 새 setting (256/128) 결과 | 해석 | 다음 액션 |
|---|---|---|
| macro-F1 ≥ 0.70 | 해상도가 핵심 병목 | 256/128 default 채택, uni2/phikon-v2 도 재학습 (다음 사이클) |
| 0.65 ≤ macro-F1 < 0.70 | 부분 효과 | 256/128 + 추가 데이터 / class-balanced sampler 병행 |
| 0.62 ± 0.02 (변화 미미) | 해상도 영향 적음 | 256 폐기, segmentation head 또는 partial FT 로 방향 전환 |
| macro-F1 후퇴 | foundation model scale mismatch | 동일 patch=512 + stride=128 만 시도해보고, 그래도 별로면 segmentation 방향 |

---

## 산출물 (이 사이클 끝)

- `/app/Gland_Seg/checkpoints/best_model_virchow2_full_512px_14slide.pth` (보존 baseline)
- `/app/Gland_Seg/checkpoints/best_model_virchow2_full_256px_14slide.pth` (신규)
- `/app/Gland_Seg/patches_stainnorm_256/` (신규 patch dir, 보관)
- `/app/Gland_Seg/results/S14-2289-1-6_256px/` (신규 추론 결과)
- `/app/Gland_Seg/logs/{extract_256px_14slide, train_full_virchow2_256px_14slide, infer_2289_256px}.log`
- `/app/Gland_Seg/designs/2026-05-25_results.md` (결과 비교 표 — 실험 끝나면 자동 생성)

---

## 다음 사이클 후보 (이 실험 결과 따라)

- **A. Class-balanced sampler 도입**: WeightedRandomSampler 추가 → gland/non-gland 비율 보정
- **B. Partial fine-tune**: virchow2 last 4 block unfreeze (config.unfreeze_epoch=5)
- **C. LoRA / adapter**: parameter-efficient FT
- **D. Segmentation head**: virchow2 backbone + DPT/Mask2Former decoder (별도 사이클, 코드량 큼)
- **E. Stage 1 cascade 통합**: Kather 100k ResNet-18 + Stage 2 결합 — normal tissue 추론 false positive 제거 (연구 본 목적 PDC 분석에 직접 기여)

추천 우선순위: **E → A → B → C → D**
- E 는 연구 최종 목표(PDC overlay)에 직접 기여, 학습 부담 없음
- A 는 가장 가벼운 코드 변경, 가능성 높음
- B/C 는 모델 더 짜내기
- D 는 본격적 framework 변경
