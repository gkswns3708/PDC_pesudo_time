# Design — Phase B (partial FT) → C (LoRA) → E (Stage 1 cascade)

작성일: **2026-05-27**
이전 사이클 결과: [2026-05-25 patch resolution 실험](2026-05-25_stage2_keep_and_patch_resolution.md) → 결론 "데이터·해상도 부족 아닌 표현 학습 부족 (LP 한계)"

---

## Context — 왜 이 3개를 순서대로

이전 사이클에서:
- **LP (linear probing)** 모드에서 ext F1 이 epoch 1 ≈ epoch 30 ≈ 0.53 → 학습이 외부 분포에 영향 없음
- **256px ↓ 변경**도 효과 미미 (오히려 0.55 → 0.53 후퇴)
- 결론: backbone freeze 상태로 head 만 조정해서는 외부 non-gland 패턴을 포착 불가

이번 사이클은 **표현 학습 (representation learning)** 방향으로 3단계:

| Phase | 방법 | 가설 |
|---|---|---|
| **B** | last 4 transformer block unfreeze (partial FT) | backbone 마지막 층만 외부 분포에 적응 → ext F1 ↑ |
| **C** | LoRA (PEFT) | partial FT 보다 안정적이고 가벼움, 같은 효과 + 더 적은 over-fit |
| **E** | Stage 1 cascade (Kather 100k ResNet-18 mask) | 모델 성능 자체보다 *추론 시 normal tissue 제거*로 false positive ↓, 본 목적 PDC 분석에 직접 기여 |

데이터·patch 설정은 **256px / stride 128 / 14-slide 유지** (B/C/E 끼리 직접 비교 가능하도록 일관성). 512px 결과(baseline)는 외부 reference 로만.

---

## 비교 기준 표 (전 cycle 누적)

| Setting | val_f1 | ext macro-F1 | ext gland | ext non-gland |
|---|---:|---:|---:|---:|
| baseline 512px LP (이전) | 0.9897 | 0.5535 | 0.8880 | 0.2191 |
| 256px LP (직전) | 0.9892 | **0.5346** | 0.8848 | 0.1902 |
| **256px partial FT (B)** — 이번 | TBD | TBD | TBD | TBD |
| **256px LoRA (C)** — 이번 | TBD | TBD | TBD | TBD |
| **256px + Stage 1 cascade (E)** — 이번 | TBD | TBD | TBD | TBD |

---

## Phase B — Partial fine-tune (last 4 blocks)

### 변경 사항 (코드)

**config.py 한 줄**:
```python
unfreeze_epoch: int = 5   # 0~4 epoch: LP, 5+ epoch: last 4 block FT (LR/10)
```

기존 학습 코드 [train_full.py:141-152](Gland_Seg/Code/train_full.py#L141-L152) 가 이미 `unfreeze_epoch + 1` 시점에 `unfreeze_all()` 호출하고 optimizer 재구성 (lr/10) 하는 로직을 가지고 있음. 단 `unfreeze_all` 이 모든 backbone 을 푸는지, last 4 만 푸는지 확인 필요.

→ 확인 결과: [model.py:69-84](Gland_Seg/Code/model.py#L69-L84) `unfreeze_last_n=4` 가 default. `freeze_early_layers` 가 이걸 적용. `unfreeze_all` 은 전체 풀기.

**진짜 partial FT (last 4 만) 을 하려면** 다음 중 하나:
- 옵션 1: `unfreeze_all` 대신 *아무 것도 안 하고* 기존 `freeze_early_layers(..., unfreeze_last_n=4)` 적용된 상태에서 학습 (이미 `freeze_early_layers` 호출 시 last 4 만 학습 가능 상태로 만듦)
- 옵션 2: train_full.py 의 unfreeze 로직을 수정해서 `unfreeze_all` 대신 `freeze_early_layers(unfreeze_last_n=4)` 호출

[model.py:97](Gland_Seg/Code/model.py#L97) 의 `freeze_early_layers(model, backbone=config.backbone)` 는 LP (head만) 만 학습되는지, 아니면 last 4 unfreeze 되는지가 핵심 — 코드 보고 확인.

→ **확인 결과**: `unfreeze_last_n=4` 인스턴스 변수가 default 인데 `freeze_early_layers` 함수가 어떻게 동작하는지 확인 필요. 코드 읽기 후 결정.

### 실행 절차

1. 현재 256px LP 결과 박제 (체크포인트 + 로그 rename)
2. config.unfreeze_epoch = 5
3. 학습 재시작 (DDP, 2 GPU)
4. 끝나면 best_model_virchow2_full.pth + best_model_virchow2_full_byext.pth → `_256px_14slide_partialFT*.pth` 로 rename
5. 외부 평가 (stride 128 으로 full)

### 예상 시간

- LP epoch (5 epoch) + partial FT epoch (15-20 epoch, LR/10 라 천천히 수렴) = **6-8시간**
- Per epoch ext eval 시간 동일 (~1.5분)

### 결정 트리 (B 결과로)

| ext macro-F1 | 해석 | 다음 |
|---|---|---|
| ≥ 0.70 | 표현 학습이 핵심이었음 | C/E 계속 진행 (cascade 도 가치) |
| 0.60-0.70 | 부분 효과 | C 시도 (LoRA 가 더 안정일 수 있음) |
| 0.55-0.60 | 비슷한 한계 | C, E 둘 다 try, 안 되면 단순 데이터 부족 |
| < 0.55 | partial FT 도 무효 | data side 로 회귀 |

---

## Phase C — LoRA (PEFT)

### 변경 사항 (코드)

**필요한 라이브러리**:
```bash
/root/miniconda3/envs/tiatoolbox/bin/pip install peft
```

**model.py 신규 옵션**:
```python
from peft import LoraConfig, get_peft_model

def create_model(num_classes=2, pretrained=True, backbone="virchow2",
                  head_type="linear", lora=False, lora_r=8, lora_alpha=16):
    # ... 기존 코드 ...
    if lora:
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
            target_modules=["qkv", "proj", "fc1", "fc2"],  # ViT 블록 attention/MLP
            bias="none",
        )
        wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg)
    return wrapper
```

**train_full.py 변경**:
- `create_model(..., lora=True)` 으로 호출
- `freeze_early_layers` 대신 LoRA 가 자동으로 backbone freeze + adapter only 학습
- 체크포인트 저장 시 PEFT save_pretrained 사용 (또는 state_dict 그대로)

### 실행 절차

1. partial FT 결과 박제
2. peft 설치
3. create_model 에 lora 옵션 추가
4. config.lora = True (또는 환경변수)
5. 학습 (LP 수준 속도, ~3-5 시간)
6. 외부 평가

### 예상 시간

- LoRA 는 trainable param ~0.5-1% → 빠름, **3-5시간** 예상
- 학습 안정성 ↑ (over-fit 위험 낮음)

### 위험

- target_modules 이름이 virchow2 의 timm 구현체와 정확히 매치되는지 사전 확인 필요 (`model.backbone.named_modules()` 로 확인)
- 저장·로드 호환성 (PEFT 형식 vs 우리 기존 state_dict 직접 저장 방식)

---

## Phase E — Stage 1 cascade

연구 최종 목표 (PDC 위치 분석) 와 가장 직접 연결된 단계. 학습이 아니라 **추론 파이프라인 통합**.

### 필요 자원 확인

- Stage 1 model (Kather 100k ResNet-18): **위치 파악 필요**. 현재 `/app/Gland_Seg/checkpoints/` 에 `best_model_resnet18_fold*.pth` 가 있지만 이건 LOSO 실험용 (gland vs non-gland), Kather 100k 학습 모델 아님
- → Kather 100k 가중치를 다른 위치(`/Public`?)에서 찾거나, torchvision/timm 의 사전학습 ResNet-18 + Kather 100k linear head 를 새로 학습할 가능성 있음

### 변경 사항 (코드, 자원 있다는 가정)

신규 `Code/stage1_infer.py`:
- 입력: WSI + Stage 1 ResNet-18 가중치
- 출력: cancer mask (thumbnail 해상도) — 한 patch 라도 cancer 면 1, 아니면 0
- API: `def cancer_mask(slide_path) -> ndarray[bool]`

신규 `Code/cascade_eval.py`:
- Stage 1 mask + Stage 2 prediction → 최종 prediction
- Stage 1 이 normal 이라고 판단한 patch 는 Stage 2 결과 무시 (자동으로 "non-cancer" 분류)
- Stage 1 이 cancer 라고 판단한 patch 만 Stage 2 의 gland/non-gland 적용
- GT 비교 (compute_gt_metrics 와 같은 parity)

→ 평가 metric 보고 cascade 효과 분석.

### 실행 절차

1. Stage 1 모델 위치 파악 (없으면 user 에게 질문 / 새로 학습)
2. stage1_infer.py 구현 + S14-2289-1-6 에 적용
3. cascade_eval.py 구현 → cascade 와 비-cascade ext F1 비교
4. cascade 가 효과 있으면 학습 pipeline 자체에도 cancer mask 활용 (학습 patch 가 이미 cancer 영역 인지 보장)

### 예상 시간

- 모델 위치 파악 + 구현: **반나절 (4-6시간)**
- 추론 + 평가: 1-2시간
- (선택적) cascade 적용 학습: 별도 사이클

---

## 전체 일정 (개략)

| Phase | 시작 | 끝 | 누적 |
|---|---|---|---|
| Plan + 박제 (지금) | — | 30분 | 30분 |
| B 학습 + 평가 | | 8시간 | ~8.5시간 |
| C 코드 + 학습 + 평가 | | 8시간 (코드 1, 학습 6, 평가 1) | ~16시간 |
| E 코드 + 평가 | | 6-10시간 | ~22-26시간 |

비동기 운영: B 학습 돌리는 동안 C 코드 작성, C 학습 돌리는 동안 E 코드.

---

## 산출물

- 체크포인트 3종:
  - `best_model_virchow2_full_256px_14slide_partialFT.pth` + `_byext.pth`
  - `best_model_virchow2_full_256px_14slide_LoRA.pth` + `_byext.pth`
  - (E 는 학습 없음, Stage 1 cascade 평가만)
- 로그: 각 Phase 별 epoch_log csv + 학습 로그
- 결과 비교 표: 이 doc 또는 `2026-05-27_results.md` 에 자동 추가
- 새 코드:
  - LoRA 추가된 [model.py](Gland_Seg/Code/model.py)
  - `stage1_infer.py`, `cascade_eval.py`

---

## 위험·주의

1. **partial FT 가 overfit** 할 수 있음 — class imbalance 심한 상황 + backbone 깸 → val_f1 1.0 으로 saturate 가능. 이 경우 ext F1 도 같이 봐야 진짜 일반화 신호 분리 가능
2. **LoRA target_modules** 미스매치 시 silent fail (학습은 되는데 LoRA 가 적용 안 됨) → 학습 전 trainable param 수 확인 필수
3. **Stage 1 가중치 없음** 시 E 보류, 또는 새로 Kather 100k 학습 (별도 사이클)
4. **DDP + LoRA** — PEFT 가 DDP 와 잘 동작하는지 확인. 일반적으로는 OK
