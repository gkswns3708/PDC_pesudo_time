# Project conventions — Gland_Seg

이 문서는 작업 관습을 모아둡니다. 새 사이클을 시작할 때 한 번 보고 기억을 맞춰주세요.

---

## 1. 설계 문서 (design docs)

**무엇**: 실험·변경 사항을 코드로 옮기기 전에 동기·가설·실행 순서·위험·결정 트리를 적어두는 .md.

**어디에**: `/app/plan/` (primary, 2026-05-27 부터). `/app/Gland_Seg/designs/` 는 legacy — 기존 파일은 그대로 두되 새 design 은 `/app/plan/` 에 작성.

**파일명 규칙**: `YYYY-MM-DD_<짧은_주제_snake_case>.md`
- 예: `2026-05-25_stage2_keep_and_patch_resolution.md`
- 같은 날 여러 건이면 주제만 다르게: `2026-05-25_class_balanced_sampler.md`

**들어가야 할 섹션 (권장)**:
1. **Context** — 왜 지금 이 변경? 이전 사이클 결과나 미해결 문제
2. **가설 / 기대 효과** — 변경이 어떤 메커니즘으로 무엇을 개선하나
3. **Step-by-step plan** — 코드 변경, 실행 명령, 예상 시간
4. **위험·주의** — 디스크, 시간, scale mismatch, 비교 일관성 등
5. **결정 트리** — 결과 시나리오별 다음 액션 (개선 / 부분 효과 / 변화 없음 / 후퇴)
6. **산출물** — 새로 생기는 파일·체크포인트·로그 경로

**왜 날짜별로**: 사이클이 길어지면 어떤 순서로 결정했는지 추적이 가장 중요해짐. 파일명 정렬만으로 타임라인이 보임.

**히스토리 누적**:
- 결과는 같은 파일에 append 하지 말고 새 파일 `YYYY-MM-DD_<topic>_results.md` 로 분리.
- baseline·이전 사이클 결과는 덮어쓰지 말고 `results/_baseline_*/` 같은 별도 폴더로 박제.

---

## 2. 코드 파일 위치

| 위치 | 용도 |
|---|---|
| `Code/` | 모든 학습·추론·시각화 Python 스크립트 |
| `Data/S14/{SVS,Annotation,Overlay}/` | 원본 슬라이드, XML annotation, GT overlay PNG |
| `patches_stainnorm/<slide>/<class>/` | 추출된 학습 patch (현재 default, 512px) |
| `patches_stainnorm_<size>/...` | 다른 patch_size 실험 시 별도 폴더로 분리 |
| `checkpoints/best_model_<backbone>_full.pth` | 학습 끝난 체크포인트 (default) |
| `checkpoints/best_model_<backbone>_full_<tag>.pth` | 실험·이전 버전 보존본 (e.g. `_8slide.pth`, `_512px_14slide.pth`) |
| `results/<slide>/` | 외부 슬라이드 추론 + 평가 산출물 |
| `results/<slide>_<tag>/` | 같은 슬라이드의 다른 setting 결과 (e.g. `_256px`) |
| `logs/` | 학습·추출·추론 로그 |
| `designs/` | 설계 문서 (위 규칙) |
| `docs/` | 영구 컨벤션·운영 문서 (이 파일) |

---

## 3. 체크포인트·결과 보존 규칙

학습 또는 추론 setting 이 바뀌면 **기존 산출물 덮어쓰기 전에 rename·copy** 로 보존:

```bash
# 예) virchow2 재학습 직전
mv checkpoints/best_model_virchow2_full.pth \
   checkpoints/best_model_virchow2_full_<이전_setting_tag>.pth
```

이름의 tag 는 setting 을 한 눈에 보이게:
- `_8slide`  : 학습 슬라이드 수
- `_512px`   : patch size
- `_14slide_512px_LP` : 복합

results/ 도 동일하게 `<slide>_<setting_tag>/` 으로 분리.

---

## 4. 외부 평가 (S14-2289-1-6) 기준

- 평가 metric 우선순위: **macro-F1 > non-gland F1 > accuracy**
  - val_f1 (10% holdout) 은 거의 항상 0.99 saturate → 외부 metric 만 의미 있음
- GT 는 [compute_gt_metrics.py](../Code/compute_gt_metrics.py) 의 parity 규칙 — polygon depth 0/1/2+ → no-GT / gland / non-gland
- 평가 patch: ROI 안 (~2,800/9,605)

새 사이클 평가표는 항상 baseline 과 함께 표시:

| Setting | n_eval | acc | f1_gland | f1_nongland | macro-F1 |
|---|---:|---:|---:|---:|---:|
| baseline (이전) | … | … | … | … | … |
| new | … | … | … | … | … |
| 변화 | — | … | … | … | … |

---

## 5. Cascade 큰 그림 기억

연구 최종 목적: **PDC (Poorly Differentiated Cancer, non-gland) 위치·count 분석** (Tumor Budding 연구).
3 stage cascade — Stage 2 단독 정확도가 100% 일 필요 없음.

| Stage | 모델 | 출력 | 비고 |
|---|---|---|---|
| 1 | ResNet-18 + Kather 100k (400x) | cancer / cancer-stroma / normal mask | 통과, 안정적 |
| 2 | virchow2 + uni2 + phikon-v2 (binary) | gland / non-gland (cancer 영역 안에서) | 현재 작업 — 거친 영역 분류 OK |
| 3 | HoverNet (200x) | cell-level instance seg | 픽셀·세포 단위 정밀도 |

Stage 2 의 경계 오류는 Stage 3 의 cell-level 결과로 자연 보정됨 — *정밀도는 stage 3 에 위임* 가정 하에 설계.

---

## 6. 카톡·메시지 산출물

교수님께 보낼 카톡 메시지는 별도 보존하지 않음 (대화 로그가 트래킹). 다만 메시지에 포함된 **링크·zip 파일 구조** 가 바뀔 때마다 design doc 에 한 줄 메모.

---

## 7. 새 사이클 시작 절차 (checklist)

1. `designs/` 최신 파일 읽기 → 이전 사이클 결정 확인
2. 새 design doc 작성 (`designs/YYYY-MM-DD_<topic>.md`) — 코드 변경 전에
3. baseline 보존 (checkpoints, results 이름 변경)
4. 코드 변경 + 실행
5. 결과 비교 표 작성 → 같은 design doc 의 "결과" 섹션 또는 별도 `_results.md`
6. 다음 사이클 후보 갱신
