# SPIDER PDC Candidate Package — for professor review

작성: 2026-05-31
대상: 우리 CRC 슬라이드에서 SPIDER colorectal model의 "Adenocarcinoma High Grade" probability가
       높은 patch들이 실제로 poorly-differentiated 영역인지 확인 요청.

## 배경

- HistAI의 SPIDER colorectal model (Hibou-L + BERT attention head, 13-class)을 우리 슬라이드에
  적용한 결과, SPIDER의 "Adeno HG" 클래스는 거의 fire하지 않음 (max p_high ≈ 0.013).
- 그러나 그 안에서 상대적으로 p_high가 가장 높은 top-30 patch를 추출.
- 비교용으로 SPIDER 자체 학습 데이터의 confirmed HG 샘플 20장 동봉.

## 확인 부탁드리고 싶은 것

1. **REFERENCE/spider_reference_hg/** 의 20장이 SPIDER가 "HG"라고 부르는 형태학.
   교수님 기준으로도 이게 poorly-differentiated CRC에 부합하는지?
2. **candidates/** 의 top-30 후보가 위 reference와 morphology가 유사한지?
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
- top-30 p_high 범위: [0.0074, 0.0135]
- top-30 슬라이드 분포: {'S14-177-1-5': np.int64(24), 'S14-2289-1-6': np.int64(6)}
- top-30 현재 GT 라벨: {0: np.int64(23), 1: np.int64(4), -1: np.int64(3)}
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
