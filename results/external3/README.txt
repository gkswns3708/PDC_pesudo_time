external3 — AI 예측 결과 (annotation 없는 외부 슬라이드 3장)

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
