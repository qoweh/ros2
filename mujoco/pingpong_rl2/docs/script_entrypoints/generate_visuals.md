# generate_visuals.py

## 한 줄 역할

`docs/rl_presentation_pack/scripts/generate_visuals.py`는 저장된 training summary와 analysis CSV를 읽어 발표용 PNG 차트와 CSV 테이블을 만드는 스크립트다.

## 대표 실행 형태

```bash
conda run -n mujoco_env python docs/rl_presentation_pack/scripts/generate_visuals.py
```

## 코드 흐름

1. `metric_rows()`가 여러 run의 training summary JSON을 읽어 version별 지표 행을 만든다.
2. `long_eval_rows()`가 long-horizon evaluation episode CSV를 읽는다.
3. `action_usage_rows()`가 contact CSV에서 action dimension별 평균 사용량과 saturation rate를 계산한다.
4. `ablation_rows()`는 action ablation 결과를 하드코딩된 표로 제공한다.
5. `write_csv()`가 발표용 data CSV를 쓴다.
6. plot 함수들이 PNG를 만든다.
   - version timeline
   - failure mode
   - long horizon target hit
   - apex height distribution
   - action usage
   - action ablation
   - observation/action diagram
   - monitor reward/length curve

## 주요 호출 관계

```text
generate_visuals.py
  -> artifacts/ppo_runs/*_training_summary.json
  -> artifacts/ppo_runs/*/analysis/*.csv
  -> matplotlib
  -> docs/rl_presentation_pack/assets/*.png
  -> docs/rl_presentation_pack/data/*.csv
```

## 현재 주의점

현재 source 목록은 v25부터 v35까지 중심으로 하드코딩되어 있다. 최종 발표 기준이 `keep1_v39_17d_mid_curriculum_fixed`라면 이 스크립트의 `sources`, long eval CSV 경로, action usage CSV 경로를 v39 산출물까지 포함하도록 업데이트해야 한다.

## 발표 때 설명 포인트

- 이미 만들 수 있는 시각화 후보가 들어 있다.
- v39 최종 run에 맞게 경로만 갱신하면 학습 곡선, 성공률 추이, 실패 모드, action usage, ablation 그래프를 재생성할 수 있다.
