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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 학습이나 평가를 새로 실행하지 않는다. 이미 저장된 training summary JSON, rebound analysis CSV, monitor CSV를 읽어서 발표용 표와 PNG 그래프를 다시 만든다.

```text
generate_visuals.py
  -> metric_rows()
       training_summary.json 여러 개 읽기
       version별 mean/max useful, failure count 정리
  -> long_eval_rows()
       long-horizon episodes.csv 읽기
       contact/useful/failure 통계 정리
  -> action_usage_rows()
       contacts.csv에서 applied_action_N 컬럼 읽기
       action limit 대비 평균 사용량 계산
  -> ablation_rows()
       수동으로 정리된 ablation 표 반환
  -> write_csv()
       발표용 data/*.csv 저장
  -> plot_*()
       assets/*.png 저장
```

각 plot 함수는 독립적으로 하나의 그림을 만든다. `plot_timeline()`은 모델 버전별 성능 변화를, `plot_failure_modes()`는 episode 종료 이유 분포를, `plot_long_targets()`는 long-horizon 목표 달성 episode 수를, `plot_apex_distribution()`은 contact 이후 예상 apex 높이 분포를 그린다.

`plot_action_usage()`는 contact CSV의 `applied_action_0...16` 컬럼을 읽어 action limit 대비 평균 절대 사용량을 계산한다. 이 값은 "policy가 어떤 축을 크게 쓰는가"를 보여주지만, "그 축이 반드시 중요한가"를 단독으로 증명하지는 않는다. 그래서 `plot_ablation()`의 ablation 결과와 같이 봐야 한다.

## 현재 코드에서 주의할 점

이 스크립트의 source path들은 대부분 v25부터 v35까지의 과거 발표 pack 기준으로 하드코딩되어 있다. 최종 발표 기준 모델을 v39로 통일하려면 `metric_rows()`, `long_eval_rows()`, `action_usage_rows()`, `plot_apex_distribution()`, `plot_monitor_curves()`의 입력 경로를 v39 산출물로 바꿔야 한다.

즉 이 파일은 "항상 최신 모델을 자동으로 찾는 도구"가 아니라, 발표 pack에 넣을 비교 대상을 명시적으로 지정해 그래프를 재생성하는 도구다.

## 현재 주의점

현재 source 목록은 v25부터 v35까지 중심으로 하드코딩되어 있다. 최종 발표 기준이 `keep1_v39_17d_mid_curriculum_fixed`라면 이 스크립트의 `sources`, long eval CSV 경로, action usage CSV 경로를 v39 산출물까지 포함하도록 업데이트해야 한다.

## 발표 때 설명 포인트

- 이미 만들 수 있는 시각화 후보가 들어 있다.
- v39 최종 run에 맞게 경로만 갱신하면 학습 곡선, 성공률 추이, 실패 모드, action usage, ablation 그래프를 재생성할 수 있다.
