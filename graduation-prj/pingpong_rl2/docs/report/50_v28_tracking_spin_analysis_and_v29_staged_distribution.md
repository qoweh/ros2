# v28 tracking/spin 분석과 v29 staged distribution 수정

작성일: 2026-06-04

## 요약

`pmk_cf_self_rally_v28_racket_tracking_spin`은 v26 계열보다 훨씬 어려운 reset 분포를 한 번에 학습한 첫 모델이다.

- `reset_xy_range`: `0.075 -> 0.16m` curriculum
- `reset_velocity_xy_range`: 학습 시작부터 `0.06m/s`
- `reset_velocity_z_range`: 학습 시작부터 `(-0.16, 0.04)m/s`
- `reset_ball_angular_velocity_range`: 학습 시작부터 `20rad/s`
- action mode: 17D `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`

결론은 "나빠졌다"라기보다, 난도를 크게 올린 상태에서도 긴 episode가 일부 살아남지만 새 tracking residual 축은 거의 사용하지 못하고 있다. 따라서 v29는 reward 전체를 다시 흔드는 대신 reset 난이도와 bootstrap을 고쳤다.

## v28 평가 결과

80 episode rebound analysis 결과:

- mean useful bounces: `8.16`
- max useful bounces: `45`
- `30+ useful` episode: `7 / 80 = 8.75%`
- failure counts:
  - `ball_out_of_bounds`: `34`
  - `low_apex_contact`: `27`
  - `robot_body_contact`: `14`
  - `time_limit`: `4`
  - `ball_speed_limit`: `1`

contact 기준:

- total contacts: `2509`
- useful contact rate: `26.0%`
- mean projected apex height above racket: `0.205m`
- useful contact mean projected apex height above racket: `0.271m`
- upward contact apex `< 0.16m`: `41.8%`
- next-intercept reachable rate: `72.9%`
- useful-contact next-intercept reachable rate: `100%`
- useful projected apex XY error: `1.35cm`

해석:

- 성공 contact는 다음 공을 다시 칠 수 있게 잘 남긴다.
- 실패 contact는 여전히 vertical energy가 부족하거나 lateral drift가 누적된다.
- `time_limit` episode가 존재하므로 기본 keep-up 구조는 동작한다.
- 하지만 넓은/속도/spin 분포 전체에서 안정적으로 `30+`를 보장하는 모델은 아직 아니다.

## action 사용량

분석 스크립트가 17D 중 15D까지만 저장하던 문제가 있어 tracking action logging을 보강했다.

대표 action 사용량:

- radial residual: mean abs `0.0184 / 0.020`, 95% bound saturation `74.5%`
- pitch residual: mean abs `0.0062 / 0.008`, 95% saturation `19.7%`
- outgoing x residual: mean abs `0.118 / 0.35`
- target apex z residual: mean abs `0.0106 / 0.08`
- tracking vx residual: mean abs `0.0056 / 0.65`
- tracking vy residual: mean abs `0.0105 / 0.65`

새 tracking x/y 축은 거의 쓰이지 않는다. 즉 17D로 확장한 것 자체가 실패했다기보다는, 그 축을 쓰는 방법이 학습 초기에 충분히 주어지지 않았다.

## 원인 판단

1. v28은 reset XY만 curriculum이고, 초기 lateral velocity와 spin은 처음부터 최종 난이도로 들어갔다.
2. heuristic bootstrap이 tracking x/y residual을 항상 `0`으로 넣고 있었다.
3. action penalty가 17D 전체에 걸려 있어 새 tracking 축을 탐색할 유인이 약했다.
4. `ball_out_of_bounds`는 useful contact 이후에도 누적 drift가 생기는 episode가 많다.
5. `low_apex_contact`는 마지막 contact에서 vertical energy가 꺼지는 패턴이다. terminal contact median apex가 `0.124m`로 낮다.

## v29 수정

새 preset:

- `contact_frame_self_rally_v29_racket_tracking_staged_distribution`

새 config:

- `configs/pmk_cf_self_rally_v29_racket_tracking_staged_distribution.json`

수정 내용:

- reset distribution curriculum을 `reset_xy_range` 전용에서 전체 reset 난이도 curriculum으로 확장했다.
- v29는 아래 값을 함께 선형 증가시킨다.
  - `reset_xy_range`: `0.075 -> 0.16`
  - `reset_velocity_xy_range`: `0.025 -> 0.06`
  - `reset_velocity_z_range`: `(-0.08, 0.02) -> (-0.16, 0.04)`
  - `reset_ball_angular_velocity_range`: `0 -> 20`
- heuristic bootstrap이 tracking residual 마지막 2축에 작은 XY velocity correction을 넣도록 수정했다.
- `contact_frame_action_penalty_weight`: `0.10 -> 0.075`
- lateral 안정화:
  - `contact_frame_lateral_brake_gain`: `0.65 -> 0.80`
  - `contact_frame_lateral_brake_max`: `0.25 -> 0.32`
  - `next_intercept_xy_error_penalty_weight`: `1.35 -> 1.45`
  - `post_contact_lateral_velocity_penalty_weight`: `0.90 -> 1.00`

## 다음 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py --config-file configs/pmk_cf_self_rally_v29_racket_tracking_staged_distribution.json
```

학습 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v29_racket_tracking_staged_distribution/pmk_cf_self_rally_v29_racket_tracking_staged_distribution_model.zip \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --analysis-name pmk_cf_self_rally_v29_racket_tracking_staged_distribution_eval100
```

viewer:

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v29_racket_tracking_staged_distribution/pmk_cf_self_rally_v29_racket_tracking_staged_distribution_model.zip \
  --episodes 100 \
  --max-episode-steps 1800
```
