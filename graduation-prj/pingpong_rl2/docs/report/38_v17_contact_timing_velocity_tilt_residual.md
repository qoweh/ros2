# v17 contact timing, velocity, and tilt residual implementation

작성일: 2026-06-03

## 목적

`pmk_cf_self_rally_v16` 1M 분석 결과, 8D 구조는 v15보다 부드러워졌지만 안정적인 self-juggling에는 부족했다.

핵심 병목:

- `ball_out_of_bounds = 75 / 100`
- mean useful bounces `0.60`, max `3`
- upward contact의 projected apex가 target 아래인 비율 `0.754`
- `tangent`, `pitch`, `roll` action 포화
- 새 8D velocity residual은 거의 쓰이지 않음
- planner intercept time 평균이 약 `0.036s`로 너무 늦음

따라서 v17은 단순 reward weight 조정이 아니라, policy가 실제 타격 속도와 tilt primitive 강도에 더 직접 개입할 수 있게 바꿨다.

## 구현한 것

새 action mode:

```text
position_contact_frame_velocity_tilt_residual
```

새 11D action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual, racket_vz_residual, trajectory_tilt_scale, centering_tilt_scale]
```

변경 요약:

- v16 8D mode는 보존했다.
- v17 11D mode를 별도로 추가했다.
- `tracking_strike_plane_offset`을 env 설정값으로 열었다.
  - v17 preset은 `0.06m`.
  - 기존 hard-coded `0.02m`보다 높게 잡아 라켓이 위로 가속할 시간을 더 확보한다.
- `racket_vz_residual`을 `_contact_frame_velocity_target()`의 z velocity target에 직접 더한다.
  - v16의 간접 `vz_scale`만으로 actual racket velocity가 올라오지 않은 문제를 겨냥한다.
- `trajectory_tilt_scale`, `centering_tilt_scale`을 추가했다.
  - 실제 scale은 `max(0, 1 + residual)`.
  - scripted tilt primitive를 끄거나 키우는 권한을 policy에 준다.
- info/training_config/analysis CSV에 새 값들을 기록한다.
- heuristic bootstrap은 새 11D mode에서 뒤 6개 residual/scale action을 0으로 채운다.

## v17 preset

새 preset:

```text
contact_frame_self_rally_v17_candidate
```

주요 설정:

- `action_mode = position_contact_frame_velocity_tilt_residual`
- `learning_rate = 2.0e-5`
- `n_epochs = 2`
- `clip_range = 0.08`
- `log_std_init = -2.5`
- `tracking_strike_plane_offset = 0.06`
- `tilt_action_limit = 0.006`
- `controller_velocity_feedback_gain = 0.55`
- `controller_max_velocity_step = 0.085`
- `contact_frame_base_strike_time_horizon = 0.18`
- `contact_frame_strike_hold_time = 0.08`
- `contact_frame_velocity_target_max = 3.2`
- `contact_frame_racket_vz_action_limit = 0.45`
- `contact_frame_tilt_scale_action_limit = 0.75`
- `contact_frame_action_penalty_weight = 0.10`

## 검증 결과

통과:

```text
PYTHONPATH=src conda run -n mujoco_env python -m py_compile ...
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_ppo_runs tests.test_keepup_contract_features tests.test_vector_env tests.test_scene_load
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py --preset contact_frame_self_rally_v17_candidate --smoke ...
git diff --check
```

결과:

- `tests.test_keepup_env`: 106 tests OK
- 주변 테스트: 17 tests OK
- v17 preset env 생성 OK
- v17 action space shape: `(11,)`
- PPO smoke 학습/저장 OK
- whitespace check OK

스모크 평가의 `mean_useful_bounces=0.0`은 정상 판단 대상이 아니다. 1024 timestep만 돌려 학습 경로가 깨지지 않는지 확인한 값이다.

## 학습 명령

v17은 action dimension이 8D에서 11D로 바뀌었으므로 v16 checkpoint를 이어받지 않는다. 새 모델로 시작한다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v17_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v17 \
  --reset-model \
  --total-timesteps 1000000
```

1M을 먼저 권장한다. 구조가 바뀌었기 때문에 2M을 바로 밀기보다, 새 action 축이 실제로 쓰이는지 확인하는 편이 낫다.

## 학습 후 분석 명령

final model:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v17/pmk_cf_self_rally_v17_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v17_final_contact_diagnosis
```

best model:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v17/pmk_cf_self_rally_v17_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v17_best_contact_diagnosis
```

## 학습 후 볼 지표

우선 확인할 것:

- `ball_out_of_bounds`가 v16의 `75/100`보다 내려가는지
- `mean_useful_bounces`, `two_or_more_useful_bounce_rate`, `three_or_more_useful_bounce_rate`
- `upward_contact_projected_apex_below_target_rate`
- `actual_outgoing_velocity_z`와 desired z의 차이
- `racket_velocity_z` mean
- `applied_action_8_racket_vz_residual`
- `applied_action_9_trajectory_tilt_scale`
- `applied_action_10_centering_tilt_scale`
- `contact_frame_trajectory_tilt_scale`
- `contact_frame_centering_tilt_scale`
- `tracking_strike_plane_offset`

판단 기준:

- `racket_vz_residual`이 양수 방향으로 의미 있게 쓰이는데 apex가 여전히 낮으면 controller velocity limit/gain 또는 contact 물성 쪽을 다시 봐야 한다.
- `trajectory_tilt_scale`이나 `centering_tilt_scale`이 한쪽 끝에 붙으면 base tilt primitive가 과하거나 부족한 것이다.
- 새 action 축들이 거의 0에 묶이면 exploration/action penalty가 아직 보수적인 것이다.
- `ball_out_of_bounds`가 더 늘면 tilt scale limit이나 outgoing XY residual limit을 줄여야 한다.

## 다음 판단

v17은 최종답이라기보다, v16에서 보인 병목에 맞춘 더 근본적인 구조 수정이다.

원하는 최종 목표인 "탁구공을 적절한 높이와 방향으로 계속 위로 치기"에는 최소한 다음 조건이 같이 맞아야 한다.

- 접촉 전에 충분한 준비 시간이 있어야 한다.
- 라켓이 실제로 위쪽 velocity를 만들어야 한다.
- 공의 outgoing XY가 다음 접촉 가능한 영역으로 돌아와야 한다.
- tilt primitive가 lateral drift를 줄이되 과하게 공을 옆으로 밀지 않아야 한다.

v17 분석에서 위 조건 중 어느 쪽이 남는 병목인지 다시 분리한다.
