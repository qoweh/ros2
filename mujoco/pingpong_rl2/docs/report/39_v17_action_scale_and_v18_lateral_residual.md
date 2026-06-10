# v17 action scale review and v18 lateral residual

작성일: 2026-06-03

## 배경

`pmk_cf_self_rally_v17` 학습 후 viewer에서는 이전 모델보다 유연해졌지만, 공이 여전히 밖으로 튀는 문제가 남았다.

분석 대상:

- run: `pmk_cf_self_rally_v17`
- completed timesteps: `1,500,000`
- action mode: `position_contact_frame_velocity_tilt_residual`
- analysis:
  - `analysis/pmk_cf_self_rally_v17_final_contact_diagnosis_summary.json`
  - `analysis/pmk_cf_self_rally_v17_best_contact_diagnosis_summary.json`

final/best 분석 파일은 같은 결과였다.

## v17 성능

100 episode 기준:

- `mean_useful_bounces = 1.20`
- `max_useful_bounces = 5`
- `one_or_more_useful_bounce_rate = 0.59`
- `two_or_more_useful_bounce_rate = 0.30`
- `three_or_more_useful_bounce_rate = 0.20`
- `mean_stable_cycles = 1.20`
- `failure_counts`
  - `ball_out_of_bounds = 66`
  - `floor_contact = 13`
  - `ball_speed_limit = 10`
  - `low_apex_contact = 6`
  - `time_limit = 5`

해석:

- v16보다 확실히 좋아졌다.
- 하지만 최종 목표인 안정적인 계속 치기에는 아직 멀다.
- 남은 주 병목은 낮은 apex보다 `ball_out_of_bounds`다.

## 11D action limit

v17 action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual, racket_vz_residual, trajectory_tilt_scale, centering_tilt_scale]
```

v17 preset limit:

```text
radial/tangent: +/-0.02 m
z: +/-0.03 m
pitch/roll: +/-0.006 rad
vz_scale: +/-0.35
outgoing_x/y_residual: +/-0.35 m/s
racket_vz_residual: +/-0.45 m/s
trajectory/centering_tilt_scale residual: +/-0.75
```

contact CSV 기준 action 사용:

- `radial = +0.020`, saturation `100%`
- `tangent = -0.020`, saturation `100%`
- `z = -0.030`, saturation `100%`
- `pitch = -0.006`, saturation `100%`
- `roll = -0.006`, saturation `100%`
- `vz_scale` mean `+0.010`, saturation `0%`
- `outgoing_x_residual` mean `-0.114`, saturation `0%`
- `outgoing_y_residual` mean `-0.046`, saturation `0%`
- `racket_vz_residual` mean `+0.033`, saturation `0%`
- `trajectory_tilt_scale` residual mean `-0.010`, saturation `0%`
- `centering_tilt_scale` residual mean `-0.060`, saturation `0%`

결론:

- 11D 전체 상/하한이 좁은 문제가 아니다.
- 앞 5개 residual만 완전히 포화되어 있다.
- 새로 추가한 velocity/scale 축들은 상한에 닿지 않았다.

## 공이 밖으로 튀는 원인

terminal `ball_out_of_bounds` contact 66개 기준:

- desired outgoing x mean: `-0.295 m/s`
- actual outgoing x mean: `+0.660 m/s`
- outgoing XY error mean: `1.074`
- next intercept XY error mean: `0.316`
- projected apex x mean: `0.768`

즉 목표는 x를 안쪽으로 보내도록 잡혀 있는데, 실제 접촉은 반대 방향으로 튕긴다.

해석:

- 단순히 desired outgoing XY를 더 바꾸는 것만으로는 부족하다.
- 실제 racket lateral velocity / contact execution이 desired outgoing XY를 만들지 못하고 있다.
- tilt가 부족할 수는 있지만, tilt만 더 키우면 out-of-bounds가 악화될 위험이 있다.

## 숨은 학습 스케일 문제

v17 PPO policy의 학습 후 `log_std`를 확인했다.

```text
std ~= 0.081 for every action dimension
```

하지만 action bound는 축마다 크게 다르다.

```text
radial/tangent bound = 0.02
pitch/roll bound = 0.006
velocity/scale bound = 0.35 ~ 0.75
```

std/action_high 비율:

- radial/tangent: 약 `4.06x`
- z: 약 `2.73x`
- pitch/roll: 약 `13.5x`
- velocity residuals: 약 `0.18x ~ 0.23x`
- tilt scale: 약 `0.11x`

결론:

- 앞 5개 작은 action 축은 PPO Gaussian std가 action range보다 훨씬 컸다.
- 이 상태에서는 초기 rollout부터 clip이 많고, 학습 후에도 미세 제어가 아니라 끝값 정책으로 굳기 쉽다.
- v17의 앞 5개 action 100% saturation은 이 문제와 잘 맞는다.

따라서 v18에서는 차원을 늘리기 전에 action std 초기화를 action limit별로 맞춘다.

## v18 구현

새 action mode:

```text
position_contact_frame_velocity_tilt_lateral_residual
```

v18 action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual, racket_vz_residual, trajectory_tilt_scale, centering_tilt_scale, racket_vx_residual, racket_vy_residual]
```

v17의 앞 11개 축은 유지하고, 마지막에 direct racket lateral velocity residual 2개만 추가했다.

추가한 것:

- `contact_frame_racket_xy_action_limit`
- `_contact_frame_racket_xy_residual()`
- `_contact_frame_velocity_target()`에서 `target_velocity[:2] += racket_xy_residual`
- info/training_config/analysis CSV logging
- heuristic bootstrap 13D zero padding
- default run name mapping
- tests

v18 preset:

```text
contact_frame_self_rally_v18_candidate
```

주요 변경:

- action mode: `position_contact_frame_velocity_tilt_lateral_residual`
- action size: `13`
- `contact_frame_racket_xy_action_limit = 0.35`
- `tilt_action_limit = 0.008`
- `post_contact_lateral_velocity_penalty_weight = 0.90`
- `contact_lateral_stability_reward_weight = 0.55`

## per-axis action std 초기화

새 옵션:

```text
--scale-log-std-by-action-limit
--action-std-limit-ratio
--action-std-min
--action-std-max
```

v18 preset:

```text
scale_log_std_by_action_limit = True
action_std_limit_ratio = 0.35
action_std_min = 0.0015
action_std_max = 0.08
```

v18 preset action std:

```text
[0.007, 0.007, 0.0105, 0.0028, 0.0028, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
```

이 설정은 작은 위치/tilt residual의 과도한 clip을 줄이면서, 큰 velocity/scale 축의 탐색은 너무 작아지지 않게 한다.

## 검증

통과:

```text
PYTHONPATH=src conda run -n mujoco_env python -m py_compile ...
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env tests.test_ppo_runs
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_contract_features tests.test_vector_env tests.test_scene_load
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py --preset contact_frame_self_rally_v18_candidate --smoke ...
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py --model-path artifacts/tmp/tmp_cfvtlr_smoke_codex/tmp_cfvtlr_smoke_codex_model.zip --episodes 2 ...
git diff --check
```

결과:

- full test set: `129 tests OK`
- v18 preset env 생성 OK
- v18 action shape: `(13,)`
- scaled log std 저장 확인 OK
- PPO smoke 학습/저장 OK
- rebound analysis smoke OK
- contact CSV 새 컬럼 확인 OK
- whitespace check OK

## v18 학습 명령

v18은 action dimension이 13D로 바뀌었고 action std 초기화도 다르다. v17 checkpoint를 이어받지 않는다.

```bash
cd mujoco/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v18_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v18 \
  --reset-model \
  --total-timesteps 1000000
```

1M 먼저 권장한다. v18의 핵심은 "더 오래"가 아니라 "앞 5축 clip이 줄고 lateral residual을 실제로 쓰는지"를 확인하는 것이다.

## 학습 후 분석 명령

final model:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v18/pmk_cf_self_rally_v18_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v18_final_contact_diagnosis
```

best model:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v18/pmk_cf_self_rally_v18_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v18_best_contact_diagnosis
```

## v18에서 볼 것

우선순위:

- 앞 5개 action saturation이 v17의 `100%`에서 내려가는지
- `applied_action_11_racket_vx_residual`
- `applied_action_12_racket_vy_residual`
- terminal out-of-bounds에서 desired x와 actual x의 부호 mismatch가 줄어드는지
- `ball_out_of_bounds`가 v17의 `66/100`보다 내려가는지
- `two_or_more_useful_bounce_rate`, `three_or_more_useful_bounce_rate`
- `outgoing_velocity_xy_error`
- `next_intercept_xy_error`
- `upward_contact_projected_apex_below_target_rate`

판단:

- 앞 5개 saturation이 여전히 높으면 action mean 자체를 recentering하거나 normalized action wrapper가 필요하다.
- lateral residual이 거의 0에 묶이면 reward보다 bootstrap/action penalty가 lateral correction을 막는 것이다.
- lateral residual이 포화되는데 out-of-bounds가 남으면 contact point residual recentering 또는 racket/contact 물성 재검토가 필요하다.
- out-of-bounds는 줄고 floor/low-apex가 늘면 lateral control은 좋아졌지만 vertical execution을 다시 보강해야 한다.
