# pingpong_rl2 reward/policy cleanup plan

## 1. 목적

지금 필요한 것은 reward를 더 붙이는 것이 아니라, 기본 학습 경로를 하나로 고정하고 실험 해석이 가능하도록 만드는 것이다.

최종 목표는 아래 한 문장으로 고정한다.

> 공을 위로 올려친 뒤, 다시 칠 수 있는 strike zone으로 돌아오게 만든다.

이 문장에 직접 답하지 못하는 옵션 추가나 reward 추가는 기본 경로에 넣지 않는다.

## 2. 지금까지 뒤섞였던 부분

- `position`, `position_strike`, `position_tilt`가 run마다 섞였다.
- `strike_tilt_ramp_pitch`, `strike_tilt_assist_limit`, `include_velocity_domain_observation`, `outgoing_x_term`이 동시에 바뀐 run이 있었다.
- `pmk_tnpv_v2`처럼 이름과 달리 실제 config가 단순 `position` baseline인 run이 있었다.
- projected apex metric은 추가됐지만, target XY 정의가 아직 strike-zone-return 목표와 완전히 일치하지 않는다.

## 3. 기본 학습 경로

### 3.1 기본 baseline

- 기본 baseline action mode: `position_strike`
- `position`은 비교용 baseline으로만 유지한다.
- `position_tilt`는 당분간 실험용으로만 남긴다.

이유:

- 현재 bottleneck은 `공 아래로 들어가기 + 올라치기 timing`이다.
- 이 문제는 `position_strike`가 가장 직접적으로 다룬다.
- `position_tilt`는 아직 chatter/해석 불가 경향이 더 크다.

### 3.2 기본으로 고정할 학습 옵션

- `n_envs=4`
- `n_steps=256`
- `batch_size=256`
- `learning_rate=3e-4`
- `gamma=0.99`
- `ball_height=0.5`
- `max_episode_steps=600`
- `reset_xy_range=0.06`
- `reset_velocity_xy_range=0.01`
- `reset_velocity_z_range=(-0.02, 0.01)`
- `success_velocity_threshold=0.5`
- `tracking_during_contact_scale=0.0`

실험에서 매번 직접 바꿀 필수 옵션은 아래만 남긴다.

- `--run-name`
- `--run-version`
- `--reset-model` 또는 resume 여부
- `--total-timesteps`
- `--seed`

## 4. reward 정리

### 4.1 유지

- `tracking_term`
  - descending strike window에서 공 아래로 들어가도록 만드는 기본 dense signal
- `contact_bonus`
  - useful upward contact event를 강화
- `apex_match_term`
  - 위로 친 공의 품질을 높이 기반으로 보정
- `failure_penalty`
  - floor/body/out-of-bounds 종료 억제

### 4.2 기본값 0 유지

- `outgoing_x_term`
  - 평균 `vx`를 줄일 수는 있지만, 공이 실제로 다음 strike zone으로 돌아오는지는 보장하지 못했다.
  - baseline에서는 끄고, metric으로 strike-zone-return을 먼저 정의한 뒤에만 재검토한다.

### 4.3 실험용으로만 유지

- `tilt_angle_penalty`
- `tilt_action_delta_penalty`

이 둘은 `position_tilt` 실험에서만 의미가 있다.

## 5. CLI preset 정리

`run_ppo_learning.py`는 이제 아래 preset을 지원한다.

- `baseline_position`
  - `action_mode=position`
- `strike_position`
  - `action_mode=position_strike`
- `strike_velocity_obs`
  - `action_mode=position_strike`
  - `include_velocity_domain_observation=True`
- `tilt_experiment`
  - `action_mode=position_tilt`
  - `tilt_profile=auto`
- `final_candidate`
  - `action_mode=position_strike`
  - `strike_tilt_ramp_pitch=-0.03`
  - `strike_tilt_ramp_xy_tolerance=0.04`
  - `include_velocity_domain_observation=True`

권장 사용 방식:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_position \
  --run-name clean_strike \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

preset과 충돌하는 수동 옵션을 같이 넘기면 에러로 막아서 실험 confound를 줄인다.

## 6. projected apex metric 정리

`run_ppo_rebound_analysis.py`는 이제 아래를 지원한다.

- `--apex-target`
  - `controller_anchor`
  - `racket_home`
  - `racket_position`
  - `target_position`
- `--compare-apex-targets`
  - 모든 target 후보에 대한 `projected_apex_xy_error` 평균을 summary에 함께 기록

현재 해석 원칙:

- projected apex metric은 계속 metric으로만 둔다.
- 어떤 target이 useful contact / out-of-bounds / second useful contact와 가장 잘 맞는지 먼저 확인한다.
- 그 다음에만 `apex_xy_term` reward 승격을 검토한다.

## 7. 다음 clean ablation

다음 비교는 reward, reset, PPO hyperparameter, seed를 모두 고정하고 아래 네 개만 비교한다.

### 7.1 position baseline

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset baseline_position \
  --run-name clean_pos \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

### 7.2 strike baseline

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_position \
  --run-name clean_strike \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

### 7.3 timed negative pitch

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --run-name clean_tnp \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7 \
  --action-mode position_strike \
  --strike-tilt-ramp-pitch -0.03 \
  --strike-tilt-ramp-xy-tolerance 0.04
```

### 7.4 velocity-domain observation

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_velocity_obs \
  --run-name clean_vobs \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

각 run 뒤에는 반드시 아래 분석을 붙인다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name clean_strike \
  --run-version v1 \
  --episodes 50 \
  --compare-apex-targets
```

핵심 지표:

- `mean_useful_bounces`
- `max_useful_bounces`
- `failure_counts`
- `ball_out_of_bounds` 비율
- useful contact rate
- first/useful contact 이후 `ball_velocity_x/y/z`
- `projected_apex_xy_error`
- 두 번째 useful contact 발생 비율

## 8. 아직 1M을 돌리면 안 되는 이유

- `pmk_tnpv_v2`는 1M run이지만 구조 선택 문제에 답을 주지 못했다.
- `pmk_tnpv_v1` resume도 성능을 개선하지 못했다.
- reward를 더 붙이기 전에, 어떤 control/observation 조합이 실제로 반복 keep-up에 기여하는지 아직 clean하게 분리되지 않았다.

1M을 돌려도 되는 조건은 아래 둘을 동시에 만족할 때다.

- 100k 또는 300k clean ablation에서 한 후보가 baseline보다 `mean_useful_bounces`와 useful contact rate를 올린다.
- 같은 후보가 `ball_out_of_bounds`와 outward rebound 경향을 악화시키지 않는다.

## 9. 현재 결론

한 줄로 정리하면 아래다.

> `position_strike`를 기준 경로로 고정하고, `outgoing_x_term`은 접고, strike-zone-return metric을 먼저 제대로 만든 뒤 clean ablation으로 무엇이 실제로 keep-up을 개선하는지 다시 확인해야 한다.