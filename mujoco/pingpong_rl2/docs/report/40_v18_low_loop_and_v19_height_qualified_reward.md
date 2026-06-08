# v18 low loop review and v19 height-qualified reward

작성일: 2026-06-03

## 질문에 대한 결론

v17에서 봤던 두 문제는 v18에서 개선됐다.

1. 작은 앞 5축 action이 PPO Gaussian std 때문에 계속 clip되던 문제
2. 11D 전체 bounds 문제가 아니라 contact point/tilt residual이 끝값에 붙던 문제

v18 final model의 실제 학습 후 std:

```text
[0.00695, 0.00697, 0.01045, 0.00278, 0.00277, 0.08000, 0.07991, 0.07979, 0.07994, 0.07999, 0.07992, 0.07994, 0.08031]
```

action bound 대비 std 비율:

```text
front 5축: 약 0.347x
velocity/lateral/scale 축: 약 0.107x ~ 0.229x
```

v18 final contact CSV 기준 앞 5축 saturation은 모두 `0%`였다.

```text
radial mean -0.000729, saturation 0%
tangent mean +0.001869, saturation 0%
z mean +0.000995, saturation 0%
pitch mean -0.004700, saturation 0%
roll mean -0.001099, saturation 0%
```

즉 v17의 "앞 5축 100% saturation" 병목은 해결됐다. 현재 병목은 action bound보다 reward landscape와 low-apex local optimum에 가깝다.

## v18 분석

분석 대상:

- run: `pmk_cf_self_rally_v18`
- preset: `contact_frame_self_rally_v18_candidate`
- action mode: `position_contact_frame_velocity_tilt_lateral_residual`
- completed timesteps: `1,000,000`
- analysis: `analysis/pmk_cf_self_rally_v18_final_contact_diagnosis_summary.json`

v17 final과 비교하면 v18은 확실히 좋아졌다.

```text
v17 mean_useful_bounces: 1.20
v18 mean_useful_bounces: 2.42

v17 max_useful_bounces: 5
v18 max_useful_bounces: 8

v17 ball_out_of_bounds: 66 / 100
v18 ball_out_of_bounds: 30 / 100

v18 two_or_more_useful_bounce_rate: 0.58
v18 three_or_more_useful_bounce_rate: 0.43
```

방향 제어도 좋아졌다.

```text
v18 mean_outgoing_velocity_xy_error = 0.093
v18 mean_predicted_apex_xy_error = 0.018
v18 next_intercept_reachable_rate = 0.726
```

하지만 높이는 아직 부족하다.

```text
mean_projected_contact_apex_height_above_racket = 0.234m
median_projected_contact_apex_height_above_racket = 0.193m
upward_contact_projected_apex_below_0_20_rate = 0.505
upward_contact_projected_apex_below_target_rate = 0.725
terminal_upward_projected_apex_below_0_20_rate = 0.538
```

목표 apex가 `0.30m`인데, upward contact의 약 `72.5%`가 목표보다 낮다. viewer에서 보이는 낮은 "통통통" episode는 이 수치와 잘 맞는다.

## 낮은 통통 루프가 생기는 이유

v18은 낮은 공을 완전히 실패로만 보지 않는다. 이건 의도적으로 회복 학습 기회를 주기 위해 넣었던 설계였지만, 이제는 일부 episode에서 local optimum이 됐다.

현재 구조상 낮은 contact라도 다음 조건을 만족하면 보상이 남는다.

- lateral velocity가 작다.
- projected apex XY가 중앙 근처다.
- easy_next_ball_score가 어느 정도 나온다.
- projected apex가 target에는 못 미치지만 height_tolerance 안쪽으로 들어오면 일부 height score가 생긴다.
- `low_apex_contact_height_threshold`가 v18에서 `0.14m`라서 `0.16m ~ 0.25m`의 낮은 반복은 오래 살아남을 수 있다.

즉 정책 입장에서는 "높이 0.30m로 올리는 위험"보다 "낮지만 중앙 근처에서 오래 맞히는 안정 루프"가 더 쉬운 보상 경로가 된다.

## 차원 추가 판단

이번에는 action dimension을 더 늘리지 않았다.

이유:

- v18의 13D action 중 새 lateral velocity residual 축은 포화되지 않았다.
- 앞 5축 saturation도 v17처럼 막혀 있지 않다.
- 현재 실패는 "정책이 더 움직일 권한이 없음"보다 "낮은 루프도 보상을 받음"에 가깝다.

따라서 v19의 우선순위는 구조 확장보다 reward qualification이다.

다음에 차원 추가를 고려할 조건:

- v19에서도 `upward_contact_projected_apex_below_target_rate`가 높게 유지된다.
- `racket_vz_residual`이나 `vz_scale`이 포화되는데도 높이가 부족하다.
- useful contact의 `outgoing_velocity_z_error`가 계속 크다.

그 경우 후보는 다음이다.

- direct desired apex residual
- contact timing residual
- vertical follow-through duration residual
- separate low-apex recovery mode residual

## v19 구현

새 preset:

```text
contact_frame_self_rally_v19_anti_low_loop
```

v19는 v18과 같은 13D action mode를 유지한다. 따라서 v18 checkpoint에서 이어서 fine-tune할 수 있다.

새로 추가한 개념:

1. height-qualified lateral stability
   - `contact_lateral_stability_term`은 projected apex가 target의 일정 비율 이상일 때만 지급된다.
   - v19 기준: `contact_lateral_stability_min_apex_ratio = 0.85`

2. height-qualified stable contact
   - `stable_contact_term`도 target apex의 일정 비율 이상일 때만 지급된다.
   - v19 기준: `stable_contact_min_apex_ratio = 0.90`

3. apex potential shaping
   - 이전 contact apex보다 target apex에 가까워지면 보상.
   - 낮아지는 방향이면 penalty.
   - CSV/summary에 `contact_apex_potential_term`과 `mean_contact_apex_potential_term`으로 기록한다.

v19 주요 값:

```text
low_apex_contact_height_threshold = 0.20
low_apex_contact_grace_count = 2
contact_apex_under_target_penalty_weight = 1.15
contact_apex_progress_reward_weight = 0.45
contact_apex_recovery_progress_reward_weight = 0.25
contact_apex_potential_reward_weight = 0.35
contact_apex_potential_gamma = 0.99
contact_apex_potential_cap = 2.0
contact_lateral_stability_min_apex_ratio = 0.85
stable_contact_min_apex_ratio = 0.90
stable_cycle_reward_weight = 1.35
stable_cycle_reward_cap = 5
```

기대 효과:

- 낮은 apex에서 XY만 안정적인 contact는 더 이상 lateral stability 보상을 받지 못한다.
- `0.20m` 아래 low-apex 반복은 더 빨리 episode를 끝낸다.
- target apex 쪽으로 회복하는 접촉에는 별도 shaping이 생긴다.
- 방향 안정성은 유지하되, 낮은 통통 루프의 reward advantage를 줄인다.

## 코드 변경

수정 파일:

- `src/pingpong_rl2/envs/keepup_env.py`
  - `contact_apex_potential_*`
  - `contact_lateral_stability_min_apex_ratio`
  - `stable_contact_min_apex_ratio`
  - height-qualified stability gate
  - `contact_apex_potential_term`
- `scripts/run_ppo_learning.py`
  - v19 preset
  - CLI/env kwargs 연결
- `scripts/run_ppo_rebound_analysis.py`
  - 새 override args
  - contact CSV의 `contact_apex_potential_term`
  - summary의 `mean_contact_apex_potential_term`
- `tests/test_keepup_env.py`
  - apex potential shaping test
  - lateral stability min apex ratio test
  - stable contact min apex ratio test

## 검증

통과:

```text
PYTHONPATH=src conda run -n mujoco_env python -m py_compile src/pingpong_rl2/envs/keepup_env.py scripts/run_ppo_learning.py scripts/run_ppo_rebound_analysis.py tests/test_keepup_env.py
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_ppo_runs tests.test_keepup_contract_features
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_vector_env tests.test_scene_load
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py --preset contact_frame_self_rally_v19_anti_low_loop --run-name tmp_v19_env_check --run-version codex --reset-model --total-timesteps 64 --smoke --output-dir artifacts/tmp/tmp_v19_env_check_codex
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py --preset contact_frame_self_rally_v19_anti_low_loop --run-name tmp_v19_resume_check --run-version codex --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v18/pmk_cf_self_rally_v18_model.zip --total-timesteps 64 --smoke --output-dir artifacts/tmp/tmp_v19_resume_check_codex
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py --model-path artifacts/tmp/tmp_v19_resume_check_codex/tmp_v19_resume_check_codex_model.zip --episodes 2 --analysis-name tmp_v19_resume_check_analysis
git diff --check
```

결과:

- `tests.test_keepup_env`: `114 tests OK`
- `tests.test_ppo_runs tests.test_keepup_contract_features`: `13 tests OK`
- `tests.test_vector_env tests.test_scene_load`: `5 tests OK`
- v19 preset 생성 OK
- v18 final checkpoint에서 v19 resume smoke OK
- analysis CSV에 `contact_apex_potential_term` 컬럼 확인
- summary에 `mean_contact_apex_potential_term` 확인
- whitespace check OK

## 학습 명령

v19는 v18에서 이어서 먼저 `500k` fine-tune을 권장한다. reward landscape를 바꿨기 때문에 처음부터 2M을 바로 태우기보다, 낮은 루프가 줄어드는지 빨리 확인하는 쪽이 낫다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v19_anti_low_loop \
  --run-name pmk_cf_self_rally \
  --run-version v19 \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v18/pmk_cf_self_rally_v18_model.zip \
  --total-timesteps 500000
```

## 학습 후 분석 명령

final model:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v19/pmk_cf_self_rally_v19_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v19_final_contact_diagnosis \
  --compare-apex-targets
```

best model:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v19/pmk_cf_self_rally_v19_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v19_best_contact_diagnosis \
  --compare-apex-targets
```

## v19에서 볼 지표

우선순위:

```text
mean_useful_bounces
three_or_more_useful_bounce_rate
ball_out_of_bounds
low_apex_contact
mean_projected_contact_apex_height_above_racket
median_projected_contact_apex_height_above_racket
upward_contact_projected_apex_below_0_20_rate
upward_contact_projected_apex_below_target_rate
terminal_upward_projected_apex_below_0_20_rate
mean_contact_apex_potential_term
mean_contact_lateral_stability_term
next_intercept_reachable_rate
mean_outgoing_velocity_xy_error
mean_outgoing_velocity_z_error
```

판단 기준:

- `upward_contact_projected_apex_below_target_rate`가 v18의 `0.725`보다 내려가야 한다.
- `upward_contact_projected_apex_below_0_20_rate`가 v18의 `0.505`보다 내려가야 한다.
- `ball_out_of_bounds`가 다시 크게 늘면 height gate가 너무 강해져서 정책이 vertical recovery 중 방향을 잃는 것이다.
- `low_apex_contact`만 늘고 useful/stable이 떨어지면 threshold/grace가 너무 강한 것이다.
- 높이는 올라가는데 `mean_outgoing_velocity_z_error`가 계속 크면 다음 단계는 vertical execution/action dimension 쪽이다.

## 다음 분기

v19 결과가 좋으면:

- 500k 추가 또는 1M 연장.
- 같은 preset으로 이어서 학습한다.

v19에서 낮은 루프가 계속 남으면:

- direct apex residual 또는 follow-through duration residual 추가를 검토한다.
- height_tolerance 자체를 줄여 useful 판정을 더 좁히는 것도 후보지만, 먼저 v19의 reward gate 효과를 본다.

v19에서 out-of-bounds가 다시 커지면:

- stable/lateral gate ratio를 조금 낮춘다.
- under-target penalty를 완화한다.
- lateral residual/action penalty balance를 다시 본다.
