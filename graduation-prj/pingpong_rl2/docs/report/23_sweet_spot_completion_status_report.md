# Sweet Spot Completion Status Report

Date: 2026-06-01

## Bottom Line

현재 프로젝트는 아직 최종 목표인 "로봇팔에 붙인 탁구채로 탁구공을 계속 위로 올려치기"에 도달하지 않았다.

가장 큰 문제는 PPO 타임스텝 부족이나 단순 reward weight 부족이 아니다. 현재 모델은 성공한 접촉에서는 다음 낙하지점을 sweet spot 근처로 만들 수 있지만, 그 접촉이 너무 드물고 실패 접촉은 여전히 공을 멀리 보내거나 link5 충돌로 끝난다.

따라서 다음 작업은 값 sweep이 아니라 아래 두 가지를 분리해서 끝내야 한다.

1. 다음 접촉이 가능한 sweet spot 범위를 성공 기준으로 고정한다.
2. 접촉 순간 라켓의 자세/속도가 원하는 outgoing velocity를 실제로 만들도록 low-level strike primitive를 개선한다.

## Relevant Prior Work Check

사용자가 준 논문과 추가로 확인한 로봇 탁구 연구들은 지금 방향에 대해 같은 신호를 준다.

- `Achieving Human Level Competitive Robot Table Tennis`는 low-level controller/skill descriptor와 high-level skill selector를 나누는 계층형 구조를 사용한다. 즉 하나의 monolithic PPO가 모든 것을 알아서 배우게 두는 방식보다, "어떤 위치/속도/자세로 칠 것인가"를 명시한 skill이 필요하다.
- `Sample-efficient Reinforcement Learning in Robotic Table Tennis`는 hitting-time의 공 상태를 보고 racket orientation/velocity를 action으로 둔다. 이 프로젝트도 결국 접촉 순간의 라켓 속도와 기울기를 직접 맞추는 문제다.
- `Robotic Table Tennis: A Case Study into a High Speed Learning System`과 model-based feedforward 관련 연구는 공 예측, reference trajectory planning, robot tracking을 분해해서 디버깅한다. 지금 프로젝트의 실패도 "목표 궤적"과 "로봇이 그 궤적을 실제로 추적하는지"를 나눠 봐야 한다.

참고:

- https://arxiv.org/abs/2408.03906
- https://arxiv.org/abs/2011.03275
- https://arxiv.org/abs/2309.03315
- https://link.springer.com/article/10.1007/s10514-023-10140-6

## What Was Implemented

### 1. Sweet spot 기준을 별도 성공 반경으로 분리

기존 `strike_zone_xy_radius = 0.10`은 "대충 다시 칠 수 있는 넓은 범위"에 가까웠다. 사용자가 원하는 것은 라켓 중심 근처로 계속 되돌리는 것이므로 `contact_centering_radius = 0.04`를 기준으로 strict next-ball 성공을 재정의했다.

추가된 환경 옵션:

```text
next_intercept_success_radius
easy_next_ball_xy_radius
```

권장 strict 값:

```text
next_intercept_success_radius = 0.04
easy_next_ball_xy_radius      = 0.04
```

### 2. Sweet spot-only reset 범위 고정

현재 reset sampler는 원형이 아니라 정사각형이다.

```python
uniform(-reset_xy_range, reset_xy_range, size=2)
```

그래서 모든 시작점이 반지름 4cm 안에 있으려면:

```text
reset_xy_range <= 0.040 / sqrt(2) = 0.028284
```

권장 curriculum 시작값:

```text
reset_xy_range = 0.028
reset_velocity_xy_range = 0.0
reset_velocity_z_range = (-0.01, 0.01)
```

자세한 계산은 `docs/report/22_sweet_spot_start_range_and_next_work.md`에 정리했다.

### 3. 공이 올라가는 중 로봇팔이 휘적이는 문제를 끌 수 있게 함

추가 옵션:

```text
post_contact_return_predict_during_rise
```

기본값은 기존 동작 보존을 위해 `True`다. `False`로 두면 공이 상승 중일 때 예측 낙하지점을 미리 쫓지 않고 anchor로 돌아간다.

이 옵션은 사용자가 본 "공이 올라가는데 로봇팔이 계속 움직여 학습이 흔들리는 문제"를 막기 위한 안전장치다. 다만 기존 정책에 런타임으로만 적용하면 성능이 자동으로 좋아지지는 않았다.

### 4. link5/body 충돌을 위한 controller hook 추가

`RacketCartesianController`에 optional nullspace posture와 body clearance hook을 추가했다.

추가 옵션:

```text
controller_nullspace_posture_gain
controller_nullspace_posture_max_step
controller_nullspace_posture_target
controller_body_clearance_gain
controller_body_clearance_margin
controller_body_clearance_vertical_margin
controller_body_clearance_max_step
controller_body_clearance_body_names
```

중요: 이 기능은 기본 off다. 런타임에 기존 정책에 바로 얹으면 성능이 좋아지지 않았다. 이 기능은 다음 학습/primitive에서 posture 안정화 실험용으로 써야 한다.

### 5. 새 preset 추가

추가된 후보 preset:

```text
contact_frame_sweet_spot_bootstrap_candidate
contact_frame_sweet_spot_nullspace_candidate
contact_frame_planned_intercept_candidate
```

첫 번째는 strict sweet spot reset/objective로 다시 시작하는 bootstrap 후보이고, 두 번째는 no-rise-chase와 controller clearance까지 포함한 실험 후보이다.

세 번째는 사용자가 지적한 "공이 내려올 때 접촉 지점으로 빨리 이동해야 한다"를 직접 다루기 위한 후보이다. `position_contact_frame`에서 접촉 목표 위치까지 남은 시간으로 나눈 reference velocity를 만들어 controller velocity target에 더한다.

추가 옵션:

```text
contact_frame_intercept_velocity_gain
contact_frame_intercept_velocity_max
contact_frame_intercept_velocity_time_floor
```

중요: 이 옵션도 기본 off다. 기존 정책을 깨지 않기 위해 명시적으로 켠 경우에만 동작한다.

## Latest Training Status

현재 가장 최근 학습 run:

```text
run_dir    = artifacts/ppo_runs/pmk_cf_sweet_spot_bootstrap_ppo_100k
model      = artifacts/ppo_runs/pmk_cf_sweet_spot_bootstrap_ppo_100k/pmk_cf_sweet_spot_bootstrap_ppo_100k_model.zip
best_model = artifacts/ppo_runs/pmk_cf_sweet_spot_bootstrap_ppo_100k/pmk_cf_sweet_spot_bootstrap_ppo_100k_best_model.zip
requested_timesteps = 100000
completed_timesteps = 80000
```

checkpoint eval 기준으로는 60000 step이 가장 나아 보였다.

```text
checkpoint 60000:
mean_useful_bounces = 1.00
two_or_more_rate    = 0.36
robot_body_contact  = 0.36
```

하지만 별도 100 episode strict 분석에서는 일반화가 좋지 않았다.

```text
analysis = sweet_spot_bootstrap_ppo_100k_best_strict100
mean_useful_bounces = 0.62
two_or_more_rate    = 0.17
max_useful_bounces  = 6
failure_counts      = robot_body_contact 12, ball_out_of_bounds 71, floor_contact 5, ball_speed_limit 12
```

즉 이 최신 PPO는 최종 목표를 해결하지 못했다. 더 오래 돌리는 것이 답이라고 보기 어렵다.

## Key Evaluation Results

### Baseline: 기존 h020 모델을 strict sweet spot 기준으로 재평가

```text
analysis = h020_sweet_spot_strict_radius_only100
reset_xy_range = 0.028
next_intercept_success_radius = 0.04
easy_next_ball_xy_radius = 0.04

mean_useful_bounces = 0.82
two_or_more_rate    = 0.25
max_useful_bounces  = 5
failure_counts      = ball_out_of_bounds 67, robot_body_contact 18, ball_speed_limit 9, floor_contact 6

all_contact_mean_next_intercept_xy_error    = 0.1042
useful_contact_mean_next_intercept_xy_error = 0.0203
mean_outgoing_velocity_error_norm           = 0.8829
```

해석:

- 성공 접촉은 다음 낙하지점이 평균 2cm 정도라서 좋다.
- 하지만 전체 접촉 평균은 약 10cm라서 실패 접촉 대부분은 계속 멀어진다.
- 따라서 문제는 "목표가 아예 틀렸다"라기보다 "원하는 outgoing velocity를 안정적으로 만들지 못한다"에 가깝다.

### Pitch/roll만 넓힌 실험

```text
analysis = h020_strict_pitch_roll_probe100
trajectory_tilt_limit = (0.03, 0.03)
centering_tilt_limit  = (0.02, 0.03)

mean_useful_bounces = 0.73
two_or_more_rate    = 0.22
max_useful_bounces  = 6
robot_body_contact  = 6
```

해석:

- pitch/roll 자유도를 더 줘도 최종 문제가 해결되지 않았다.
- link5 충돌은 줄었지만, ball_out_of_bounds가 늘었고 반복 성공률은 baseline보다 좋아지지 않았다.
- 따라서 "pitch/roll이 없어서 안 된다"가 아니라, pitch/roll과 라켓 속도를 접촉 순간에 맞추는 low-level primitive가 약한 것이다.

### 공 상승 중 낙하지점 추적을 끈 실험

```text
analysis = h020_strict_no_rise_chase_probe100
post_contact_return_predict_during_rise = False

mean_useful_bounces = 0.61
two_or_more_rate    = 0.21
robot_body_contact  = 15
```

해석:

- 휘적임 방지 옵션은 필요하지만 기존 정책에 런타임으로 끼워 넣으면 바로 좋아지지 않는다.
- 이 옵션은 새 학습/새 primitive의 기본 안정화 옵션으로 쓰는 편이 맞다.

### no-rise-chase + pitch/roll

```text
analysis = h020_strict_no_rise_chase_pitch_probe100

mean_useful_bounces = 0.83
two_or_more_rate    = 0.25
max_useful_bounces  = 5
robot_body_contact  = 14
```

해석:

- baseline과 거의 비슷하다.
- 이 조합도 근본 해결은 아니다.

### velocity target까지 켠 실험

```text
analysis = h020_strict_pitch_roll_velocity_probe100

mean_useful_bounces = 0.46
two_or_more_rate    = 0.12
ball_speed_limit    = 14
```

해석:

- 현재 방식의 controller velocity target은 너무 거칠거나 현재 position primitive와 충돌한다.
- 그냥 속도 target을 키우는 방식은 답이 아니다.

### planned intercept reference velocity

```text
analysis = h020_strict_planned_intercept_probe100
post_contact_return_predict_during_rise = False
contact_frame_intercept_velocity_gain = 0.65
contact_frame_intercept_velocity_max = 1.2
contact_frame_intercept_velocity_time_floor = 0.08

mean_useful_bounces = 0.87
two_or_more_rate    = 0.25
max_useful_bounces  = 8
failure_counts      = robot_body_contact 16, ball_out_of_bounds 63, floor_contact 8, ball_speed_limit 13

all_contact_mean_next_intercept_xy_error    = 0.1049
useful_contact_mean_next_intercept_xy_error = 0.0225
mean_outgoing_velocity_error_norm           = 0.8676
```

해석:

- 사용자가 지적한 "접촉 시각까지 라켓이 도착해야 한다"는 방향은 맞다.
- max useful bounce가 8까지 올라간 것은 긍정적이다.
- 하지만 two-or-more rate는 baseline strict와 같은 0.25이고, 전체 next-intercept error는 여전히 약 10cm다.
- 따라서 이 기능은 다음 primitive에 넣을 재료이지, 단독 해결책이 아니다.

heuristic zero-policy에 같은 reference velocity만 넣은 30 episode 진단은 `mean_useful_bounces=0.0`이었다. 즉 scripted policy 자체도 다시 설계해야 한다.

## User-Observed Problems

### 1. "공이 맞을 것 같으면 로봇팔이 후다닥 내려오는 위치로 이동해야 한다"

맞는 지적이다. 다만 단순히 controller max step을 키우면 타격 안정성이 깨질 수 있다.

현재 필요한 것은 step 크기 증가가 아니라:

```text
ball prediction -> planned contact pose/time -> smooth reference trajectory -> controller tracking
```

즉 매 step 목표점을 튀게 바꾸지 않고, 예상 접촉 시각까지 도달하는 reference trajectory를 만들어야 한다.

이번 작업에서 그 첫 hook으로 `contact_frame_intercept_velocity_*` 옵션을 추가했다. 다만 평가 결과 단독으로는 부족하다. 다음 작업은 이 값을 sweep하는 것이 아니라, planned contact pose/time/velocity를 하나의 primitive로 묶어서 heuristic/BC가 그 primitive를 재현하게 만드는 것이다.

### 2. "공이 올라가는 중 로봇팔이 휘적거린다"

이 문제를 막기 위해 `post_contact_return_predict_during_rise=False`를 추가했다. 다만 기존 정책에 런타임으로 적용하는 것만으로는 성능이 좋아지지 않았다.

새 primitive에서는 상승 중에는 anchor/recovery posture로 돌아가고, 공이 apex 이후 내려오기 시작할 때 다음 접촉 trajectory를 계획하는 구조가 맞다.

### 3. "공이 로봇팔과 가까워질 때 관절을 더 굽혀서 칠 수 있지 않나"

가능성이 있다. 하지만 end-effector 목표 위치만 바꿔서는 해결되지 않는다.

필요한 것은 nullspace posture다.

```text
라켓 pose는 유지
남는 자유도로 link5/link6가 공 경로에서 비켜나게 함
```

현재 hook은 추가되어 있지만, 런타임 적용만으로는 성능 향상이 없었다. 다음 단계에서는 이 posture objective를 포함해 새 primitive를 다시 bootstrap해야 한다.

### 4. "탁구공을 연속해서 칠 수 있는 위치로 떨어지게 해야 한다"

이것이 핵심이다. 현재 strict metric으로 보면 해결되지 않았다.

성공한 접촉은 가능하다.

```text
useful_contact_mean_next_intercept_xy_error ~= 0.02 m
```

하지만 전체 접촉은 실패 쪽이 많다.

```text
all_contact_mean_next_intercept_xy_error ~= 0.10 m
```

따라서 목표는 reward 숫자를 더 키우는 것이 아니라, 실패 접촉에서도 desired outgoing velocity와 실제 outgoing velocity 차이를 줄이는 것이다.

## What Not To Do Next

아래 작업은 중단한다.

- `target_ball_height`, `strike_z_boost`, `reward_weight`를 소폭 바꿔가며 PPO 재실행
- 기존 h020 정책에 런타임 옵션만 계속 덧씌우기
- pitch/roll limit만 넓히기
- mean reward만 보고 좋아졌다고 판단하기

이 작업들은 이미 병목을 해결하지 못했다.

## Correct Next Work

### Step 1. Low-level strike primitive를 새로 잡는다

접촉 직전 상태에서 아래 값을 명시적으로 계산해야 한다.

```text
predicted_contact_position
predicted_contact_time
desired_next_intercept_xy = keepup_target_xy
desired_apex_z = racket_anchor_z + target_ball_height
desired_outgoing_velocity
required_racket_face_normal
required_racket_velocity_at_contact
```

현재 일부 계산은 이미 있다.

```text
_desired_outgoing_velocity()
_contact_frame_trajectory_tilt()
_required_contact_frame_racket_velocity()
```

하지만 실제 controller가 접촉 순간 그 속도/자세를 만들지 못하고 있다. 다음 작업은 이 계산을 reward feature로만 두지 말고 reference trajectory 생성으로 바꾸는 것이다.

### Step 2. Contact-time reference trajectory를 만든다

현재는 매 step target position을 바꾸는 방식에 가깝다. 대신 예측 접촉 시각까지 도달해야 하는 목표를 만들고, 현재 라켓 위치/속도에서 부드럽게 연결해야 한다.

구현 방향:

```text
src/pingpong_rl2/envs/keepup_env.py

새 함수 후보:
_planned_contact_frame_target()
_planned_contact_reference_velocity()

역할:
- 공이 상승 중이면 anchor/recovery posture 유지
- 공이 내려오기 시작하고 predicted_intercept_time이 유효하면 접촉 시각까지 이동
- 접촉 직전에는 required_racket_velocity_at_contact를 맞추도록 target_velocity를 부드럽게 ramp
- 접촉 직후에는 followthrough 후 anchor로 복귀
```

단, velocity target을 바로 크게 넣으면 ball_speed_limit이 늘었다. ramp와 clipping이 필요하다.

현재 구현된 `contact_frame_intercept_velocity_*`는 이 중 "접촉 위치까지 도착하는 reference velocity"만 담당한다. 아직 다음 두 부분이 남아 있다.

```text
1. planned contact pose/time을 episode phase state로 유지해서 target이 매 step 튀지 않게 하기
2. impact normal velocity와 intercept velocity를 접촉 직전 ramp로 합성하기
```

### Step 3. 다음 낙하지점 strict success를 학습 기준으로 유지한다

계속 아래 기준을 사용한다.

```text
reset_xy_range = 0.028
next_intercept_success_radius = 0.04
easy_next_ball_xy_radius = 0.04
require_reachable_next_intercept_for_success = True
min_easy_next_ball_score_for_success = 0.0
```

이 기준을 풀면 모델이 좋아진 것처럼 보일 수 있지만 최종 목표와 멀어진다.

### Step 4. Body/link5 회피는 end-effector 목표가 아니라 nullspace에서 처리한다

다음 기준으로 평가한다.

```text
robot_body_contact_counts.link5
contact_xy_alignment_error
next_intercept_xy_error
outgoing_velocity_error_norm
```

link5가 줄어도 next-intercept가 나빠지면 실패다.

### Step 5. PPO 전에 deterministic/heuristic diagnostic으로 통과시킨다

PPO는 primitive가 이미 어느 정도 되는 상태에서 residual 보정으로 써야 한다.

minimum gate:

```text
episodes = 100
mean_useful_bounces >= 2.0
two_or_more_rate >= 0.60
three_or_more_rate >= 0.30
robot_body_contact_rate <= 0.15
useful_contact_mean_next_intercept_xy_error <= 0.03
all_contact_mean_next_intercept_xy_error <= 0.06
```

이 gate를 못 넘으면 PPO를 돌리지 않는다.

## Recommended Commands

현재 best 계열을 strict 기준으로 확인:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_recovery_roll_retract_bootstrap_h020_zero_eval/pmk_cf_recovery_roll_retract_bootstrap_h020_zero_eval_model.zip \
  --episodes 100 \
  --analysis-name h020_sweet_spot_strict_radius_only100 \
  --reset-xy-range 0.028 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --next-intercept-success-radius 0.04 \
  --easy-next-ball-xy-radius 0.04
```

새 strict bootstrap PPO 확인:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_sweet_spot_bootstrap_ppo_100k/pmk_cf_sweet_spot_bootstrap_ppo_100k_best_model.zip \
  --episodes 100 \
  --analysis-name sweet_spot_bootstrap_ppo_100k_best_strict100 \
  --reset-xy-range 0.028 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --next-intercept-success-radius 0.04 \
  --easy-next-ball-xy-radius 0.04
```

다음 primitive 작업 후 통과시켜야 할 평가:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name planned_contact_frame_primitive \
  --variant-name strict_sweet_spot_gate \
  --action-mode position_contact_frame \
  --episodes 100 \
  --reset-xy-range 0.028 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --require-reachable-next-intercept-for-success \
  --min-easy-next-ball-score-for-success 0.0 \
  --print-episodes
```

## Final Judgment

사용자의 직감인 "탁구공을 치기 쉬운 쪽으로 올려쳐야 한다"는 맞다. 다만 해결책은 단순히 pitch/roll 값을 조금 키우거나 reward를 더 주는 것이 아니다.

현재 코드가 배운 것은 "가끔 중앙으로 잘 보내는 접촉"이지, "항상 다음 접촉을 계획해서 보내는 타격 skill"이 아니다.

프로젝트를 완성하려면 다음 에이전트는 `planned contact-time strike primitive`를 만들어야 한다. 그 primitive가 strict sweet spot gate를 통과한 뒤에야 PPO residual 학습을 이어가는 것이 맞다.
