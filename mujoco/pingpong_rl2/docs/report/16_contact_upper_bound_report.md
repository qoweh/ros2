# contact upper bound report

## 1. what changed

이번 단계의 목적은 `scripted controller ceiling`과 `task geometry ceiling`을 분리하는 것이었다.

이를 위해 reward나 PPO는 건드리지 않고, contact 직후 outgoing velocity만 upper-bound 방식으로 덮어쓸 수 있는 diagnostic hook을 추가했다.

변경한 코드:

- `src/pingpong_rl2/envs/pingpong_sim.py`
  - `set_ball_velocity()`를 추가했다.
- `src/pingpong_rl2/envs/keepup_env.py`
  - `contact_oracle_mode`, `contact_oracle_blend` 옵션을 추가했다.
  - `contact_oracle_mode="desired_outgoing_velocity"`일 때 contact 직후 ball velocity를 `desired outgoing velocity` 방향으로 blend해서 덮어쓰도록 했다.
  - oracle이 적용된 경우 `oracle_post_contact_ball_velocity_*`를 실제 outgoing source로 사용하도록 했다.
- `scripts/run_heuristic_keepup_diagnostic.py`
  - oracle CLI 옵션을 추가했다.
  - summary에 `mean_contacts`, `max_contacts`, `time_limit_episode_rate`, `oracle_contact_rate`를 추가했다.

중요한 점은 이 oracle이 학습용 primitive가 아니라는 것이다.

이 단계의 oracle은 오직 아래 질문에 답하기 위한 diagnostic이다.

> 현재 reset/task geometry에서 contact 직후 outgoing을 충분히 좋게 만들 수만 있다면, 실제로 `3+` keep-up이 가능한가?

## 2. commands executed

baseline best current config:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_baseline_best30_v1 \
  --variant-name baseline_best30 \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02
```

oracle sweep around the same config:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend025_best30_v1 \
  --variant-name oracle_blend025_best30 \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 0.25

PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend05_best30_v1 \
  --variant-name oracle_blend05_best30 \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 0.5

PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend075_best30_v1 \
  --variant-name oracle_blend075_best30 \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 0.75

PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend1_best30_v1 \
  --variant-name oracle_blend1_best30 \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 1.0
```

100-episode confirmations:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend05_best100_v1 \
  --variant-name oracle_blend05_best100 \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 0.5

PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_upper_bound_oracle_blend075_best100_v1 \
  --variant-name oracle_blend075_best100 \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --contact-oracle-mode desired_outgoing_velocity \
  --contact-oracle-blend 0.75
```

## 3. numeric result

### 3.1 baseline still fails

`contact_upper_bound_baseline_best30_v1_summary.json`:

- `mean_useful_bounces = 0.1667`
- `max_useful_bounces = 1`
- `mean_contacts = 3.83`
- `time_limit_episode_rate = 0.0`
- `two_or_more_useful_bounce_rate = 0.0`
- `three_or_more_useful_bounce_rate = 0.0`
- `next_intercept_reachable_rate = 0.4261`
- `all_contact_mean_outgoing_velocity_error_norm = 1.6853`

즉 기존 best current scripted config는 여전히 `3+`는커녕 `2+`도 안정적으로 만들지 못했다.

### 3.2 small oracle correction is not enough

`contact_upper_bound_oracle_blend025_best30_v1_summary.json`:

- `mean_useful_bounces = 0.30`
- `max_useful_bounces = 2`
- `mean_contacts = 8.53`
- `time_limit_episode_rate = 0.0333`
- `two_or_more_useful_bounce_rate = 0.0333`
- `three_or_more_useful_bounce_rate = 0.0`
- `next_intercept_reachable_rate = 0.6992`
- `all_contact_mean_outgoing_velocity_error_norm = 1.1948`

`0.25` blend는 baseline보다 분명 좋아졌지만, 아직 `3+` gate를 열지는 못했다.

### 3.3 moderate oracle correction crosses the gate

`contact_upper_bound_oracle_blend05_best30_v1_summary.json`:

- `mean_useful_bounces = 3.2333`
- `max_useful_bounces = 7`
- `mean_contacts = 22.7`
- `time_limit_episode_rate = 1.0`
- `two_or_more_useful_bounce_rate = 0.70`
- `three_or_more_useful_bounce_rate = 0.6667`
- `next_intercept_reachable_rate = 1.0`
- `useful_contact_next_intercept_reachable_rate = 1.0`
- `all_contact_mean_outgoing_velocity_error_norm = 0.6392`

100-episode confirmation `contact_upper_bound_oracle_blend05_best100_v1_summary.json`:

- `mean_useful_bounces = 3.34`
- `max_useful_bounces = 8`
- `mean_contacts = 22.6`
- `time_limit_episode_rate = 1.0`
- `one_or_more_useful_bounce_rate = 0.78`
- `two_or_more_useful_bounce_rate = 0.66`
- `three_or_more_useful_bounce_rate = 0.60`

이건 우연이 아니다. 현재 narrow reset/task geometry에서는 `outgoing correction`만 충분하면 이미 `3+`가 가능하다.

### 3.4 stronger but not full correction is even better

`contact_upper_bound_oracle_blend075_best30_v1_summary.json`:

- `mean_useful_bounces = 4.5333`
- `max_useful_bounces = 11`
- `mean_contacts = 20.27`
- `time_limit_episode_rate = 1.0`
- `two_or_more_useful_bounce_rate = 0.80`
- `three_or_more_useful_bounce_rate = 0.7667`
- `all_contact_mean_outgoing_velocity_error_norm = 0.2597`

100-episode confirmation `contact_upper_bound_oracle_blend075_best100_v1_summary.json`:

- `mean_useful_bounces = 4.53`
- `max_useful_bounces = 11`
- `mean_contacts = 20.15`
- `time_limit_episode_rate = 1.0`
- `one_or_more_useful_bounce_rate = 0.92`
- `two_or_more_useful_bounce_rate = 0.87`
- `three_or_more_useful_bounce_rate = 0.80`
- `next_intercept_reachable_rate = 1.0`
- `useful_contact_next_intercept_reachable_rate = 1.0`
- `all_contact_mean_outgoing_velocity_error_norm = 0.2576`

즉 현재 best current scripted contact에 `75%` 정도의 outgoing correction만 더해도 repeated keep-up이 강하게 열린다.

### 3.5 full desired-velocity clamp is not the right training target

`contact_upper_bound_oracle_blend1_best30_v1_summary.json`:

- `mean_useful_bounces = 0.0`
- `mean_contacts = 19.0`
- `time_limit_episode_rate = 1.0`
- `next_intercept_reachable_rate = 1.0`
- `all_contact_mean_outgoing_velocity_error_norm = 0.0`
- `failure_counts = {"time_limit": 30}`

이 결과는 겉보기엔 이상하지만 중요한 해석 포인트가 있다.

- geometry 자체는 문제 없으므로 episode는 계속 유지된다.
- 하지만 `desired outgoing velocity`를 100% 강제로 맞추면 discrete useful bounce contract 대신 long-contact / chatter에 가까운 regime으로 들어가 current success label과 어긋난다.

즉 다음 primitive의 목표는 `perfect post-contact clamp`가 아니라 `moderate, contact-phase-aware correction`이어야 한다.

## 4. interpretation

이번 단계의 결론은 명확하다.

1. `task geometry ceiling` 가설은 기각됐다.
2. current narrow reset/task geometry에서는 좋은 outgoing만 만들 수 있으면 `3+`가 충분히 가능하다.
3. 심지어 `oracle_blend=0.5`만으로도 100-episode에서 `three_or_more_useful_bounce_rate = 0.60`이 나왔다.
4. 따라서 지금 병목은 reward가 아니라 `contact/control abstraction`이다.
5. 다만 target은 `100% desired outgoing clamp`가 아니다. `0.5~0.75` 수준의 moderated correction band가 더 실제 useful-bounce contract와 잘 맞았다.

## 5. conclusion

이번 stage-3 upper-bound 결과로 이제 다음 순서는 확정됐다.

- PPO reward tuning으로 돌아가면 안 된다.
- physics/reset geometry를 먼저 의심할 단계도 아니다.
- 다음 작업은 `contact primitive` 또는 `hybrid strike`처럼, current `position_strike`보다 훨씬 직접적으로 contact-time outgoing을 조절하는 action/controller abstraction이다.

한 줄 결론:

> current environment is keep-up-feasible, and the remaining bottleneck is controller/action abstraction, not task geometry or PPO reward design.