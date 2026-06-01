# non-tilt contact residual report

## 1. what changed

이번 단계의 목적은 tilt-only primitive 이후 실제로 어떤 non-tilt residual이 useful chain을 여는지 확인하는 것이었다.

이 단계에서 한 일은 세 가지다.

1. 기존 `position_strike(_tilt)` action의 앞 `3`차원(position residual)을 scripted diagnostic에서 phase별로 직접 줄 수 있게 했다.
2. 그 상태에서 `strike z` / `recovery z` / `strike x` residual sweep을 돌렸다.
3. follow-up chain 전용 semantics를 한 축 더 주는 새 mode `action_mode="position_strike_tilt_lift"`와 PPO preset `contact_lift_candidate`를 추가하고, explicit `followup lift residual`이 실제로 도움이 되는지 확인했다.

추가/변경 코드:

- `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - fixed / phase-specific position residual support 추가
  - fixed / phase-specific followup-lift residual support 추가
- `scripts/run_heuristic_keepup_diagnostic.py`
  - `--fixed-position-residual`, `--strike-position-residual`, `--recovery-position-residual` 추가
  - `--fixed-followup-lift-residual`, `--strike-followup-lift-residual`, `--recovery-followup-lift-residual` 추가
  - 새 `--action-mode position_strike_tilt_lift` 지원
- `src/pingpong_rl2/envs/keepup_env.py`
  - 새 `action_mode="position_strike_tilt_lift"` 추가
  - `followup_lift_action_limit` 추가
  - 새 mode에서 마지막 action dim을 explicit follow-up lift residual로 사용
- `scripts/run_ppo_learning.py`
  - 새 preset `contact_lift_candidate` 추가
  - bootstrap allowlist에 새 mode 추가
- `src/pingpong_rl2/defaults.py`, `src/pingpong_rl2/utils/ppo_runs.py`
  - 새 mode용 run-name alias 추가

## 2. commands executed

best local non-tilt grid around the current best tilt candidate:

```bash
for strike_z in 0.00 0.01 0.02 0.03; do
  for recovery_z in 0.00 0.01 0.02; do
    PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
      --action-mode position_strike_tilt \
      --episodes 30 \
      --strike-position-residual 0.0 0.0 ${strike_z} \
      --recovery-position-residual 0.0 0.0 ${recovery_z} \
      --strike-tilt-residual 0.0 -0.02 \
      --strike-z-boost 0.024 \
      --strike-tilt-ramp-pitch -0.06 \
      --followup-strike-target-tilt -0.06 0.0 \
      --followup-strike-lift-boost 0.02 \
      --post-contact-return-assist-weight 0.5 \
      --post-contact-return-max-intercept-time 0.6
  done
done
```

100-episode confirmation:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_primitive_best100_v2 \
  --variant-name strike_z_001_roll_m02 \
  --action-mode position_strike_tilt \
  --episodes 100 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual 0.0 -0.02 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --post-contact-return-assist-weight 0.5 \
  --post-contact-return-max-intercept-time 0.6
```

follow-up offset combination check:

```bash
for offset in 0.00 0.25 0.50 0.75; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --action-mode position_strike_tilt \
    --episodes 50 \
    --strike-position-residual 0.0 0.0 0.01 \
    --strike-tilt-residual 0.0 -0.02 \
    --followup-strike-contact-offset-ratio ${offset} \
    --followup-strike-contact-offset-max 0.06 \
    --followup-strike-lift-boost 0.02 \
    --post-contact-return-assist-weight 0.5 \
    --post-contact-return-max-intercept-time 0.6
done
```

new lift mode smoke and preset resolve:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --action-mode position_strike_tilt_lift \
  --episodes 2 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual 0.0 -0.02 \
  --recovery-followup-lift-residual 0.01 \
  --followup-strike-lift-boost 0.02 \
  --followup-lift-action-limit 0.02

PYTHONPATH=src conda run -n mujoco_env python -c "... --preset contact_lift_candidate ..."
```

new lift residual sweep:

```bash
for lift in -0.01 0.00 0.01 0.02; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --action-mode position_strike_tilt_lift \
    --episodes 30 \
    --strike-position-residual 0.0 0.0 0.01 \
    --strike-tilt-residual 0.0 -0.02 \
    --strike-followup-lift-residual ${lift} \
    --followup-strike-lift-boost 0.02 \
    --followup-lift-action-limit 0.02
done
```

strike x sweep on top of the best z/roll candidate:

```bash
for strike_x in -0.02 -0.01 0.00 0.01 0.02; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --strike-position-residual ${strike_x} 0.0 0.01 \
    --strike-tilt-residual 0.0 -0.02 \
    --followup-strike-lift-boost 0.02
done
```

## 3. numeric result

### 3.1 phase-aware non-tilt residuals finally open `2+`

best 30-episode candidate from the z-grid was:

- `strike_position_residual = (0.0, 0.0, +0.01)`
- `recovery_position_residual = (0.0, 0.0, 0.0)`
- `strike_tilt_residual = (0.0, -0.02)`

`contact_primitive_zgrid_s0p01_r0p00_v1_summary.json`:

- `mean_useful_bounces = 0.6667`
- `max_useful_bounces = 2`
- `one_or_more_useful_bounce_rate = 0.50`
- `two_or_more_useful_bounce_rate = 0.1667`
- `three_or_more_useful_bounce_rate = 0.0`
- `useful_contact_next_intercept_reachable_rate = 0.30`

이건 이전 tilt-only sweep과 중요한 차이가 있다.

- 이전 best는 `max_useful_bounces = 1`
- 이제는 scripted candidate가 `2+`를 실제로 연다

즉 현재 action abstraction 안에서도 `phase-aware z residual`은 real signal이다.

### 3.2 but the 100-episode confirmation still stops at `2`

`contact_primitive_best100_v2_summary.json`:

- `mean_useful_bounces = 0.58`
- `max_useful_bounces = 2`
- `one_or_more_useful_bounce_rate = 0.47`
- `two_or_more_useful_bounce_rate = 0.11`
- `three_or_more_useful_bounce_rate = 0.0`
- `useful_contact_next_intercept_reachable_rate = 0.1897`

결론:

- noise는 아니고 `2+` signal은 유지된다
- 하지만 stage gate인 `3+`는 아직 전혀 열리지 않는다

### 3.3 follow-up offset still does not break the ceiling

best offset combination was still effectively `offset = 0.0`.

50-episode results:

- `offset=0.00`: `mean_useful_bounces = 0.60`, `max_useful_bounces = 2`, `two_plus_rate = 0.16`
- `offset=0.25`: `0.58`, `2`, `0.14`
- `offset=0.50`: `0.58`, `2`, `0.14`
- `offset=0.75`: `0.58`, `2`, `0.14`

즉 이전과 마찬가지로 follow-up offset은 근본 병목을 풀지 못했다.

### 3.4 the new lift mode scaffold works, but lift residual shows no gain yet

`contact_lift_candidate` preset resolve result:

- `action_mode = position_strike_tilt_lift`
- `action_size = 6`
- `observation_size = 52`
- `followup_lift_action_limit = 0.02`

diagnostic smoke also passed.

하지만 첫 sweep에서는 `strike_followup_lift_residual in {-0.01, 0.0, 0.01, 0.02}` 전 구간이 사실상 같은 결과를 보였다.

30-episode summary line for all tested lift values:

- `mean_useful_bounces = 0.667`
- `two_or_more_useful_bounce_rate = 0.167`

해석:

- scaffold는 정상 동작한다
- 하지만 current best candidate 주변에서는 explicit follow-up lift residual이 아직 decisive axis로 보이지 않는다

### 3.5 strike x residual also does not beat the z-only candidate

30-episode strike-x sweep:

- `strike_x = 0.00`: `mean_useful_bounces = 0.667`, `two_plus_rate = 0.167`
- `strike_x = +0.01`: `0.300`, `0.133`
- `strike_x = -0.01`: `0.233`, `0.067`
- `strike_x = +0.02`: `0.200`, `0.033`
- `strike_x = -0.02`: `0.200`, `0.000`

즉 current narrow reset에서는 strike x를 상수 residual로 미는 것 역시 best candidate를 이기지 못했다.

## 4. interpretation

이번 단계의 핵심 업데이트는 두 가지다.

1. tilt-only에서 막혔던 ceiling은 `phase-aware strike z residual`로 `2+`까지는 열린다.
2. 하지만 `3+`는 여전히 막혀 있고, explicit follow-up lift나 constant strike x로는 아직 그 ceiling을 넘지 못한다.

따라서 다음 branch는 아래 중 하나여야 한다.

- constant residual이 아니라 state-dependent `strike x/y contact-point residual`
- `impact timing / z pulse`를 더 직접 표현하는 residual
- 또는 follow-up chain을 겨냥한 더 직접적인 phase primitive

현재까지의 evidence로는 `followup lift residual`보다 `contact-point / impact-time residual` 쪽이 더 유력하다.

## 5. conclusion

한 줄 결론:

> the next branch is no longer “tilt vs no tilt”; it is now about pushing the existing phase-aware z signal beyond the `max=2` ceiling with a more direct contact-point or impact-time primitive.