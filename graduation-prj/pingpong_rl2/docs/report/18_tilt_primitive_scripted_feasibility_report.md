# tilt primitive scripted feasibility report

## 1. what changed

이번 단계의 목표는 새 `action_mode="position_strike_tilt"`가 실제 scripted control surface로서 의미가 있는지 확인하는 것이었다.

upper-bound report `16_contact_upper_bound_report.md`는 이미 geometry가 ceiling이 아니라는 점을 보여줬다.

그래서 이번 단계에서는 oracle 없이 아래 두 가지를 확인했다.

1. constant tilt residual만으로 contact chain이 열리는가?
2. strike phase에만 residual을 주면 더 직접적인 improvement가 생기는가?

추가한 코드:

- `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - fixed tilt residual support 추가
  - strike/recovery phase별 tilt residual support 추가
- `scripts/run_heuristic_keepup_diagnostic.py`
  - `--fixed-tilt-residual` 추가
  - `--strike-tilt-residual`, `--recovery-tilt-residual` 추가
- `scripts/run_ppo_learning.py`
  - heuristic bootstrap이 `action_mode="position_strike_tilt"`도 허용하도록 수정

## 2. commands executed

constant pitch residual sweep:

```bash
for pitch in 0.00 0.01 0.02 0.03 0.04 0.05; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_primitive_pitch_${pitch} \
    --variant-name pitch_${pitch}_roll_0.00 \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --fixed-tilt-residual ${pitch} 0.0 \
    --reset-xy-range 0.0 \
    --reset-velocity-xy-range 0.0 \
    --reset-velocity-z-range -0.01 0.01 \
    --strike-z-boost 0.024 \
    --strike-tilt-ramp-pitch -0.06 \
    --followup-strike-target-tilt -0.06 0.0 \
    --followup-strike-lift-boost 0.02 \
    --post-contact-return-assist-weight 0.5 \
    --post-contact-return-max-intercept-time 0.6
done
```

constant roll residual sweep:

```bash
for roll in -0.03 -0.02 -0.01 0.00 0.01 0.02 0.03; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_primitive_roll_${roll} \
    --variant-name pitch_0.00_roll_${roll} \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --fixed-tilt-residual 0.0 ${roll} \
    --reset-xy-range 0.0 \
    --reset-velocity-xy-range 0.0 \
    --reset-velocity-z-range -0.01 0.01 \
    --strike-z-boost 0.024 \
    --strike-tilt-ramp-pitch -0.06 \
    --followup-strike-target-tilt -0.06 0.0 \
    --followup-strike-lift-boost 0.02 \
    --post-contact-return-assist-weight 0.5 \
    --post-contact-return-max-intercept-time 0.6
done
```

strike-phase-only residual sweeps:

```bash
for pitch in 0.00 0.01 0.02 0.03 0.04 0.05; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_primitive_strike_pitch_${pitch} \
    --variant-name strike_pitch_${pitch}_roll_0.00 \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --fixed-tilt-residual 0.0 0.0 \
    --strike-tilt-residual ${pitch} 0.0 \
    --recovery-tilt-residual 0.0 0.0 \
    --reset-xy-range 0.0 \
    --reset-velocity-xy-range 0.0 \
    --reset-velocity-z-range -0.01 0.01 \
    --strike-z-boost 0.024 \
    --strike-tilt-ramp-pitch -0.06 \
    --followup-strike-target-tilt -0.06 0.0 \
    --followup-strike-lift-boost 0.02 \
    --post-contact-return-assist-weight 0.5 \
    --post-contact-return-max-intercept-time 0.6
done

for roll in -0.03 -0.02 -0.01 0.00 0.01 0.02 0.03; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_primitive_strike_roll_${roll} \
    --variant-name strike_pitch_0.00_roll_${roll} \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --fixed-tilt-residual 0.0 0.0 \
    --strike-tilt-residual 0.0 ${roll} \
    --recovery-tilt-residual 0.0 0.0 \
    --reset-xy-range 0.0 \
    --reset-velocity-xy-range 0.0 \
    --reset-velocity-z-range -0.01 0.01 \
    --strike-z-boost 0.024 \
    --strike-tilt-ramp-pitch -0.06 \
    --followup-strike-target-tilt -0.06 0.0 \
    --followup-strike-lift-boost 0.02 \
    --post-contact-return-assist-weight 0.5 \
    --post-contact-return-max-intercept-time 0.6
done
```

bootstrap unblock check:

```bash
PYTHONPATH=src conda run -n mujoco_env python -c "... collect_heuristic_bootstrap_dataset(... action_mode='position_strike_tilt' ...)"
```

## 3. numeric result

### 3.1 constant pitch residual does not help

`roll=0.0` 고정에서 `pitch residual`을 `0.00`부터 `0.05`까지 바꿨다.

- best mean was only `0.167` for most settings
- worst was `0.133` at `pitch residual = 0.04`
- every config had `max_useful_bounces = 1`
- every config had `two_or_more_useful_bounce_rate = 0.0`

중요한 해석:

- base strike pitch가 이미 `-0.06` limit 근처이기 때문에, 이 sweep은 사실상 strike-phase pitch를 완화하는 방향만 시험했다.
- 그 완화는 useful chain을 열지 못했고, 대부분 `next_intercept_reachable_rate`만 낮췄다.

### 3.2 constant roll residual gives only a tiny lift

best constant roll candidate는 `roll residual = -0.02`였다.

`contact_primitive_roll_m0p02_p0_v1_summary.json`:

- `mean_useful_bounces = 0.2333`
- `max_useful_bounces = 1`
- `two_or_more_useful_bounce_rate = 0.0`

즉 baseline `0.1667`보다 약간 나아졌지만, useful second bounce조차 열지 못했다.

### 3.3 strike-only pitch residual still fails

best strike-only pitch candidate는 `pitch residual = +0.02`였다.

`contact_primitive_strike_pitch_0p02_v1_summary.json`:

- `mean_useful_bounces = 0.20`
- `max_useful_bounces = 1`
- `two_or_more_useful_bounce_rate = 0.0`
- `useful_contact_next_intercept_reachable_rate = 0.0`

즉 non-strike neutral tilt를 보존해도 결과는 달라지지 않았다.

### 3.4 strike-only roll residual is still below the gate

best strike-only roll candidate는 `roll residual = -0.02`였다.

`contact_primitive_strike_roll_m0p02_v1_summary.json`:

- `mean_useful_bounces = 0.2333`
- `max_useful_bounces = 1`
- `mean_contacts = 3.77`
- `two_or_more_useful_bounce_rate = 0.0`
- `three_or_more_useful_bounce_rate = 0.0`
- `useful_contact_next_intercept_reachable_rate = 0.0`
- `all_contact_mean_outgoing_velocity_error_norm = 1.7192`
- `useful_contact_mean_outgoing_velocity_error_norm = 0.8949`

baseline zero-residual strike config와 비교하면 mean useful bounce는 조금 좋아졌지만:

- `max_useful_bounces`는 여전히 `1`
- useful contact 이후 next intercept reachable signal은 여전히 `0.0`
- outgoing error도 baseline보다 더 좋아지지 않았다

즉 contact chain이 열린 것이 아니라, first useful contact frequency만 소폭 흔들린 것이다.

## 4. interpretation

이번 결과는 꽤 명확하다.

1. `position_strike_tilt` scaffold 자체는 동작한다.
2. 하지만 tilt-only residual은 constant로 주든 strike phase에만 주든 `2+`도 못 연다.
3. 따라서 upper-bound에서 보인 gap은 `tilt residual 2-dim`만으로는 메워지지 않는다.

즉 다음 primitive는 단순 tilt 보정이 아니라 아래 중 적어도 하나를 직접 표현해야 한다.

- strike contact point residual (`dx`, `dy`)
- strike z / impact timing residual
- follow-up lift residual

현재 마지막 residual 두 축 중 하나를 `roll` 대신 `followup lift` 또는 `strike z residual`로 바꾸는 쪽이 더 유력하다.

## 5. PPO implication

이 결과에서는 PPO를 다시 본격적으로 돌리면 안 된다.

reason:

- scripted primitive가 아직 `max_useful_bounces >= 3`는커녕 `>= 2`도 안정적으로 못 만든다.
- `three_or_more_useful_bounce_rate`는 모든 sweep에서 `0.0`이다.

다만 infrastructure 쪽 최소 unblock은 했다.

- `collect_heuristic_bootstrap_dataset()`는 이제 `action_mode="position_strike_tilt"`를 허용한다.
- 실제 1-episode bootstrap check에서 `(obs_shape=(32, 52), act_shape=(32, 5))`가 확인됐다.

## 6. conclusion

한 줄 결론:

> geometry is still not the ceiling, but the current tilt-only primitive is also not enough; the next structural step should add non-tilt contact-intent residuals before PPO resumes.