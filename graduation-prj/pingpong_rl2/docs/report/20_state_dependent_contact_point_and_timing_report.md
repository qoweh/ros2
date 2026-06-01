# state-dependent contact point and timing report

## 1. what changed

이번 단계의 목적은 `19_non_tilt_contact_residual_report.md` 이후 남은 두 local hypothesis를 빠르게 falsify 또는 promote 하는 것이었다.

남아 있던 질문은 아래 두 개였다.

1. current action abstraction 안에서 `state-dependent strike XY contact-point residual`이 실제로 의미가 있는가?
2. constant `strike z` residual보다 `late strike-only z pulse`가 더 직접적인 impact-time primitive로 작동하는가?

이번 단계에서 바꾼 코드:

- `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - `strike_xy_correction_gain`, `strike_xy_correction_max` 추가
  - `anchor - predicted_intercept` 기반 state-dependent strike XY correction 추가
  - `strike_phase_only_position_residual` 추가
- `scripts/run_heuristic_keepup_diagnostic.py`
  - `--strike-xy-correction-gain`, `--strike-xy-correction-max` 추가
  - `--strike-phase-only-position-residual` 추가

핵심은 env primitive 자체를 다시 뒤엎지 않고, 현재 abstraction 안에서 남아 있던 가장 싼 local branch를 모두 실제로 확인했다는 점이다.

## 2. commands executed

state-dependent XY correction smoke:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_xy_correction_smoke_v1 \
  --variant-name gain_050 \
  --action-mode position_strike_tilt \
  --episodes 2 \
  --strike-xy-correction-gain 0.5 \
  --strike-xy-correction-max 0.02 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual 0.0 -0.02 \
  --followup-strike-lift-boost 0.02
```

state-dependent XY gain sweep:

```bash
for gain in -0.50 0.00 0.25 0.50 0.75 1.00; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_xy_gain_${gain} \
    --variant-name gain_${gain} \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --strike-xy-correction-gain ${gain} \
    --strike-xy-correction-max 0.02 \
    --strike-position-residual 0.0 0.0 0.01 \
    --strike-tilt-residual 0.0 -0.02 \
    --followup-strike-lift-boost 0.02
done
```

late strike-only z pulse smoke:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_strike_pulse_smoke_v1 \
  --variant-name pulse_001 \
  --action-mode position_strike_tilt \
  --episodes 2 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-phase-only-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual 0.0 -0.02 \
  --followup-strike-lift-boost 0.02
```

late strike-only z pulse sweep:

```bash
for pulse in 0.00 0.005 0.010 0.015 0.020; do
  PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
    --analysis-name contact_strike_pulse_${pulse} \
    --variant-name pulse_${pulse} \
    --action-mode position_strike_tilt \
    --episodes 30 \
    --strike-position-residual 0.0 0.0 0.01 \
    --strike-phase-only-position-residual 0.0 0.0 ${pulse} \
    --strike-tilt-residual 0.0 -0.02 \
    --followup-strike-lift-boost 0.02
done
```

## 3. numeric result

### 3.1 state-dependent XY correction does nothing in the current narrow reset

gain sweep results were effectively identical for all tested values:

- `gain = -0.50`: `mean_useful_bounces = 0.667`, `max_useful_bounces = 2`, `two_plus_rate = 0.167`
- `gain = 0.00`: `0.667`, `2`, `0.167`
- `gain = 0.25`: `0.667`, `2`, `0.167`
- `gain = 0.50`: `0.667`, `2`, `0.167`
- `gain = 0.75`: `0.667`, `2`, `0.167`
- `gain = 1.00`: `0.667`, `2`, `0.167`

이 실험이 의미 없는 것은 아니다.

오히려 중요한 결론을 준다.

- current narrow reset (`reset_xy_range=0`, `reset_velocity_xy_range=0`)에서는 `anchor - predicted_intercept` lateral correction signal이 사실상 죽어 있다.
- 그래서 local linear XY contact-point correction gain은 현재 병목 축이 아니다.

즉 다음 branch가 `state-dependent x/y`라 하더라도, 단순 gain-on-correction 형태로는 충분하지 않다.

### 3.2 late strike-only z pulse is worse than the best current candidate

base candidate (`pulse = 0.00`) was still the best:

- `pulse = 0.00`: `mean_useful_bounces = 0.667`, `max_useful_bounces = 2`, `two_plus_rate = 0.167`

adding any late-only pulse degraded the result:

- `pulse = 0.005`: `mean_useful_bounces = 0.433`, `max_useful_bounces = 1`, `two_plus_rate = 0.0`
- `pulse = 0.010`: `0.400`, `1`, `0.0`
- `pulse = 0.015`: `0.400`, `1`, `0.0`
- `pulse = 0.020`: `0.400`, `1`, `0.0`

해석:

- 현재 best candidate를 만든 것은 `prepare + strike`를 통틀어 작동하는 작은 upward z bias였다.
- 반대로 `phase == strike`에서만 주는 late-only z pulse는 contact timing을 좋아지게 하지 못했고 오히려 chain을 끊었다.

즉 다음 impact-time branch도 단순 late z pulse로는 충분하지 않다.

## 4. interpretation

이번 단계는 아래 두 가설을 둘 다 local falsification했다.

1. `anchor - predicted_intercept` 기반 linear XY correction gain
2. simple late strike-only z pulse

그 결과 남는 결론은 더 좁아진다.

- 현재 best signal은 여전히 `strike_position_residual z = +0.01` + `strike_tilt_residual roll = -0.02`
- 하지만 이 abstraction은 반복적으로 `max = 2` ceiling에 걸린다
- 다음 branch는 더 직접적인 state-dependent contact primitive여야 한다

이 말의 의미는 다음 중 하나다.

1. strike contact-point를 단순 gain이 아니라 explicit primitive semantics로 분리하기
2. impact timing을 single z pulse가 아니라 contact-window primitive로 직접 제어하기
3. 또는 current narrow reset definition 자체를 revisit해서 x/y branch가 실제로 식별 가능한 regime를 따로 분리하기

현재까지의 evidence로는 1 또는 2가 더 유력하다.

## 5. conclusion

한 줄 결론:

> we are now past the cheap local branches; the next meaningful step is a richer state-dependent contact primitive, not another constant residual, linear XY gain, or late z pulse tweak.
