# contact feasibility map report

## 1. what changed

이번 단계의 목표는 PPO를 더 돌리는 것이 아니라, scripted controller가 현재 `position_strike` 제어면에서 원하는 outgoing velocity를 실제로 만들 수 있는지 확인하는 것이었다.

추가한 코드:

- `scripts/run_contact_feasibility_map.py`
  - pitch / roll / strike z boost / followup lift boost / contact offset ratio를 grid sweep한다.
  - coarse 1-episode sweep 후 top-k config를 narrow multi-episode로 다시 평가한다.
  - config summary CSV와 contact-level CSV를 동시에 남긴다.

이번 단계에서 바꾼 것은 reward나 PPO preset이 아니다. `scripted control surface`가 어디까지 가능한지 먼저 측정했다.

## 2. commands executed

main coarse sweep:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_contact_feasibility_map.py \
  --analysis-name contact_feasibility_map_v1 \
  --pitch-values -0.06 -0.03 0.0 0.03 0.06 \
  --roll-values -0.03 0.0 0.03 \
  --strike-z-boost-values 0.012 0.018 0.024 \
  --followup-lift-boost-values 0.0 0.02 \
  --contact-offset-ratio-values 0.0 \
  --coarse-episodes 1 \
  --top-k 3 \
  --finalist-episodes 30
```

best config 주변 contact offset follow-up sweep:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_contact_feasibility_map.py \
  --analysis-name contact_feasibility_offset_v1 \
  --pitch-values -0.06 \
  --roll-values 0.0 \
  --strike-z-boost-values 0.024 \
  --followup-lift-boost-values 0.02 \
  --contact-offset-ratio-values 0.0 0.25 0.5 0.75 \
  --contact-offset-max 0.06 \
  --coarse-episodes 5 \
  --top-k 4 \
  --finalist-episodes 50
```

best finalist 100-episode confirmation:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_feasibility_best100_v1 \
  --variant-name contact_feasibility_best_offset025 \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --followup-strike-lift-boost 0.02 \
  --followup-strike-contact-offset-ratio 0.25 \
  --followup-strike-contact-offset-max 0.06
```

## 3. main sweep result

`contact_feasibility_map_v1`은 총 `90`개 coarse config를 평가했고, top `3`개를 `30` narrow episodes로 다시 돌렸다.

### 3.1 best coarse config

best single-episode coarse result:

| pitch | roll | strike z boost | followup lift boost | max useful bounces | useful contact rate | all-contact mean outgoing error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `-0.06` | `0.0` | `0.024` | `0.02` | `2` | `1.00` | `0.5357` |

같은 deterministic single-episode에서 이 config는 `2` useful bounces까지는 만들었다.

그 시점 contact-scale 수치:

- `mean_contact_racket_velocity_z = 0.6037`
- `mean_pre_contact_ball_velocity_z = -3.0610`
- `mean_actual_outgoing_velocity_z = 3.4559`

즉, 원하는 outgoing 방향으로 공을 보내는 조합 자체는 존재한다. 문제는 그것이 narrow rollout 전체에서 `3+`로 이어지느냐이다.

### 3.2 finalist re-evaluation over 30 episodes

best finalist는 같은 config였다.

| config | mean useful bounces | max useful bounces | one+ rate | two+ rate | three+ rate | all-contact mean outgoing error | useful-contact mean outgoing error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pitch=-0.06, roll=0.0, strike_z_boost=0.024, followup_lift_boost=0.02` | `0.5667` | `2` | `0.50` | `0.0667` | `0.00` | `1.5253` | `0.7318` |

positive pitch finalist 두 개는 오히려 더 약했다.

| config | mean useful bounces | max useful bounces | three+ rate |
| --- | ---: | ---: | ---: |
| `pitch=+0.03, roll=0.0, strike_z_boost=0.018, followup_lift_boost=0.0` | `0.40` | `1` | `0.00` |
| `pitch=+0.06, roll=+0.03, strike_z_boost=0.018, followup_lift_boost=0.0` | `0.3667` | `1` | `0.00` |

이 결과만 보면 이번 sweep 범위에서는 `pitch sign`이 반대로 뒤집혀 있지는 않았다. 현재 sweep에서는 여전히 음수 pitch 쪽이 더 좋았다.

하지만 중요한 결론은 더 단순하다.

> best finalist도 `3+`를 전혀 만들지 못했다.

## 4. focused offset sweep result

best base config 주변에서 follow-up contact offset만 다시 좁혀 봤다.

50-episode finalist 결과:

| contact offset ratio | mean useful bounces | max useful bounces | one+ rate | two+ rate | three+ rate | all-contact mean outgoing error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.00` | `0.62` | `2` | `0.54` | `0.08` | `0.00` | `1.5292` |
| `0.25` | `0.50` | `2` | `0.38` | `0.12` | `0.00` | `1.6704` |
| `0.50` | `0.56` | `2` | `0.46` | `0.10` | `0.00` | `1.7384` |
| `0.75` | `0.42` | `2` | `0.34` | `0.08` | `0.00` | `1.7744` |

해석:

- `offset=0.25`는 `two+ rate`만 보면 가장 높았다 (`0.12`).
- 하지만 `mean useful bounces`, `one+ rate`, `all-contact error`는 base offset `0.0`가 더 나았다.
- 더 중요한 점은 어떤 offset에서도 `max_useful_bounces`가 `2`를 넘지 못했다는 것이다.

즉 follow-up contact target을 anchor 쪽으로 조금 당기는 것만으로는 `3+` 병목이 닫히지 않았다.

## 5. 100-episode best-candidate confirmation

`two+ rate` 기준으로 가장 유망했던 `offset=0.25` config를 100 narrow episodes로 다시 확인했다.

결과:

- `mean_useful_bounces = 0.33`
- `max_useful_bounces = 2`
- `one_or_more_useful_bounce_rate = 0.29`
- `two_or_more_useful_bounce_rate = 0.04`
- `three_or_more_useful_bounce_rate = 0.0`
- `all_contact_mean_outgoing_velocity_error_norm = 1.6327`
- `useful_contact_mean_outgoing_velocity_error_norm = 0.6910`

즉 stage-2 pass criterion인 아래 조건을 충족하지 못했다.

- deterministic / narrow scripted best가 `max_useful_bounces >= 3`
- narrow evaluation에서 `three_or_more_useful_bounce_rate > 0`

현재 scripted control surface는 여전히 `2` useful bounces 근처에서 ceiling이 있다.

## 6. controller / physics interpretation

이번 단계에서 physics parameter를 바꾸지는 않았다. 다만 scene sanity check는 했다.

- `ball_geom`
  - `mass = 0.0027`
  - `friction = 0.02 0.001 0.0001`
  - `solref = 0.001 0.01`
  - `solimp = 0.9 0.95 0.001 0.5 2`
- `racket_head`
  - `mass = 0.11`
  - `friction = 0.22 0.001 0.0001`
  - `solref = 0.002 1`
  - `solimp = 0.95 0.99 0.001`

이 값들만 보면 바로 비정상이라고 단정할 정도의 극단값은 보이지 않았다. 반면 scripted feasibility는 반복적으로 `2`에서 멈췄다.

수치적으로도 controller ceiling 신호가 보인다.

- best 30-episode finalist에서 `mean_contact_racket_velocity_z = 0.4125`
- 같은 config에서 `mean_pre_contact_ball_velocity_z = -2.2715`
- `mean_actual_outgoing_velocity_z = 1.8612`

즉 공은 분명 위로 되돌아가지만, 지금 control surface가 후속 strike zone으로 안정적으로 되돌리는 접촉을 반복 재현할 만큼 충분히 강하고 직접적이지 않다.

## 7. conclusion

이번 stage-2 결론은 `보류`가 아니라 사실상 `현재 구조에서 불충분`이다.

1. scripted sweep은 원하는 outgoing velocity에 가까운 contact 조합을 일부 찾을 수 있다.
2. 하지만 현재 `position_strike + heuristic/controller` 조합으로는 narrow reset에서도 `3+` useful bounces를 만들지 못했다.
3. contact offset을 추가해도 ceiling은 `2`를 넘지 못했다.
4. 따라서 지금은 PPO를 재개할 단계가 아니다.

다음 단계는 아래 중 하나여야 한다.

- `contact_primitive` 또는 `hybrid_strike` action mode를 만들어 strike timing / pitch / roll / z pulse를 primitive 내부에서 직접 보장한다.
- 또는 controller/action abstraction을 바꿔 contact-time racket velocity / normal을 더 직접 제어할 수 있게 만든다.

한 줄 결론:

> current scripted feasibility map still fails the `3+` gate, so the bottleneck remains contact/control execution rather than PPO reward shaping.