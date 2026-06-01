# contact primitive training plan

## 1. why this is the next step

`16_contact_upper_bound_report.md`에서 이미 확인됐다.

- baseline `position_strike` scripted best는 `3+`를 못 만든다.
- 하지만 같은 narrow reset/task geometry에서 contact 직후 outgoing을 `0.5~0.75` 수준으로만 보정해도 `3+`가 안정적으로 열린다.

따라서 다음 구조 변경의 목적은 아래 하나다.

> PPO가 raw EE delta를 우연히 맞추기를 기다리지 말고, contact-time outgoing correction을 직접 표현할 수 있는 작은 primitive/action space를 제공한다.

## 2. design target

새 mode는 physics를 속이는 oracle이 아니어야 한다.

즉 training-time에 ball velocity를 직접 덮어쓰면 안 된다.

대신 아래 역할을 env/controller primitive 내부에서 보장해야 한다.

1. pre-contact timing
   - 현재 best scripted contract의 intercept tracking, strike z feed-forward, follow-up return assist는 유지한다.
2. contact-time intent
   - policy가 raw cartesian delta가 아니라 `contact intent residual`을 낼 수 있어야 한다.
3. post-contact recovery
   - 현재 return assist를 유지하되, primitive가 next-ball zone을 더 직접 겨냥하도록 해야 한다.

## 3. recommended primitive v1

권장 v1은 새 `action_mode="position_strike_tilt"` 또는 이에 준하는 `contact_primitive_v1`이다.

핵심은 기존 `position_strike` base contract 위에 contact-phase residual을 추가하는 것이다.

추천 residual action dimensions:

1. `contact_dx`
   - predicted intercept xy 기준 strike point x residual
2. `contact_dy`
   - predicted intercept xy 기준 strike point y residual
3. `strike_z_residual`
   - current `strike_z_boost` / `followup_strike_lift_boost` around the best base config
4. `pitch_residual`
   - current best negative pitch base (`-0.06`) 주변 residual
5. `followup_lift_residual`
   - second-bounce 이후 recovery margin residual

초기 v1에서는 `roll`은 고정 `0.0`으로 두는 편이 낫다.

이번 upper-bound 결과는 x/z outgoing correction이 핵심이라는 쪽으로 더 강하게 기울어 있기 때문이다.

## 4. base contract to preserve

primitive v1의 내부 base는 아래를 그대로 가져가는 것이 맞다.

- `strike_z_boost = 0.024`
- `strike_tilt_ramp_pitch = -0.06`
- `followup_strike_target_tilt = (-0.06, 0.0)`
- `followup_strike_lift_boost = 0.02`
- `post_contact_return_assist_weight = 0.5`
- `post_contact_return_max_intercept_time = 0.6`

즉 primitive는 zero-action일 때 최소한 current best scripted contact contract를 재현해야 한다.

그 위에 policy residual이 contact outcome을 미세 조정하도록 만드는 것이 안전하다.

## 5. training order

다음 순서를 유지한다.

1. new primitive mode 구현
2. heuristic scripted baseline 추가
3. narrow reset 100-episode scripted diagnostic 실행
4. scripted primitive가 oracle 없이 `3+`를 재현하면
5. 그 다음에만 PPO 재개

## 6. gating criteria

primitive branch는 아래를 만족해야 다음 단계로 넘어간다.

1. scripted primitive baseline
   - `max_useful_bounces >= 3`
   - `three_or_more_useful_bounce_rate > 0`
2. 가능하면 100-episode narrow confirmation에서
   - `three_or_more_useful_bounce_rate >= 0.1`
3. oracle 없이도
   - `time_limit_episode_rate`가 의미 있게 올라가야 한다.

이 조건을 통과하지 못하면 PPO를 다시 돌리면 안 된다.

## 7. what not to do

아래 방향은 지금 우선순위가 아니다.

- outgoing reward를 더 만지는 것
- current `position_strike` 그대로 PPO를 더 오래 돌리는 것
- oracle ball-velocity clamp를 training feature로 승격하는 것

## 8. immediate implementation checklist

최소 scaffold는 이번 단계에서 이미 추가했다.

- `keepup_env.py`
  - `action_mode="position_strike_tilt"`를 추가했다.
- `controllers/heuristic_keepup.py`
  - zero tilt residual heuristic가 새 mode를 지원한다.
- `scripts/run_heuristic_keepup_diagnostic.py`
  - `--action-mode position_strike_tilt`를 받을 수 있게 했다.
- `scripts/run_ppo_learning.py`
  - 새 preset `contact_primitive_candidate`를 추가했다.

현재 확인된 smoke:

- zero-residual heuristic smoke 통과
- `contact_primitive_candidate` preset resolve + env construction 통과 (`action_size=5`, `observation_size=52`)
- `position_strike_tilt` heuristic bootstrap collection도 통과 (`obs_shape=(32, 52)`, `act_shape=(32, 5)`)
- 새 `contact_lift_candidate` preset resolve + env construction 통과 (`action_size=6`, `observation_size=52`)

추가 scripted finding:

- constant tilt residual sweep 실패: `2+`를 전혀 열지 못함
- strike-phase-only tilt residual sweep도 실패: best candidate도 `max_useful_bounces = 1`
- phase-aware non-tilt residual은 의미 있음: `strike z = +0.01` + `strike roll = -0.02` candidate가 `max_useful_bounces = 2`, `two_plus_rate = 0.11`까지 올림
- 하지만 100-episode confirmation에서도 `three_or_more_useful_bounce_rate = 0.0`이라 현재 abstraction은 여전히 `max=2` ceiling에 걸려 있음
- explicit `followup lift residual` scaffold는 추가했지만 첫 sweep에서는 gain이 없었음

남은 구현 체크리스트는 아래다.

1. `keepup_env.py`
   - current `position_strike_tilt(_lift)` base 위에서 state-dependent contact-point / impact-time residual semantics를 추가하기
2. `run_heuristic_keepup_diagnostic.py`
   - 다음 primitive semantics용 scripted tuning sweep 추가
3. `scripts/run_ppo_learning.py`
   - 새 primitive가 scripted gate를 통과한 뒤에만 PPO smoke/train/eval cycle 실행
4. `scripts/run_viewer.py`
   - 새 mode viewer playback 지원
5. 새 report
   - 다음 primitive scripted feasibility 결과 기록

한 줄 결론:

> next work should be a non-cheating contact primitive that preserves the current best strike contract and gives PPO a small contact-intent residual action space.