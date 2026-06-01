# next agent plan: pingpong_rl2 keep-up completion

작성일: 2026-06-01

이 문서는 `5.3 codex spark` 모델에게 넘길 작업 윤곽이다.  
목표는 코드를 많이 바꾸는 것이 아니라, `pingpong_rl2`가 최종 목표로 가도록 작업 순서를 다시 고정하는 것이다.

## 1. 최종 목표

`pingpong_rl2`의 최종 목표는 좁다.

- Franka Panda 로봇팔에 탁구채가 붙어 있다.
- MuJoCo 환경에서 탁구공을 계속 위로 올려친다.
- 중요한 것은 단순히 한 번 맞히는 것이 아니라, 공이 다시 라켓이 칠 수 있는 영역으로 돌아오게 만드는 것이다.
- 즉 `3+ useful bounces`가 narrow reset에서 먼저 가능해야 한다.

지금까지의 가장 중요한 실패는 이것이다.

> 공은 맞지만, 다음 타격 가능한 위치로 안정적으로 돌아오지 않는다.

따라서 지금부터의 작업은 reward weight 조절이나 PPO 장시간 학습이 아니라, contact-time control abstraction을 완성하는 일이다.

## 2. 먼저 읽을 파일

반드시 아래 순서로 읽어라.

1. `../next-agent-contact-primitive-handoff-2026-06-01.md`
2. `../last-agent-answer.txt`
3. `docs/report/16_contact_upper_bound_report.md`
4. `docs/report/17_contact_primitive_training_plan.md`
5. `docs/report/19_non_tilt_contact_residual_report.md`
6. `docs/report/20_state_dependent_contact_point_and_timing_report.md`
7. `src/pingpong_rl2/envs/keepup_env.py`
8. `src/pingpong_rl2/controllers/heuristic_keepup.py`
9. `src/pingpong_rl2/envs/pingpong_sim.py`
10. `scripts/run_heuristic_keepup_diagnostic.py`
11. `scripts/run_ppo_learning.py`

주의:

- `README.md`는 일부 내용이 오래됐다.
- 최신 판단은 `docs/report/16~20`과 root handoff 문서가 더 정확하다.

## 3. 현재 확정된 사실

현재 기준에서 이미 확인된 사실은 아래다.

- task geometry 자체는 ceiling이 아니다.
- `contact_oracle_blend=0.5~0.75`로 desired outgoing velocity를 일부만 보정하면 narrow reset에서 stable `3+`가 열린다.
- 따라서 물리적으로 반복 keep-up이 불가능한 환경은 아니다.
- 하지만 현재 plain `position_strike` abstraction은 `3+`를 못 만든다.
- `position_strike_tilt` tilt-only branch도 불충분했다.
- phase-aware non-tilt residual은 처음으로 `2+`를 열었다.
- 현재 best scripted candidate는 `max_useful_bounces=2`에서 막힌다.
- cheap local branch들은 이미 falsify됐다.

현재 best scripted candidate:

```text
action_mode = position_strike_tilt
strike_position_residual = (0.0, 0.0, +0.01)
strike_tilt_residual = (0.0, -0.02)
followup_strike_lift_boost = 0.02
strike_z_boost = 0.024
strike_tilt_ramp_pitch = -0.06
followup_strike_target_tilt = (-0.06, 0.0)
post_contact_return_assist_weight = 0.5
post_contact_return_max_intercept_time = 0.6
```

100-episode confirmation:

```text
mean_useful_bounces = 0.58
max_useful_bounces = 2
one_or_more_useful_bounce_rate = 0.47
two_or_more_useful_bounce_rate = 0.11
three_or_more_useful_bounce_rate = 0.0
```

## 4. 하지 말 것

아래 작업을 반복하지 마라.

- plain `position_strike`로 PPO를 더 오래 돌리기
- reward weight를 조금씩 바꾸기
- `trajectory_match_reward_weight`를 다시 키우기
- tilt-only residual sweep 반복
- constant strike x/y/z residual만 더 찍기
- linear `anchor - predicted_intercept` XY gain 재시도
- simple late strike-only z pulse 재시도
- follow-up contact offset만 재시도
- oracle ball velocity override를 학습 feature로 승격

이것들은 이미 의미 있게 테스트됐거나, 현재 결론상 우선순위가 낮다.

## 5. 핵심 병목

현재 병목은 policy가 어떤 값을 얼마나 학습하느냐가 아니다.

병목은 이거다.

> 현재 action/control surface가 contact 순간의 라켓 속도, 라켓 법선, contact point, impact timing을 충분히 직접 표현하지 못한다.

`keepup_env.py`의 현재 action modes:

- `position`
- `position_strike`
- `position_tilt`
- `position_strike_tilt`
- `position_strike_tilt_lift`

현재 best는 `position_strike_tilt` 기반이지만, 이것도 아직 raw target position/tilt residual에 가깝다.  
`3+`를 열려면 더 직접적인 contact primitive가 필요하다.

## 6. 다음 작업의 우선순위

우선순위는 아래 순서다.

### Step 1. scripted gate를 먼저 통과시켜라

PPO 전에 scripted diagnostic으로 새 primitive가 `3+`를 만들 수 있어야 한다.

통과 기준:

```text
max_useful_bounces >= 3
three_or_more_useful_bounce_rate > 0
```

가능하면 100-episode narrow reset에서:

```text
three_or_more_useful_bounce_rate >= 0.10
```

이 기준 전에는 PPO를 시작하지 마라.

### Step 2. explicit contact-frame primitive를 먼저 시도해라

가장 유력한 다음 branch는 explicit contact-frame contact-point primitive다.

기존 방식:

```text
target_position[:2] = predicted_intercept_xy + raw_action_xy
```

추천 방향:

```text
contact_dx, contact_dy를 world x/y가 아니라 contact frame 또는 return-target frame 기준으로 정의한다.
```

의도:

- predicted intercept 주변에서 그냥 x/y를 더하는 것이 아니라,
- 공을 target apex/anchor 쪽으로 보내기 위한 접촉점 의미를 action에 부여한다.

후보 semantics:

```text
action[0] = radial_contact_offset
action[1] = tangential_contact_offset
action[2] = strike_z_residual
action[3] = pitch_residual
action[4] = roll_residual
```

여기서 radial direction은 예를 들면:

```python
target_xy = env._controller_anchor_position()[:2]
incoming_xy = env._predicted_intercept_xy()
radial = normalize(target_xy - incoming_xy)
tangent = [-radial_y, radial_x]
contact_offset_xy = radial * action[0] + tangent * action[1]
```

단, narrow reset에서는 `target_xy - incoming_xy`가 작을 수 있으므로 fallback direction을 정의해야 한다.

fallback 후보:

- racket local x/y axes
- ball incoming horizontal velocity direction
- contact normal projected to xy

목표는 단순 gain이 아니라, `contact point를 어떤 좌표계에서 조절하는지`를 명확히 하는 것이다.

### Step 3. impact-time primitive를 두 번째 후보로 둬라

contact-frame primitive가 바로 열리지 않으면 impact-time primitive를 시도한다.

단순 late z pulse는 이미 실패했다.  
그러므로 다음은 "phase == strike에서 z를 더한다"가 아니라 windowed profile이어야 한다.

후보:

```text
action = [contact_lead_time, z_pulse_scale, z_pulse_width, pitch_residual, roll_residual]
```

또는 더 작게:

```text
action[0] = strike_window_z_lead
action[1] = strike_window_z_release
```

핵심은 contact 직전/직후의 짧은 window 안에서만 target height 또는 target velocity를 바꾸는 것이다.

## 7. 구현할 때 지켜야 할 구조

새 mode를 만들 경우 이름은 명확하게 하라.

추천:

```text
action_mode = contact_frame_strike
```

또는 기존 mode 위에서 실험한다면:

```text
position_strike_tilt_contact
```

수정해야 할 가능성이 높은 파일:

- `src/pingpong_rl2/envs/keepup_env.py`
  - `_ACTION_MODES`
  - action size 구성
  - `step()` action parsing
  - `_strike_action_target_position()`
  - info/training_config export
- `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - 새 primitive action을 scripted로 내는 policy logic
- `scripts/run_heuristic_keepup_diagnostic.py`
  - 새 primitive CLI 옵션과 summary 기록
- `scripts/run_ppo_learning.py`
  - scripted gate 통과 후에만 preset 추가
- `scripts/run_viewer.py`
  - 새 mode를 viewer로 확인할 수 있게 연결
- `tests/test_keepup_env.py`
  - action size, observation, clipping, env construction test
- `tests/test_keepup_contract_features.py`
  - heuristic action shape test

## 8. 실험 순서

### 8.1 smoke

새 action mode가 생성되고 heuristic action이 shape/clipping을 통과하는지 먼저 본다.

```bash
cd pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_frame_smoke_v1 \
  --episodes 2 \
  --action-mode <new_mode>
```

### 8.2 30-episode narrow screening

작은 sweep으로 후보를 걸러라.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name <candidate_name> \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  ...
```

정렬 기준:

1. `max_useful_bounces`
2. `three_or_more_useful_bounce_rate`
3. `two_or_more_useful_bounce_rate`
4. `mean_useful_bounces`
5. `all_contact_mean_outgoing_velocity_error_norm`

### 8.3 100-episode confirmation

30-episode에서 `max_useful_bounces >= 3`가 나온 후보만 100 episodes로 확인한다.

### 8.4 viewer

숫자가 좋아도 반드시 viewer로 본다.

확인할 것:

- 공이 라켓에서 멀어지는 방향으로 누적 drift하지 않는가
- contact가 chatter/long-contact가 아닌가
- 유용한 bounce로 count되는 것이 실제로 사람이 보기에도 keep-up인가

## 9. PPO 재개 조건

PPO는 아래 조건 이후에만 시작한다.

```text
scripted new primitive:
max_useful_bounces >= 3
three_or_more_useful_bounce_rate > 0
```

PPO preset은 하나만 만든다.

```text
--preset contact_frame_candidate
```

그 preset에는 필요한 env kwargs를 모두 넣어서 CLI 조합이 흩어지지 않게 한다.

초기 PPO:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_candidate \
  --run-name contact_frame \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7 \
  --checkpoint-interval 5000 \
  --checkpoint-eval-episodes 10 \
  --eval-episodes 10
```

처음부터 1M을 돌리지 마라.  
100k에서 `one+`, `two+`, `three+`, failure distribution을 보고 결정한다.

## 10. 문서 산출물

작업 후 반드시 report를 남겨라.

추천 파일:

```text
docs/report/21_contact_frame_primitive_report.md
```

필수 내용:

- 어떤 primitive semantics를 추가했는가
- 왜 기존 residual과 다른가
- smoke 명령어
- 30-episode sweep 결과
- 100-episode confirmation 결과
- viewer에서 본 qualitative result
- PPO를 시작해도 되는지 여부

## 11. 한 줄 결론

이 프로젝트의 다음 단계는 "더 학습"이 아니다.  
다음 단계는 `3+`를 scripted로 먼저 만드는 contact primitive를 찾는 것이다.  
그 primitive가 열리면 그때 PPO를 붙인다.
