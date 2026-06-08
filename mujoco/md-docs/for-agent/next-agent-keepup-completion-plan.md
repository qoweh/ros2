# 다음 에이전트 작업 지시서: keep-up 완성 방향 재정렬

## 0. 목적

이 문서는 다음 에이전트가 `pingpong_rl2` 프로젝트를 이어받아, 뒤섞인 실험 흐름을 정리하고 최종 목표에 맞는 보완 작업을 진행하도록 지시하기 위한 문서다.

최종 목표:

> Franka Panda 로봇팔이 탁구채로 탁구공을 계속 위로 튕기고, 공이 다시 치기 쉬운 위치로 돌아오도록 학습시키는 것.

여기서 중요한 말은 “계속”과 “치기 쉬운 위치”다.

단순히 공을 한 번 맞히거나 위로 튕기는 것은 성공이 아니다. 공을 친 뒤 다음 공이 다시 라켓이 접근 가능한 strike zone으로 돌아와야 한다.

## 1. 현재 판단

사용자의 직감은 맞다.

> 지금 보완해야 할 핵심은 “탁구공을 치기 쉬운 쪽으로 올려치는 것”이다.

다만 이것을 단순히 `ball_velocity_x`를 줄이거나 로봇 base 중심으로 보내는 문제로 정의하면 안 된다.

정확한 목표는 아래다.

> contact 후 공의 다음 예상 intercept가 racket/controller anchor 기준 strike zone 근처에 오고, 속도와 높이도 다음 strike가 가능한 범위에 있어야 한다.

즉 “치기 쉬운 공”은 방향 하나가 아니라 아래 조건의 조합이다.

- 다음 intercept XY가 strike zone 중심 근처
- 다음 intercept까지 시간이 너무 짧지 않음
- 공 속도가 너무 빠르지 않음
- lateral velocity가 과하지 않음
- 공 높이와 하강 타이밍이 strike 가능한 범위
- 로봇팔이 body contact 없이 이동 가능한 위치
- racket이 neutral에 가까운 자세로 회복 가능

## 2. 현재 상태 요약

최근 작업에서 여러 실험이 진행됐다. 중요한 흐름은 아래다.

### 2.1 깨끗하게 보이는 결론

- `position`만으로는 useful bounce가 거의 나오지 않는다.
- `position_strike`는 contact와 upward strike를 살리는 데 도움이 된다.
- timed negative pitch는 `position_strike`보다 useful contact 쪽에서 개선 경향이 있었다.
- velocity-domain observation은 일부 지표를 개선했지만 최종 candidate로 확정할 정도는 아니다.
- reward로 projected apex XY를 직접 당기는 실험은 regression이었다.
- control-side `post_contact_return_assist`는 지금까지의 결과 중 가장 목적에 가까운 개선을 보였다.

### 2.2 현재 후보

현재 실무 후보는 아래로 본다.

- model: `clean_tnp_return_assist_v1_best_model.zip`
- 구조:
  - `action_mode=position_strike`
  - `strike_tilt_ramp_pitch=-0.03`
  - `strike_tilt_ramp_xy_tolerance=0.04`
  - `post_contact_return_assist_weight=0.5`
  - `post_contact_return_max_intercept_time=0.6`
  - velocity-domain observation은 기본으로 쓰지 않음
  - reward-side inward-return shaping은 기본으로 쓰지 않음

100-episode 기준 비교:

| model | mean useful bounces | one+ useful rate | two+ useful rate | useful contact rate | ball_out_of_bounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clean_tnp_ckpt_v1_best_model.zip` | `0.31` | `0.31` | `0.00` | `12.97%` | `78/100` |
| `clean_tnp_return_assist_v1_best_model.zip` | `0.40` | `0.38` | `0.02` | `16.0%` | `77/100` |

해석:

- return assist는 개선 방향이 맞다.
- 하지만 아직 완성 수준은 아니다.
- `ball_out_of_bounds`가 여전히 매우 많다.
- 두 번 이상 useful bounce는 매우 드물다.

## 3. 중요한 주의: 지금 바로 reward를 더 붙이지 말 것

지금까지의 문제는 reward가 부족해서만 생긴 것이 아니다.

문제:

- action mode, assist, observation, reward가 섞인 실험이 많았다.
- 어떤 변경이 실제 개선을 만들었는지 해석하기 어려운 run이 있다.
- 일부 run 이름과 실제 config가 맞지 않는 경우가 있었다.
- final model보다 best checkpoint가 나은 경우가 명확히 있었다.

따라서 다음 작업은 바로 새 reward를 추가하는 것이 아니라 아래 순서여야 한다.

1. 현재 active candidate와 preset이 실제로 일치하는지 확인한다.
2. “치기 쉬운 다음 공” metric을 먼저 정의하고 분석한다.
3. metric이 유효하면 control 또는 reward로 최소 승격한다.
4. 짧은 실험으로 확인한 뒤에만 긴 학습을 한다.

## 4. 먼저 확인할 파일

반드시 읽어라.

- `agent-answer.md`
- `agent-todo.md`
- `TODO.md`
- `pingpong_rl2/README.md`
- `pingpong_rl2/docs/report/07_reward_policy_cleanup_plan.md`
- `pingpong_rl2/docs/report/05_project_completion_plan.md`
- `pingpong_rl2/docs/report/06_learning_design_checklist.md`
- `pingpong_rl2/scripts/run_ppo_learning.py`
- `pingpong_rl2/scripts/run_ppo_rebound_analysis.py`
- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
- `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py`
- `pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py`

## 5. 1단계 작업: 현재 상태 정합성 확인

먼저 코드 구현보다 정합성 확인을 해라.

### 5.1 preset 확인

`run_ppo_learning.py`의 `final_candidate` preset이 아래와 일치하는지 확인한다.

- `action_mode=position_strike`
- `strike_tilt_ramp_pitch=-0.03`
- `strike_tilt_ramp_xy_tolerance=0.04`
- `post_contact_return_assist_weight=0.5`
- `post_contact_return_max_intercept_time=0.6`
- `include_velocity_domain_observation=False`
- reward-side return shaping 기본 off

### 5.2 문서/실험 결과 확인

`README.md`, `07_reward_policy_cleanup_plan.md`, artifact summary가 서로 같은 후보를 가리키는지 확인한다.

특히 주의:

- `clean_final_v1` 결과가 현재 `final_candidate` preset 변경 이후의 결과인지, 이전 stale run인지 확인한다.
- final model과 best checkpoint를 구분한다.
- `*_best_model.zip` 분석 시 env config가 원래 training summary에서 제대로 복원되는지 확인한다.

## 6. 2단계 작업: “치기 쉬운 다음 공” metric 정의

다음 에이전트가 가장 먼저 보완해야 할 부분은 reward가 아니라 metric이다.

현재 필요한 핵심 metric:

### 6.1 next-strike intercept metric

공이 contact 이후 다시 내려올 때, strike plane에서 어디를 지나갈지 예측한다.

대략:

```python
target_z = anchor_z + strike_plane_offset
predicted_next_intercept_time = solve_ballistic_time(ball_pos, ball_vel, target_z)
predicted_next_intercept_xy = ball_xy + ball_vxy * predicted_next_intercept_time
next_intercept_xy_error = norm(predicted_next_intercept_xy - anchor_xy)
```

기록할 것:

- `predicted_next_intercept_time`
- `predicted_next_intercept_xy`
- `next_intercept_xy_error`
- `next_intercept_reachable`: error가 strike zone radius 안인지

### 6.2 easy-ball metric

단순 위치 말고 다음 공이 쉬운지 종합 점수를 만든다.

추천 구성:

```text
easy_next_ball_score =
  + intercept_xy_score
  + intercept_time_score
  + height/descending_score
  - lateral_speed_penalty
  - excessive_speed_penalty
  - recovery_distance_penalty
```

처음에는 reward로 쓰지 말고 analysis summary에만 기록한다.

### 6.3 contact quality metric

contact 순간 공이 안정적으로 튀었는지도 기록한다.

후보:

- relative velocity: `ball_velocity - racket_velocity`
- racket face normal
- normal component
- tangential relative speed
- contact xy alignment
- outgoing lateral/vertical ratio

목적:

- 왜 공이 다음 strike zone으로 안 돌아오는지 구분한다.
- timing 문제인지, racket normal 문제인지, post-contact target 문제인지 본다.

## 7. 3단계 작업: reward/policy 보완 방향

metric 분석 후에만 아래 중 하나를 선택해라.

### 7.1 우선 후보: post-contact return assist 개선

현재 가장 유효한 방향은 reward가 아니라 control-side assist였다.

보완 후보:

- assist를 useful bounce 이후에만 켜는 현재 조건이 너무 늦은지 확인
- 첫 non-useful upward contact 후에도 약하게 켜야 하는지 검토
- assist target을 predicted intercept만 쓰지 말고 anchor와 predicted intercept 사이에서 time/readiness에 따라 조절
- `post_contact_return_assist_weight=0.5` 근처에서 `0.4`, `0.6`, `0.7`만 소수 실험
- `post_contact_return_max_intercept_time=0.6` 근처에서 `0.5`, `0.7`만 소수 실험

주의:

- 한 번에 weight와 max time을 같이 바꾸지 마라.
- reward도 같이 바꾸지 마라.

### 7.2 다음 후보: easy-next-ball reward

metric이 useful bounce 지속 여부와 잘 맞으면 reward로 승격한다.

조건:

- next_intercept_xy_error가 낮은 episode가 실제로 2회 이상 useful bounce와 상관이 있어야 한다.
- easy_next_ball_score가 높은 episode가 viewer에서도 더 안정적이어야 한다.

reward 적용 원칙:

- contact event 직후에만 적용
- weight는 작게 시작
- 기존 contact/upward reward를 압도하지 않게 함
- reward term은 하나만 추가

예:

```python
if contact_event and contact_ball_velocity_z > 0:
    reward_terms["easy_next_ball_term"] = weight * easy_next_ball_score
```

### 7.3 당장 피할 것

- global `vx` penalty 재튜닝
- `position_tilt` 재도입
- center-seeking tilt assist 미세조정
- reset curriculum 추가
- PPO에서 SAC로 바로 전환

이들은 나중 후보일 수 있지만 지금은 원인 분리를 더 흐린다.

## 8. 4단계 작업: 실험 프로토콜

긴 학습 전에는 반드시 짧은 실험으로 확인한다.

### 8.1 기본 명령

현재 후보 학습:

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
python scripts/run_ppo_learning.py \
  --preset final_candidate \
  --run-name <run_name> \
  --run-version <version> \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --run-name <run_name> \
  --run-version <version> \
  --episodes 50 \
  --compare-apex-targets
```

viewer는 최종 후보만 본다.

### 8.2 실험 단계

1. metric-only 분석
2. 50k 또는 100k 단일 seed
3. 300k 단일 seed
4. 100-episode rebound analysis
5. seed repeat
6. 1M

### 8.3 1M을 돌려도 되는 조건

아래 조건을 만족해야 한다.

- 100k 또는 300k에서 `mean_useful_bounces`가 현재 candidate보다 개선
- `one_or_more_useful_bounce_rate` 개선
- `two_or_more_useful_bounce_rate`가 유지 또는 개선
- `ball_out_of_bounds`가 악화되지 않음
- next-strike intercept metric이 개선
- viewer에서 공이 확실히 다시 칠 수 있는 쪽으로 돌아오는 경향이 보임

## 9. 평가 기준

중요 지표 우선순위:

1. `mean_useful_bounces`
2. `one_or_more_useful_bounce_rate`
3. `two_or_more_useful_bounce_rate`
4. useful contact rate
5. `ball_out_of_bounds`
6. next-strike intercept XY error
7. next intercept time
8. contact lateral/vertical ratio

주의:

- total contact count만 늘면 실패일 수 있다.
- `vx`만 줄고 useful bounce가 줄면 실패다.
- short 10-episode checkpoint eval은 참고만 한다.
- 최종 판단은 50 또는 100 episode rebound analysis로 한다.

## 10. 다음 에이전트의 산출물

다음 에이전트는 작업 후 아래 문서를 만들어라.

추천 파일:

- `pingpong_rl2/docs/report/08_easy_next_ball_completion_plan.md`

내용:

1. 현재 active candidate 정합성 확인 결과
2. `final_candidate` preset 실제 env config
3. 현재 best model과 stale run 구분
4. easy-next-ball metric 정의
5. metric-only 분석 결과
6. reward/control 보완 제안
7. 다음 실험 명령어
8. 1M 학습 가능 여부 판단

## 11. 최종 결론

지금 보완 방향은 “탁구공을 치기 쉬운 쪽으로 올려치는 것”이 맞다.

하지만 다음 에이전트는 이것을 단순 x 방향 보정으로 구현하면 안 된다.

정확히는:

> contact 이후 공의 다음 predicted intercept가 racket strike zone 안에 들어오도록 만들고, 그 intercept가 충분한 시간과 안정적인 속도를 갖도록 만드는 것.

이 기준으로 metric을 먼저 만들고, 그 다음 control assist 또는 reward를 최소로 승격해라.
