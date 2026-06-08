# pingpong_rl2 keep-up task rethink plan

## 0. 참고 소스와 전제

- 이번 검토는 아래 실제 존재 파일을 기준으로 했다.
  - `pingpong_rl2/docs/report/07_reward_policy_cleanup_plan.md`
  - `pingpong_rl2/docs/report/08_easy_next_ball_completion_plan.md`
  - `pingpong_rl2/README.md`
  - `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
  - `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py`
  - `pingpong_rl2/scripts/run_ppo_learning.py`
  - `pingpong_rl2/scripts/run_ppo_rebound_analysis.py`
  - `pingpong_rl2/scripts/run_viewer.py`
- 지시문에 적힌 `agent-answer.md`, `agent-todo.md`는 현재 워크스페이스에서 찾지 못했다.
- 대신 최근 맥락 확인용으로 아래 파일도 함께 참고했다.
  - `next-agent-keepup-completion-plan.md`
  - `todo-agent.md`

## 1. 지금 방향이 왜 미세 튜닝만으로는 부족한가

현재 최고 후보인 `clean_tnp_return_assist_v1_best_model.zip`은 방향은 맞지만, 최종 목표에는 아직 멀다.

- 100-episode 기준 `mean_useful_bounces=0.40` 수준이다.
- `two_or_more_useful_bounce_rate=0.02`로 반복 keep-up이 거의 없다.
- `ball_out_of_bounds`가 여전히 매우 많다.
- `post_contact_return_assist_weight 0.5 -> 0.6`처럼 한 변수만 바꿔도 일부 aggregate는 비슷해 보이지만, useful-contact 기준 next-ball quality는 크게 악화됐다.
- reward-side projected-apex return shaping은 이미 regression이었다.

핵심 문제는 숫자 하나를 더 깎는 튜닝이 아니라, 현재 MDP와 controller 책임 분리가 `한 번 잘 치기`에는 더 가깝고 `다음 공을 다시 칠 수 있게 만들기`에는 직접적이지 않다는 점이다.

현재 구조를 phase 관점으로 보면:

- prepare/strike는 비교적 잘 정의돼 있다.
- return shaping은 controller assist로 일부 들어가 있다.
- recovery는 사실상 명시적 task contract가 없다.

또한 `position_strike`에서는 policy action이 주로 hand-coded target generator 위의 residual이다. 즉 RL policy가 반복 rally 전략을 직접 배우기보다, controller가 이미 정한 목표를 주변에서 미세 수정하는 구조다. 이런 구조에서는 assist weight나 pitch ramp만 만지는 미세 조정이 phase 책임 공백을 메우지 못한다.

## 2. keep-up task phase 분해

### 2.1 prepare

- phase 정의
  - 공이 하강 중이고 아직 contact 전인 구간.
  - 현재 코드에서는 `ball_velocity_z < descending_ball_velocity_threshold`일 때 prepare/strike 쪽 로직으로 들어간다.
- 현재 observation
  - joint/racket/ball position, velocity
  - `ball_relative_position`
  - `predicted_intercept_relative_xy`
  - `predicted_intercept_time`
  - 선택적으로 `relative_velocity`, `racket_face_normal`
- 현재 action/control
  - `position_strike`에서 XY는 `_predicted_intercept_xy() + action[:2]`
  - Z는 `anchor_z + _strike_lift_feedforward() + action[2]`
  - `_guarded_target_position()`가 pre-contact XY/Z를 제한하고 body keepout을 건다.
- 현재 reward
  - descending strike window에서만 `tracking_term`
- 현재 info/logging
  - `predicted_intercept_xy_error`, `predicted_intercept_time`
  - `target_position`, `ball_height_above_racket`, `xy_alignment_error`
- 부족한 점
  - phase flag가 없다.
  - `successful_bounce_count`는 info에는 있지만 observation에는 없다.
  - 현재 observation의 intercept는 이번 strike용이고, contact 이후 다음 공의 descending intercept는 아니다.
  - `next_intercept_reachable`, required recovery distance/time 같은 readiness 정보가 없다.

### 2.2 strike

- phase 정의
  - 공이 strike window 안으로 들어왔고, 라켓이 위로 통과하며 useful contact를 만들어야 하는 구간.
- 현재 observation
  - prepare와 동일하다.
  - active preset에서는 `racket_face_normal`, `relative_velocity`가 observation에 기본 포함되지 않는다.
- 현재 action/control
  - `_strike_lift_feedforward()`가 upward strike timing을 일부 controller 쪽에서 만든다.
  - `strike_tilt_ramp_pitch`가 정렬이 맞을 때 음의 pitch ramp를 건다.
  - policy는 그 위에 residual 위치 delta만 준다.
- 현재 reward
  - useful contact일 때만 `contact_bonus`
  - useful contact일 때만 `apex_match_term`
  - optional return/x reward는 기본 off
- 현재 info/logging
  - contact-time ball/racket velocity
  - contact xy error
  - projected contact apex height
  - contact-time racket face normal
- 부족한 점
  - policy가 `지금 올라쳐야 하는 순간`을 phase 변수로 직접 받지 않는다.
  - active preset 기준으로 contact 직전 face normal이나 relative velocity를 observation에서 못 본다.
  - useful strike 판정은 명확하지만, 왜 useful strike가 다음 공 준비로 이어졌는지는 reward/observation에 연결되지 않는다.

### 2.3 return shaping

- phase 정의
  - contact 이후 공을 다음 strike zone 쪽으로 보내는 구간.
  - 현재 코드는 `successful_bounce_count > 0` 이고 공이 다시 상승 중일 때만 assist를 켠다.
- 현재 observation
  - 여전히 현재 ball/racket 상태와 이번 intercept 정보만 본다.
  - analysis script의 `next_intercept_*`, `easy_next_ball_score`는 env observation에 없다.
- 현재 action/control
  - `_post_contact_return_target_xy()`가 anchor와 future intercept XY를 weight로 섞는다.
  - `position_strike`는 공이 상승 중이면 XY target을 이 return target 쪽으로 보낸다.
- 현재 reward
  - 기본 preset에서는 없다.
  - `useful_contact_return_target_xy_reward_weight`는 존재하지만 기본 off다.
- 현재 info/logging
  - env info에는 next-intercept quality가 없다.
  - `run_ppo_rebound_analysis.py`에서만 contact 이후 descending next intercept, reachable 여부, easy-next-ball score를 계산한다.
- 부족한 점
  - assist가 `첫 useful bounce 이후`에만 켜져 시작 시점이 늦다.
  - target이 `controller_anchor` 중심이라 실제 racket workspace region을 직접 표현하지 못한다.
  - policy가 다음 공 feasibility를 보지 못한 채 controller assist 결과만 뒤늦게 따른다.

### 2.4 recovery

- phase 정의
  - contact 이후 라켓과 팔을 다음 strike 가능한 자세와 위치로 되돌리는 구간.
  - 현재 코드는 explicit phase가 아니라, anchor/return target과 keepout 제한의 결과로 간접 처리한다.
- 현재 observation
  - racket position/velocity, target position은 있다.
  - last contact timing, time since contact, recovery readiness는 없다.
- 현재 action/control
  - `_post_contact_return_target_xy()` 또는 anchor 복귀
  - `_guarded_target_position()`의 body keepout
  - tilt는 `position_strike`에서 사실상 pre-contact ramp 위주라 recovery tilt contract가 없다.
- 현재 reward
  - 명시적 recovery reward가 없다.
  - 실패 penalty만 간접적으로 있다.
- 현재 info/logging
  - `target_position`, `racket_velocity_z`, body-contact failure reason
  - recovery 성공/실패를 직접 나타내는 scalar가 없다.
- 부족한 점
  - recovery가 task의 일부가 아니라 side effect로만 존재한다.
  - 다음 intercept를 칠 수 있는 pose/time margin이 reward나 observation으로 연결되지 않는다.
  - prepare와 recovery가 policy 입장에서 거의 구분되지 않는다.

## 3. phase coverage 요약

| phase | observation | control | reward | 현재 판단 |
| --- | --- | --- | --- | --- |
| prepare | 부분적으로 충분 | 강함 | 강함 | 한 번 맞히기용 준비는 된다 |
| strike | 부분적으로 충분 | 강함 | 강함 | useful upward contact까지는 다룬다 |
| return shaping | 부족 | 부분적으로 존재 | 거의 없음 | 다음 공 방향은 controller assist에 과하게 의존한다 |
| recovery | 부족 | 약함 | 없음 | 반복 keep-up 실패의 큰 공백이다 |

## 4. 현재 MDP와 controller 책임 경계에 대한 판단

### 4.1 현재 MDP가 반복 keep-up을 배우기에 충분한가

단일 strike 학습에는 어느 정도 충분하지만, 반복 keep-up에는 부족하다.

이유:

- observation이 현재 strike용 intercept 정보는 주지만, contact 이후 다음 공 feasibility는 직접 주지 않는다.
- `successful_bounce_count`는 info에는 있지만 observation에는 없다.
- policy는 explicit phase flag 없이 공 속도와 위치에서 phase를 추론해야 한다.
- recovery readiness가 observation/reward에 없다.
- `position_strike` action은 직접적인 rally action이 아니라, hand-coded target 위의 residual이라 phase 전략 학습이 간접적이다.

### 4.2 controller가 무엇을 책임지고 있는가

현재 controller가 이미 많은 책임을 지고 있다.

- prepare: current intercept tracking
- strike: lift feed-forward + timed negative pitch
- return shaping: post-contact return assist
- recovery: anchor 복귀와 keepout 제한

반대로 policy는 그 사이에서 residual correction을 할 뿐이다. 이 구조는 안정성에는 도움이 되지만, 반복 keep-up 실패 원인이 policy 부족인지 controller contract 부족인지 흐리게 만든다.

## 5. “치기 쉬운 다음 공”의 정확한 정의

현재 analysis의 `easy_next_ball`은 useful metric이지만, 아직 최종 task contract는 아니다. 가장 큰 이유는 target이 여전히 `controller_anchor` point 중심이기 때문이다.

반복 keep-up 관점에서 다음 공은 아래를 만족해야 한다.

- contact 이후 공이 다시 descending strike plane을 통과한다.
- 그 intercept가 단일 점이 아니라 `다음 strike 가능한 racket workspace region` 안에 들어온다.
- intercept까지 남은 시간이 controller가 recovery를 끝내기에 충분하다.
- descending speed와 lateral speed가 다음 strike를 망칠 정도로 크지 않다.
- required recovery displacement와 pose change가 controller 한계와 body keepout 안에서 가능하다.

즉 target은 `home anchor` 한 점이 아니라 아래와 같은 feasible region이어야 한다.

- 중심: controller anchor 근처 strike zone
- 반경: `strike_zone_xy_radius`와 실제 recovery 가능 거리
- 시간 조건: `t_min <= next_intercept_time <= t_max`
- 속도 조건: descending speed, lateral speed upper bound
- pose 조건: neutral 또는 다음 strike 가능한 face orientation으로 복귀 가능

### 5.1 왜 first-contact easy-next-ball gap이 아직 역방향인가

현재 역방향 현상은 이상한 것이 아니라 target mismatch 가능성이 크다.

- first contact에서 anchor 점오차가 조금 커도, 오히려 시간이 더 벌어져 다음 strike가 쉬워질 수 있다.
- 현재 metric은 recovery cost를 점 하나의 XY error로 충분히 표현하지 못한다.
- 첫 contact quality보다 contact 이후 recovery 경로가 두 번째 bounce를 더 크게 좌우할 수 있다.

결론:

- easy-next-ball은 유지하되, target을 `controller_anchor point`에서 `reachable strike workspace region`으로 올려야 한다.
- reward 승격 전에는 first-contact gap 부호가 덜 뒤집히는지 먼저 다시 봐야 한다.

## 6. scripted sanity controller가 필요한가

필요하다.

지금은 PPO가 못 배우는 것인지, 환경/제어/물리가 반복 keep-up을 허용하지 않는 것인지가 아직 분리되지 않았다. 특히 current best가 `0.40` / `0.02` 수준에 머무르는 상황에서는, 먼저 heuristic baseline이 2~3회 keep-up을 만들 수 있는지 봐야 한다.

### 6.1 왜 필요한가

- 반복 keep-up이 controller/physics 상 불가능하면 reward나 PPO를 더 만져도 소모전이 된다.
- 현재 contact 파라미터는 이미 커스텀돼 있다.
  - `ball_geom`: `friction="0.02 0.001 0.0001"`, `solref="0.001 0.01"`
  - `racket_head`: `friction="0.22 0.001 0.0001"`, `solref="0.002 1"`
- 즉 물리는 기본값이 아니라 사람이 정한 설정이고, 이 설정이 반복 keep-up에 충분한지는 heuristic으로 먼저 진단하는 편이 낫다.

### 6.2 최소 heuristic controller 후보

- prepare
  - current descending intercept XY로 이동
- strike
  - 공이 내려오고 `predicted_intercept_time`이 작아질 때만 upward strike
  - pitch는 fixed negative pitch 또는 current timed negative pitch ramp 사용
- return shaping
  - contact 이후 descending next-intercept feasible region 쪽으로 target 이동
- recovery
  - contact 직후 tilt를 neutral band로 되돌리고, 다음 intercept에 맞는 XY/time margin을 확보
- safety
  - Z strike velocity 상한과 body keepout 유지

### 6.3 진단 기준

heuristic baseline은 완벽할 필요가 없다. 아래만 확인하면 충분하다.

- deterministic 또는 좁은 reset 범위에서 2 useful bounce 이상이 반복적으로 나오는가
- `ball_out_of_bounds`보다 `time_limit` 또는 다음 strike 실패가 더 자주 보이는가
- useful contact 이후 next-intercept reachable rate가 의미 있게 올라가는가

이게 안 되면 PPO reward 미세 튜닝 전에 env/control/physics를 먼저 봐야 한다.

## 7. observation 보강 후보 검토

아래 표는 현재 policy가 실제로 보는 것과, info/analysis에만 있거나 아직 없는 것을 정리한 것이다.

|후보 정보|현재 observation|info/logging만 존재|현재 없음|판단|
|---|---|---|---|---|
|descending/ascending/post-contact/recovery phase flag|||O|가장 먼저 검토할 가치가 있다|
|last contact step / time since contact|||O|recovery 분리를 위해 필요하다|
|clipped successful bounce count||O||observation에 넣을 가치가 있다|
|current predicted intercept xy/time|O|O||이미 있다|
|next descending intercept xy/time after contact||analysis only||반복 keep-up용 핵심 후보다|
|next intercept reachable flag||analysis only||task contract용 핵심 후보다|
|required recovery distance / readiness|||O|recovery phase에 필요하다|
|racket velocity|O|O||이미 있다|
|racket face normal|active preset에서는 보통 없음|O||active preset에서는 observation 보강 후보|
|relative ball-racket velocity|active preset에서는 없음|||active preset에서는 observation 보강 후보|
|target position error|간접 계산 가능|O||명시적 scalar로 넣을 수 있다|
|controller anchor position||O||target region 해석용 후보|

### 7.1 observation 보강 우선순위

바로 다 넣지 말고 아래 순서가 적절하다.

1. phase flag + time since contact
2. next descending intercept xy/time + reachable flag
3. recovery readiness 또는 required recovery distance

`relative_velocity`, `racket_face_normal`은 보조 후보지만, 지금 막힌 지점은 contact physics 자체보다 next-ball preparation contract에 더 가깝다.

## 8. reward 보강 후보

새 reward는 지금 바로 넣지 않는다. 다만 설계 후보는 아래 세 개 이하로 제한한다.

### 8.1 후보 1: next-intercept reachable bonus

- 적용 시점
  - upward contact event 직후
- 제안 초기 weight
  - `0.2`
- 정의 초안
  - contact 이후 계산한 descending next-intercept가 feasible region 안이면 작은 bonus
- exploit 가능성
  - 공을 너무 약하게 띄워 reachable만 맞추고 useful strike 품질은 놓칠 수 있다.
- 기존 reward와 충돌 가능성
  - `contact_bonus`, `apex_match_term`보다 이 보상이 커지면 low-energy safe tap을 학습할 수 있다.
- 성공/실패 판단 지표
  - `useful_contact_next_intercept_reachable_rate`
  - `two_or_more_useful_bounce_rate`
  - `ball_out_of_bounds`

### 8.2 후보 2: easy-next-ball event reward

- 적용 시점
  - upward contact event 직후
- 제안 초기 weight
  - `0.1`
- 정의 초안
  - current `easy_next_ball_score`를 region-based target으로 수정한 뒤 작은 event reward로 사용
- exploit 가능성
  - anchor-centric score가 남아 있으면 실제 repeatability보다 점수 최적화로 흐를 수 있다.
- 기존 reward와 충돌 가능성
  - `apex_match_term`과 target mismatch가 있으면 높이는 맞추지만 recovery가 어려운 공을 강화할 수 있다.
- 성공/실패 판단 지표
  - first-contact easy-score gap 부호
  - `one_or_more_useful_bounce_rate`
  - `two_or_more_useful_bounce_rate`

### 8.3 후보 3: recovery readiness reward

- 적용 시점
  - contact 직후부터 다음 descending strike window 전까지
- 제안 초기 weight
  - step당 최대 `0.02`
- 정의 초안
  - 라켓이 다음 intercept feasible pose/velocity에 가까워질수록 작은 dense reward
- exploit 가능성
  - anchor 근처에서 흔들리며 dense reward를 먹는 recovery chatter가 생길 수 있다.
- 기존 reward와 충돌 가능성
  - prepare의 `tracking_term`과 겹쳐 phase가 다시 흐려질 수 있다.
- 성공/실패 판단 지표
  - `robot_body_contact` 감소
  - next-intercept reachable rate 증가
  - viewer에서 contact 후 arm recovery가 빨라지는지

## 9. 학습 전략 재검토

### 9.1 PPO는 당장 유지

현재 정보만으로는 PPO가 본질적으로 틀렸다고 보기 어렵다. 더 큰 문제는 task contract와 phase 정보 부족이다.

### 9.2 checkpoint selection은 계속 유지

best checkpoint 중심 선택은 이미 효과가 확인됐다. 이 부분은 유지해야 한다.

### 9.3 curriculum은 1순위가 아니다

현재는 과제가 너무 어려워서가 아니라, 다음 공 feasibility가 task로 명확히 표현되지 않는 문제가 더 크다. contract가 정리되기 전 curriculum 확대는 원인 분리를 다시 흐릴 수 있다.

### 9.4 imitation/bootstrap은 heuristic baseline 결과 다음 단계

heuristic controller가 2~3회 keep-up을 만들 수 있으면, 그 rollout은 PPO warm-start나 behavior cloning seed로 쓸 가치가 있다. 반대로 heuristic도 실패하면 imitation보다 env/control 수정이 먼저다.

### 9.5 two-stage policy는 보류

prepare/strike와 return/recovery를 분리하는 아이디어는 가능하지만, 지금은 phase 정보와 observation contract를 먼저 정리하는 편이 작다. 그 이후에도 여전히 충돌하면 검토할 수 있다.

### 9.6 SAC 전환은 마지막 후보

지금 바로 알고리즘을 바꾸면 잘못 정의된 목표를 다른 알고리즘으로 다시 최적화할 위험이 크다. task contract 정리 뒤에도 PPO가 명확히 막히면 그때 검토한다.

## 10. 다음 구현 우선순위 3개

### 10.1 우선순위 1: scripted/heuristic keep-up diagnostic baseline

- 목적
  - 환경/물리/제어가 반복 keep-up을 허용하는지 먼저 확인
- 완료 조건
  - deterministic 또는 좁은 reset에서 2 useful bounce 이상을 재현

### 10.2 우선순위 2: next-intercept/recovery observation 추가

- 목적
  - policy가 다음 공 준비에 필요한 정보를 직접 보게 만들기
- 최소 후보
  - phase flag
  - time since contact
  - next descending intercept xy/time
  - reachable flag 또는 recovery distance

### 10.3 우선순위 3: event-level return/recovery reward 한 개만 승격

- 목적
  - 맞힌 뒤 다음에도 칠 수 있는 공을 만들게 하기
- 조건
  - heuristic baseline 또는 observation 보강 후에도 first-contact/episode-level metric 정렬이 나아질 것

## 11. 1M 학습을 다시 돌려도 되는 조건

아래를 만족하기 전에는 1M 학습을 다시 돌리지 않는 편이 맞다.

1. heuristic baseline이 최소한 일부 reset 조건에서 2 useful bounce 이상을 만든다.
2. policy observation에 next-intercept/recovery 정보가 들어가고, phase 구분이 더 명확해진다.
3. 50k~100k 짧은 PPO run에서 아래가 동시에 개선된다.
   - `one_or_more_useful_bounce_rate`
   - `two_or_more_useful_bounce_rate`
   - `useful_contact_next_intercept_reachable_rate`
4. 같은 run에서 아래가 악화되지 않는다.
   - `ball_out_of_bounds`
   - `robot_body_contact`
   - first-contact easy-next-ball gap의 역방향성
5. best checkpoint가 final model보다 일관되게 낫거나 최소 동급이다.

## 12. 바로 돌릴 실험 command

이번 단계에서는 바로 새 학습을 돌리지 않는다. 대신 아래 read-only 진단이 우선이다.

### 12.1 active candidate viewer 확인

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --run-name clean_tnp_return_assist \
  --run-version v1 \
  --best-model \
  --episodes 5
```

### 12.2 active candidate easy-next-ball 재분석

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/clean_tnp_return_assist_v1/clean_tnp_return_assist_v1_best_model.zip \
  --episodes 50 \
  --analysis-name clean_tnp_return_assist_v1_rethink50 \
  --compare-apex-targets
```

### 12.3 current physics/reset viewer sanity

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --mode zero_action \
  --episodes 3
```

## 13. 아직 하지 말아야 할 것

- assist weight를 `0.4/0.5/0.6` 식으로 계속 찍는 실험
- 새 reward term을 붙이고 곧바로 학습 시작
- `vx` penalty 재튜닝
- `position_tilt` 재도입
- SAC로 즉시 전환
- 1M 장기 학습 재개

## 14. 최종 결론

지금 필요한 것은 `더 세게 치게 만들기`나 `vx를 조금 줄이기`가 아니다.

필요한 것은 아래 세 가지다.

1. 반복 keep-up을 prepare / strike / return shaping / recovery로 분리해 책임 경계를 다시 세우기
2. scripted heuristic으로 이 환경이 실제로 2~3회 keep-up을 허용하는지 먼저 진단하기
3. easy-next-ball을 anchor 점 metric이 아니라 다음 strike 가능한 workspace contract로 승격시키기

이 세 가지가 정리된 뒤에야 reward나 policy를 건드리는 것이 맞다.
