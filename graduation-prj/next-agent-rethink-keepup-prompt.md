# 다음 에이전트 프롬프트: keep-up 완성을 위한 방향 재검토

너는 `pingpong_rl2` 프로젝트를 이어받는 에이전트다.

작업 루트:

```bash
/Users/pilt/project-collection/ros2/graduation-prj
```

실제 패키지:

```bash
/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
```

## 1. 최종 목표

Franka Panda 로봇팔이 탁구채로 탁구공을 계속 위로 튕기는 강화학습 프로젝트를 완성한다.

성공은 다음을 모두 만족해야 한다.

- 공을 라켓으로 친다.
- 공을 위로 올린다.
- 공이 다시 라켓이 칠 수 있는 strike zone으로 돌아온다.
- 이 과정이 여러 번 반복된다.

즉 목표는 단순 `contact`, 단순 `upward bounce`, 단순 `vx 감소`가 아니다.

목표는:

> 다음 공도 치기 쉬운 상태를 만드는 return trajectory를 학습하는 것.

## 2. 먼저 읽을 문서

반드시 읽어라.

- `agent-answer.md`
- `agent-todo.md`
- `pingpong_rl2/docs/report/08_easy_next_ball_completion_plan.md`
- `pingpong_rl2/docs/report/07_reward_policy_cleanup_plan.md`
- `pingpong_rl2/README.md`
- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
- `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py`
- `pingpong_rl2/scripts/run_ppo_learning.py`
- `pingpong_rl2/scripts/run_ppo_rebound_analysis.py`
- `pingpong_rl2/scripts/run_viewer.py`

## 3. 현재 판단

값을 조금씩 바꾸는 방식만으로는 최종 목표까지 가기 어렵다.

이유:

- 현재 최고 후보도 `mean_useful_bounces=0.40` 수준이다.
- `two_or_more_useful_bounce_rate`가 매우 낮다.
- `ball_out_of_bounds`가 여전히 많다.
- assist weight `0.5 -> 0.6` 같은 미세조정은 일부 aggregate는 비슷해도 useful-contact 기준 next-ball quality가 악화됐다.
- reward-side projected apex XY term도 regression이었다.

따라서 지금은 `weight 0.4`, `weight 0.6`, `vx penalty`를 계속 찍어보는 단계가 아니다.

현재 해야 할 일은:

1. keep-up 문제를 tracking/strike/return/recovery phase로 다시 나눈다.
2. RL policy가 무엇을 배우고, controller/assist가 무엇을 책임지는지 다시 정한다.
3. “치기 쉬운 다음 공”을 metric이 아니라 학습 가능한 task contract로 만든다.

## 4. 중요한 의심 지점

아래를 반드시 검토해라.

### 4.1 현재 MDP가 반복 keep-up을 배우기에 충분한가?

확인할 것:

- observation에 phase 정보가 충분한가?
- contact 이후 recovery/next-intercept 정보가 policy에 보이는가?
- `successful_bounce_count`, last contact info, next intercept feasibility 같은 정보가 observation에 필요한가?
- action이 residual target delta인데, 반복 rally를 만들기에는 너무 간접적인가?
- `position_strike`의 hand-coded target generation이 policy 학습을 도와주는지, 오히려 책임 경계를 흐리는지?

### 4.2 reward가 최종 목표와 직접 연결되어 있는가?

현재 reward는 주로:

- 공 아래로 가기
- upward useful contact
- apex height
- failure penalty

에 가깝다.

하지만 최종 목표에는 아래가 빠져 있다.

- 다음 strike 가능성
- recovery 가능성
- 다음 intercept까지 충분한 시간
- 다음 intercept 위치가 reachable zone 안인지
- contact quality가 안정적인지

단, 바로 reward를 추가하지 말고 먼저 metric과 causality를 확인해라.

### 4.3 controller/physics가 문제일 가능성

RL 튜닝 전에 아래 sanity를 확인해라.

- scripted controller가 2회 이상 keep-up을 만들 수 있는가?
- 라켓 normal, pitch sign, contact stiffness/friction이 물리적으로 말이 되는가?
- 탁구공이 라켓에 맞은 뒤 너무 빠르거나 불안정하게 튀는 MuJoCo contact 설정 문제는 없는가?
- current racket face orientation이 keep-up에 유리한가?
- post-contact target 이동이 실제 라켓을 다음 intercept로 충분히 빨리 보내는가?

만약 scripted 또는 heuristic controller도 반복 keep-up을 못 만들면, PPO reward 튜닝 전에 환경/제어/물리 문제를 먼저 봐야 한다.

## 5. 이번 작업에서 하지 말 것

- 바로 1M 학습 돌리지 마라.
- assist weight만 0.4/0.5/0.6 계속 찍지 마라.
- reward term을 하나 더 붙이고 학습부터 돌리지 마라.
- `vx` penalty를 다시 튜닝하지 마라.
- `position_tilt`를 성급히 재도입하지 마라.
- SAC로 바로 갈아타지 마라.

## 6. 이번 작업에서 해야 할 것

### Step 1. 현재 구조를 phase 관점으로 재정리

`keepup_env.py`를 읽고 현재 구조를 아래 네 phase로 나눠 문서화해라.

1. prepare: 공 아래로 이동
2. strike: upward contact
3. return shaping: 공을 다음 strike zone으로 보냄
4. recovery: 라켓/팔을 다음 intercept 가능한 자세로 돌림

각 phase에 대해 정리:

- 현재 observation
- 현재 action/control
- 현재 reward
- 현재 info/logging
- 부족한 점

### Step 2. scripted/heuristic sanity controller 필요 여부 판단

다음 질문에 답해라.

> 현재 환경과 controller에서 사람이 짠 단순 heuristic으로 2~3회 keep-up이 가능한가?

가능하지 않다면, PPO가 배우기 전에 task가 너무 불안정한 것이다.

필요하면 다음 에이전트가 구현할 후보를 문서화해라.

- predicted intercept로 XY 이동
- 공이 내려올 때만 upward strike
- contact 후 next intercept로 이동
- face normal은 fixed negative pitch 또는 current best ramp
- 공 속도가 너무 커지지 않게 z strike velocity 제한

목표:

- 완벽한 controller가 아니라, 이 환경에서 반복 keep-up이 물리적으로 가능한지 확인하는 diagnostic baseline.

### Step 3. easy-next-ball을 metric에서 task contract로 승격할 설계

현재 `08_easy_next_ball_completion_plan.md`에는 metric이 있다.

이제 다음을 검토해라.

- 이 metric이 실제 two-or-more useful bounce와 얼마나 상관 있는가?
- first contact 기준으로 역방향이면 왜 그런가?
- metric target이 controller anchor가 맞는가?
- target을 fixed home strike zone, current reachable region, predicted feasible region 중 무엇으로 해야 하는가?

중요:

“로봇 중심”이 아니라 “다음 strike가 가능한 라켓 workspace”가 target이어야 한다.

### Step 4. observation 보강 후보 설계

반복 keep-up에는 policy가 다음 정보를 알아야 할 수 있다.

검토 후보:

- phase flag: descending/ascending/post-contact/recovery
- last contact step 또는 time since contact
- successful bounce count clipped
- predicted next intercept xy/time
- next intercept reachable flag
- racket velocity
- racket face normal
- relative ball-racket velocity
- target position error

바로 다 넣지 말고, 어떤 정보가 현재 이미 있는지와 없는지를 표로 정리해라.

### Step 5. reward 재설계 후보를 “작게” 제안

새 reward를 구현하기 전에 설계만 해라.

후보는 세 개 이하로 제한한다.

권장 후보:

1. next-intercept reachable bonus
   - contact 이후 다음 strike plane intercept가 reachable radius 안이면 작은 bonus
2. easy-next-ball score
   - time, xy error, speed를 합친 작은 event reward
3. recovery readiness reward
   - post-contact 이후 라켓이 다음 intercept를 칠 수 있는 위치/속도에 가까워지는지

각 reward에 대해:

- 적용 시점
- weight 초기값
- exploit 가능성
- 기존 reward와 충돌 가능성
- 성공/실패 판단 지표

를 적어라.

### Step 6. 학습 전략 재검토

PPO를 유지하되, 다음 중 무엇이 필요한지 판단해라.

- curriculum
  - 처음에는 쉬운 1-bounce, 그 다음 2-bounce, 그 다음 reset range 확대
- checkpoint selection
  - final model이 아니라 best checkpoint 중심
- imitation/bootstrap
  - scripted controller rollout으로 policy warm-start 또는 behavior cloning
- two-stage policy
  - prepare/strike와 return/recovery를 분리
- algorithm 변경
  - SAC는 마지막 후보. 환경 목표가 정리된 뒤 검토

## 7. 산출물

코드 구현보다 먼저 아래 문서를 만들어라.

추천 파일:

```text
pingpong_rl2/docs/report/09_keepup_task_rethink_plan.md
```

문서에 반드시 포함:

1. 현재 미세 튜닝이 부족한 이유
2. keep-up task phase 분해
3. 현재 observation/action/reward가 각 phase를 커버하는지
4. “치기 쉬운 다음 공”의 정확한 정의
5. scripted sanity controller 필요 여부
6. observation 보강 후보
7. reward 보강 후보
8. 학습 전략 후보
9. 다음에 실제로 구현할 우선순위 3개
10. 1M 학습을 다시 돌려도 되는 조건

## 8. 다음 구현 우선순위 초안

문서 검토 후 구현 우선순위는 아마 아래가 될 가능성이 높다.

1. scripted/heuristic keep-up diagnostic baseline
   - 목적: 환경/물리/제어가 반복 keep-up을 허용하는지 확인
2. next-intercept/recovery observation 추가
   - 목적: policy가 다음 공을 준비할 정보를 갖게 하기
3. easy-next-ball event reward 또는 reachable bonus
   - 목적: 맞힌 뒤 다음에도 칠 수 있는 공을 만들게 하기

단, 실제 우선순위는 문서 검토 후 확정해라.

## 9. 최종 보고 형식

작업 후 아래 형식으로 보고해라.

```text
1. 지금 방향이 왜 미세 튜닝만으로는 부족한지
2. keep-up task를 phase별로 나눈 결과
3. 현재 코드에서 빠진 핵심 정보/보상/제어
4. scripted sanity controller가 필요한지
5. 다음 구현 우선순위 3개 + keep-up 완성을 위한 최종 작업과정들 후보(가늠이 안 잡혀도 일단 작성해보기)
6. 바로 돌릴 실험 command
7. 아직 하지 말아야 할 것
```

## 10. 핵심 질문

모든 판단은 이 질문으로 돌아와야 한다.

> 이 변경은 공을 한 번 더 맞히게 하는가, 아니면 계속 칠 수 있는 다음 공을 만들게 하는가?

후자에 답하지 못하면 최종 프로젝트 목적과 맞지 않는다.
