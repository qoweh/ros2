# stable keep-up 기준 재검토 보고서

## 1. 이 문서의 목적

이 문서는 현재 `pingpong_rl`을 `경기형 탁구`가 아니라,

- 탁구채로 공을 위 방향으로 반복적으로 튕기고
- 너무 높거나 옆으로 날리지 않으며
- 다음 bounce를 다시 만들기 쉬운 상태를 유지하는

`stable repeated keep-up` 과제로 다시 해석한 정리본이다.

중요한 전제:

- 지금 목표는 forward shot, target return, rally 전략이 아니다.
- 지금 목표는 `안정적으로 반복 가능한 bounce dynamics`를 만드는 것이다.
- 따라서 핵심 평가는 `위로 한번 뜨는가`보다 `반복 가능한 attractor를 형성하는가`에 있어야 한다.

## 2. 현재 로그가 보여주는 것

현재 `ppo_keepup_v1` 로그 기준으로 보면 정책은 이미 `공을 전혀 못 맞추는 상태`는 아니다.

관측된 사실:

- 총 `29,181` episode 중 `28,244` episode에서 적어도 한 번 이상 upward bounce가 발생했다.
- 총 contact 수는 `163,607`이다.
- 하지만 episode당 median `successful_bounce_count`는 `2`에 머문다.
- p90 `successful_bounce_count`는 `7`, 최대는 `19`다.
- episode 길이 median은 `68` step, contact_count median은 `4`, first_contact_step median은 `16`이다.

즉, `한두 번은 튕기지만 장기 안정성은 낮다`는 해석이 맞다.

실패 원인 분포도 이 해석과 일치한다.

- `ball_out_of_bounds`: `16,504`
- `robot_body_contact`: `8,374`
- `ball_speed_limit`: `2,292`
- `floor_contact`: `2,011`

핵심은 `upward bounce 부재`보다 `제어되지 않은 rebound`다.

contact 시 공 속도 분포도 같은 결론을 준다.

- contact의 약 `95.7%`에서 `ball_velocity_z > 0`이다.
- contact 시 lateral speed median은 약 `0.242 m/s`다.
- lateral speed p90은 약 `0.599 m/s`다.

즉 공은 위로는 잘 뜨지만, 옆으로 흐르는 에너지도 꽤 크다. 현재 병목은 `위로 보내기 실패`가 아니라 `bounded repeatability 실패`다.

## 3. Task 정의 재검토

현재 env의 성공 개념은 사실상 `stable keep-up`보다 `useful upward bounce`에 더 가깝다.

현재 구조에서 중심인 것:

- contact 순간 upward racket motion
- contact 이후 positive `ball_velocity_z`
- success threshold를 넘는 upward bounce

이 정의는 bootstrap에는 유리하지만, stable keep-up 목표와는 다르다.

stable keep-up에서 더 중요한 것은 아래다.

- 원하는 높이 대역 안에서 apex가 유지되는가
- bounce마다 apex 편차가 작아지는가
- lateral drift가 누적되지 않는가
- 다음 contact를 만들기 쉬운 상태로 rebound가 형성되는가
- 같은 패턴이 여러 번 반복 가능한가

즉 objective는 `높이 뜨는 것 자체`가 아니라 `반복 가능한 bounce state를 만드는 것`이어야 한다.

현재 코드에는 이 목표와 어긋나는 지점이 하나 더 있다.

- 기본 spawn 높이는 `0.50m`
- 기본 `target_ball_height`는 `0.30m`
- 현재 목표 높이는 `spawn + target = 0.80m above racket`

이 의미는 `초기 드롭 높이 근처를 안정적으로 유지`가 아니라 `초기 드롭보다 더 높게 띄우기`다.

keep-up 기준으로 보면 이건 과하게 공격적인 목표일 수 있다. stable keep-up의 주 목표는 `더 높게`가 아니라 `같은 패턴으로 다시 칠 수 있게`다.

## 4. 현재 로깅의 공백

현재 로깅은 contact/failure 분석에는 충분하지만, stable keep-up 평가지표에는 아직 공백이 있다.

현재 바로 볼 수 있는 것:

- contact 순간 ball/racket velocity
- episode 길이
- bounce 수
- failure type

현재 바로 보기 어려운 것:

- bounce별 apex height
- apex variance
- bounce-to-bounce cadence consistency
- XY drift trajectory
- contact 이후 다음 apex까지의 lateral displacement

env 자체는 `ball_height_above_racket` 같은 값을 계산하지만, PPO CSV summary에는 이 값들이 직접 저장되지 않는다.

그래서 지금은 `안정성`을 판단해야 하는데 로그는 여전히 `contact`와 `failure` 쪽에 더 치우쳐 있다. stable keep-up로 넘어가려면 평가 지표도 같이 바뀌어야 한다.

## 5. Current action space 한계 검토

### 5.1 현재 구조가 무엇을 제어하는가

현재 PPO 정책은 사실상 아래를 제어한다.

- `Delta x, y, z` 기반 EE target position

그리고 controller는 그 target을 Jacobian 기반 site IK로 joint target으로 바꾼다.

이 구조의 장점:

- 저차원 action이라 PPO가 시작하기 쉽다.
- fixed-orientation keep-up의 초기 signs of life를 만들기 좋다.
- raw 7-DoF action보다 탐색 난이도가 낮다.

즉 `첫 bounce 만들기`에는 꽤 합리적이다.

### 5.2 stable keep-up 관점에서 부족한 점

하지만 stable repeated bouncing에서는 중요한 것이 단순 위치가 아니다.

- contact 순간 racket velocity
- impact timing
- outgoing ball velocity shaping
- orientation에 따른 lateral correction
- 너무 강한 타격을 줄이는 damping 성격

현재 EE delta only 구조는 `라켓이 어디 있는가`는 표현하지만, `공을 어떤 속도로 어떻게 되튕길 것인가`는 직접 표현하지 못한다.

특히 limitation은 아래와 같다.

- impact velocity가 position tracking error에 간접적으로만 결정된다.
- timing이 policy 출력이 아니라 controller response에 크게 의존한다.
- orientation은 action에도 없고 observation에도 없다.
- compliance나 contact-aware modulation이 없다.
- 결과적으로 policy가 `안정적 rebound shaping`보다 `공을 따라가 다시 맞추기` 쪽으로 기울기 쉽다.

### 5.3 각 대안의 의미

`EE delta position control`

- bootstrap에는 가장 좋다.
- 하지만 robust keep-up의 최종 표현력은 제한적일 가능성이 크다.

`joint velocity control`

- impact timing과 타격 속도를 더 직접 다룰 수 있다.
- 그러나 PPO가 처음부터 raw joint velocity를 바로 배우기에는 난도가 더 높다.
- 따라서 `다음 단계의 LLC`나 structured baseline에는 적합하지만, 지금 당장 첫 카드로 쓰기엔 부담이 있다.

`orientation 포함 control`

- keep-up에서는 full joint control보다 이쪽이 더 높은 우선순위일 수 있다.
- 작은 pitch/roll만 있어도 lateral drift correction 능력이 크게 달라질 수 있기 때문이다.

`impact velocity shaping`

- 현재 구조에서 가장 빠진 요소에 가깝다.
- 예를 들어 `desired racket vertical velocity`, `strike primitive amplitude`, `contact-time residual` 같은 형태가 stable keep-up에는 더 직접적이다.

`compliant/contact-aware control`

- 과도한 rebound를 줄이는 데는 도움이 될 수 있다.
- 하지만 지금 단계에서 1순위 병목은 아니다.

결론적으로, 현재 action space는 `초기 bounce 학습`에는 충분할 수 있지만 `장기 안정 keep-up`에는 한계가 보인다. 특히 `orientation + impact velocity shaping`이 없는 점이 중요하다.

## 6. Reward 구조 재검토

현재 reward는 stable keep-up보다 `활성 upward hit`를 강하게 밀고 있다.

episode 로그를 보면 median 기준으로 아래처럼 reward 비중이 형성된다.

- `reward_success_sum`: `30.0`
- `reward_active_hit_sum`: `13.37`
- `reward_contact_sum`: `2.5`
- `reward_height_sum`: `-4.70`
- `reward_distance_sum`: `-4.19`

즉 reward mass의 중심은

- upward bounce가 있었는가
- active upward hit였는가
- contact가 있었는가

에 있다.

반면 stable keep-up에 필요한 항목은 직접적으로 거의 들어가 있지 않다.

- apex가 target band를 유지하는가
- 너무 높게 뜨지 않는가
- lateral rebound가 작아지는가
- bounce-to-bounce variance가 줄어드는가
- 에너지를 적당히만 주고 있는가

현재 reward의 구체적 문제는 세 가지다.

### 6.1 upward local optimum 유도

`active_hit_term`, `lift_term`, `success_bonus`는 모두 positive vertical bounce를 강하게 밀어준다.

그래서 정책은 `적당히 안정적으로 유지`보다 `강하게 위로 올리는 것`에 먼저 수렴하기 쉽다.

### 6.2 height term의 의미가 stable keep-up과 다름

현재 height term은 `spawn + target_ball_height`를 향한다.

즉 stable apex band라기보다 `spawn보다 더 높은 bounce`를 유도한다.

keep-up 목표에서는 `이번 bounce가 다음 bounce를 쉽게 만드는가`가 중요하지, `무조건 더 높이`가 중요하지는 않다.

### 6.3 distance term이 reactive alignment에 머뭄

현재 distance term은 현재 시점의 XY alignment error를 벌점으로 준다.

이건 `지금 공이 라켓 중심에서 멀다`는 사실은 잡지만,

- contact 이후 lateral speed가 큰지
- 다음 apex에서 얼마나 drift하는지
- bounce마다 drift가 누적되는지

는 직접 잡지 못한다.

즉 `공을 따라가는 것`과 `공을 안정적으로 다시 중앙으로 보내는 것`을 구분하지 못한다.

### 6.4 pre-contact shaping의 정렬 문제

현재 `_strike_zone_score()`는 descending ball에 대해 `target_ball_height_above_racket`를 참조한다.

그런데 기본값에서는 이 target이 `0.80m above racket`이므로, 첫 descending ball의 높이와는 잘 맞지 않는다.

그 결과 pre-contact upward shaping이 초기 bounce에서 약하거나 거의 비활성일 가능성이 크다.

즉 reward는 한편으로는 `더 높게`를 밀고, 다른 한편으로는 `언제 칠 준비를 해야 하는가`에는 충분히 정렬돼 있지 않을 수 있다.

## 7. stable keep-up 기준 reward가 더 중요하게 봐야 할 것

구조를 다시 세운다면, keep-up에서는 아래 축이 더 중심이어야 한다.

- target apex height band 유지
- excessive apex penalty
- contact 후 lateral speed penalty
- bounce-to-bounce apex variance penalty
- repeatable cadence 유지
- controllable rebound 보상

여기서 핵심은 `큰 upward velocity`가 아니라 `bounded, repeatable, low-drift rebound`다.

즉 reward의 질문이 `높이 뜨는가`에서 `다음 bounce를 쉽게 만드는가`로 바뀌어야 한다.

## 8. Reset distribution 검토

keep-up에서는 복잡한 rally distribution이 필요 없다. 하지만 `거의 동일한 조건`만 계속 보면 overfit이 생길 수 있다.

현재 방향 자체는 크게 틀리지 않았다.

- 작은 XY variation
- 작은 lateral velocity variation
- near-vertical drop

이 정도면 keep-up bootstrap에는 충분하다.

다만 현재에서 보완이 필요한 점은 아래다.

### 8.1 bootstrap exact drop은 acquisition용으로만 써야 한다

초기 `정확히 같은 드롭`은 first contact를 배우는 데는 유리하다. 하지만 그 상태가 너무 오래 유지되면 policy는 하나의 cadence와 하나의 공간 위치에만 맞는 해를 학습할 수 있다.

### 8.2 height randomization이 빠져 있다

stable keep-up에서는 XY만큼이나 `cadence`가 중요하다. cadence는 drop height 변화에 민감하다.

그런데 현재 reset randomization은 XY와 velocity 쪽에 비해 `spawn height` variation이 없다.

이건 keep-up robustness 입장에서는 아쉬운 부분이다.

### 8.3 적절한 randomization 수준

keep-up 기준의 적절한 범위는 대략 아래 정도가 맞다.

`stage 1: acquisition`

- XY: 거의 0 또는 매우 작게
- height: 아주 작은 변동
- lateral velocity: 거의 0
- vertical velocity: 거의 0

`stage 2: stability`

- XY: `5~15mm`
- spawn height: `수 cm` 수준 randomization
- lateral velocity: 작은 범위에서만 허용

`stage 3: robustness`

- XY: `1~3cm`
- spawn height: 더 넓은 band
- lateral velocity: 작지만 무시할 수는 없는 수준

중요한 것은 `opponent shot distribution`이 아니라 `같은 attractor를 여러 근방 상태에서 유지할 수 있는가`다.

## 9. 현재 병목 우선순위

현재 문제의 우선순위는 아래처럼 보는 것이 맞다.

### 9.1 1순위: task / reward definition

지금 env는 `stable repeated keep-up`보다 `upward bounce`에 더 정렬돼 있다.

특히 `spawn + target_ball_height` semantics는 현재 keep-up 목표와 가장 크게 충돌하는 지점이다.

### 9.2 2순위: impact controllability

현재 action space는 contact timing과 outgoing rebound shaping을 간접적으로만 다룬다.

장기 안정성은 여기서 막힐 가능성이 크다.

### 9.3 3순위: 평가 지표와 로깅

지금은 stable keep-up을 얘기하면서도 apex/drift/consistency를 직접 측정하지 못한다.

측정이 안 되면 reward 재설계도 감으로 가기 쉽다.

### 9.4 4순위: reset distribution

현재 reset 방향은 keep-up에 맞지만, height variation과 stage 전환 강도는 더 검토할 필요가 있다.

### 9.5 5순위: 알고리즘

PPO에서 SAC로 바꾸거나 BGS/ES 계열을 고려하는 것은 그다음 문제다.

SAC는 연속 제어에서 도움이 될 수 있지만, 현재 root cause를 바로 없애주지는 않는다.

BGS는 논문에서 smoothness와 sim-to-real에 강점이 있었지만, 현재 과제는 simulation-only keep-up이다. 지금 병목은 smoothness optimizer 선택보다 `무엇을 안정화할 것인가` 정의 쪽에 더 가깝다.

즉, 지금은 `알고리즘 변경`보다 `구조 변경`이 우선이다.

## 10. 최종 결론

현재 PPO 정책은 이미 `공을 위로 치는 것` 자체는 어느 정도 배우고 있다.

하지만 아직 배우지 못한 것은 `반복 가능한 bounded keep-up regime`이다.

가장 중요한 결론 세 가지:

1. 현재 문제는 `upward bounce 실패`가 아니라 `stable repeated bouncing 실패`다.
2. 현재 env는 keep-up 안정성보다 upward bounce를 더 강하게 최적화하고 있다.
3. 알고리즘보다 먼저 `task semantics`, `reward target`, `impact controllability`, `stability logging`을 다시 정렬해야 한다.

이 관점에서 보면, 지금 가장 강한 의심 지점은 아래 두 가지다.

- `target_ball_height`가 stable apex band가 아니라 `spawn보다 더 높게 띄우기`로 해석되는 점
- current action space가 rebound shaping을 직접 표현하지 못하는 점

따라서 다음 단계의 기준 질문은 `어떤 알고리즘이 더 센가`가 아니라,

`정책이 안정적인 bounce attractor를 배우도록 문제를 다시 정의했는가`

가 되어야 한다.