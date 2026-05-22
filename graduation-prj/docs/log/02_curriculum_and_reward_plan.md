# curriculum 및 reward 설계 메모

## 1. 왜 curriculum이 필요한가

현재 task에는 동시에 풀어야 할 요구가 많다.

- 공을 라켓 중심 근처로 맞추기
- 공을 위로 충분히 올리기
- 라켓 면을 수평에 가깝게 유지하기
- 관절을 과도하게 흔들지 않기
- 움직임이 뚝뚝 끊기지 않게 만들기

이걸 처음부터 한 reward 안에 모두 강하게 넣으면, 정책은 `무엇부터 맞춰야 하는지`를 배우기 어려워진다.

그래서 현재 방향은 `처음에는 기본 hit`, `그다음에는 안정성`, `마지막에는 자세와 부드러움` 순서로 나누는 것이다.

## 2. 현재 curriculum stage

### 2.1 bootstrap

목표:

- 일단 공을 위로 치는 최소한의 패턴 만들기

특징:

- reset randomization 제거
- success threshold 완화
- tilt/joint/smoothness 항목 off

### 2.2 stabilize

목표:

- 어느 정도 맞추는 동작이 나온 뒤, 반복 성공률을 높이기

특징:

- 약한 reset randomization 복귀
- 자세 유지와 관절/부드러움 penalty를 약하게 켬

### 2.3 refine

목표:

- 실제로 더 자연스럽고 안정적인 keep-up 행동 만들기

특징:

- full threshold 복귀
- action filter 사용
- tilt/joint/smoothness penalty 강화

## 3. 관절 보상을 왜 late-stage로 미뤘는가

관절 관련 penalty를 너무 일찍 넣으면, policy가 공을 치기 전에 먼저 `안 움직이려는 방향`으로 수렴할 수 있다.

따라서 현재 원칙은 아래다.

- stage 1: 공을 올릴 수 있는가
- stage 2: 그 동작을 반복할 수 있는가
- stage 3: 그 동작이 더 수평하고 덜 튀는가

즉, 관절 penalty는 `기본 동작 형성 이후`에 들어가야 한다.

## 4. 부자연스러운 뚝뚝 끊김을 줄이는 방법

현재 코드에 반영된 방법:

- `action_smoothness_term`
- `action_filter_alpha`
- `joint_motion_term`

추가로 이후 검토 가능한 방법:

- PPO action std 스케줄 조정
- control_dt / n_substeps 재조정
- jerk penalty를 joint acceleration 기준으로 추가
- policy network를 더 작은 action scale로 재학습

## 5. 다음 실험 권장 순서

1. 새 run name으로 curriculum 기본값 유지한 채 재학습
2. 150k~200k마다 렌더로 upward strike 패턴 확인
3. strike는 있는데 motion이 거칠면 smoothness 계열 weight만 미세 조정
4. strike 자체가 약하면 reward보다 controller/threshold를 먼저 다시 확인

## 6. 현재 코드 기준 curriculum 역할

현재 `keepup_v1` curriculum은 단순한 옵션이 아니라 실제로 아래를 stage별로 바꾼다.

`bootstrap`

- `reset_ball_height_range = 0.0`
- `reset_xy_range = 0.0`
- `reset_velocity_xy_range = 0.0`
- `reset_velocity_z_range = (0.0, 0.0)`
- `success_velocity_threshold = 0.35`
- `tracking_assist_weight = 0.35`
- `tracking_alignment_reward_weight = 2.0`
- `contact_centering_reward_weight = 0.5`
- tilt/joint/smoothness regularization off

`stabilize`

- 약한 randomization 복귀
- `success_velocity_threshold = 0.45`
- `tracking_assist_weight = 0.20`
- `tracking_alignment_reward_weight = 1.5`
- `contact_centering_reward_weight = 1.0`
- tilt/joint/smoothness penalty 약하게 on

`refine`

- full randomization 복귀
- `success_velocity_threshold = 0.5`
- `tracking_assist_weight = 0.10`
- `tracking_alignment_reward_weight = 0.75`
- `contact_centering_reward_weight = 1.25`
- tilt/joint/smoothness penalty 강화

즉 지금 curriculum의 역할은 아래 두 줄로 요약할 수 있다.

- 초반에는 heuristic과 쉬운 reset으로 `공을 치는 패턴`부터 만들고
- 후반에는 assist를 줄이고 penalty를 키워 `정교한 반복 keep-up` 쪽으로 이동시킨다.

## 7. curriculum이 필요한가

현재 task에서는 `필요하다`는 쪽이 맞다.

이유:

1. 목표가 단순 contact가 아니라 `정렬`, `위로 올려치기`, `중앙 적중`, `자세 유지`를 동시에 요구한다.
2. action이 `position_tilt`로 확장되면 exploration 난도가 더 올라간다.
3. 실제 40k run들에서 후반부 `robot_body_contact`가 커지는 경향이 보였다.

즉 curriculum이 없으면 정책이 처음부터 너무 많은 것을 동시에 맞추려다가, 아예 유의미한 strike 패턴을 만들기 전에 무너질 가능성이 높다.

## 8. 현재 판단

최근 실험 기준으로 보면:

- reward-only shaping만으로는 부족했다.
- predicted-intercept 정렬과 중앙 적중 shaping은 필요한 보강이었다.
- `position_tilt` action 추가는 구조적으로 맞는 방향이지만, `40k`에서는 아직 `v6`를 넘지 못했다.

### 8.1 `ppo_keepup_v10` 회귀 점검

`ppo_keepup_v10` 100k run은 이전 `v9`보다 명확하게 악화됐다.

- `v9`: 평균 bounce `2.2918`, `episodes_with_bounces = 2734`, contact `10069`
- `v10`: 평균 bounce `0.0267`, `episodes_with_bounces = 142`, contact `1195`
- 특히 `robot_body_contact = 3287/5318`로 크게 증가했다.

이번 회귀에서 가장 직접적인 차이는 `episode step 1`부터 tilt가 항상 켜졌다는 점이다.

- `v9` step1 nonzero tilt ratio: `0.0`
- `v10` step1 nonzero tilt ratio: `1.0`

즉 `position_tilt` 자체가 문제라기보다, reset 직후부터 tilt 관련 개입이 들어가면서 정책이 너무 이른 시점에 팔을 흔들도록 학습된 쪽으로 보는 편이 맞다.

그래서 현재 코드는 아래처럼 다시 단순화했다.

- 기본 `tilt_tracking_assist_weight = 0.0`
- curriculum 모든 stage에서도 tilt heuristic assist 제거
- 공이 초기 spawn 높이에서 충분히 내려오기 전에는 tilt residual action 자체를 막음

정리하면, 현재 우선순위는 `tilt를 더 세게 도와주기`가 아니라 `tilt는 늦게 열고, 위치 추종/타이밍부터 다시 안정화`하는 쪽이다.

따라서 현재 우선순위는 아래다.

1. `position_tilt` 모드를 더 긴 budget으로 돌려 보기
2. bootstrap assist를 더 오래 유지하거나 adaptive curriculum으로 바꾸기
3. 그 다음에야 reward 미세조정을 다시 보기
