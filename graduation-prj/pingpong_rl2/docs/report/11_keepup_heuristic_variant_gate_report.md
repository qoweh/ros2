# pingpong_rl2 keep-up heuristic variant gate report

## 1. 목적

이번 단계의 목적은 아래 gate를 실제로 통과하는지 확인하는 것이다.

1. pitch/contact-direction heuristic variant 2~3개 비교
2. 50~100 episode 분석
3. `2+ useful bounce` 발생 여부 확인
4. 발생하면 `phase_contract_candidate` PPO 50k~100k 진행
5. checkpoint 분석
6. 필요시 bootstrap 판단

결론부터 쓰면, 이번 gate는 **PPO 진행 전 단계에서 멈추는 것이 맞다**.

이유는 best heuristic variant도 100 episodes에서 `max_useful_bounces=1`이었기 때문이다.

## 2. 실행한 heuristic variant

비교 variant는 3개로 제한했다.

### 2.1 baseline_ramp

- timed negative pitch ramp
- `strike_tilt_ramp_pitch=-0.03`
- `post_contact_return_assist_weight=0.5`

### 2.2 strong_ramp

- stronger timed negative pitch ramp
- `strike_tilt_ramp_pitch=-0.05`
- 나머지는 baseline과 동일

### 2.3 fixed_inward

- fixed inward pitch bias
- `initial_target_tilt=(-0.03, 0.0)`
- timed ramp는 끔
- 나머지는 baseline과 동일

## 3. 50-episode 비교 결과

### 3.1 baseline_ramp

- `mean_useful_bounces=0.42`
- `max_useful_bounces=1`
- `two_or_more_useful_bounce_rate=0.00`
- `useful_contact_next_intercept_reachable_rate=0.190`
- `useful_contact_mean_easy_next_ball_score=-0.451`
- `ball_out_of_bounds=40/50`

### 3.2 strong_ramp

- `mean_useful_bounces=0.42`
- `max_useful_bounces=1`
- `two_or_more_useful_bounce_rate=0.00`
- `useful_contact_next_intercept_reachable_rate=0.429`
- `useful_contact_mean_easy_next_ball_score=-0.342`
- `ball_out_of_bounds=37/50`

### 3.3 fixed_inward

- `mean_useful_bounces=0.42`
- `max_useful_bounces=1`
- `two_or_more_useful_bounce_rate=0.00`
- `useful_contact_next_intercept_reachable_rate=1.000`
- `useful_contact_mean_easy_next_ball_score=+0.278`
- `ball_out_of_bounds=41/50`

## 4. 50-episode 결과 해석

세 variant 모두 useful bounce의 개수는 거의 같았다.

하지만 next-ball quality는 차이가 컸다.

- `fixed_inward`는 useful contact 이후 다음 공 quality가 가장 좋았다.
- useful-contact 기준으로는 이미 strike zone으로 돌아올 공을 만들고 있었다.
- 그런데도 `2+ useful bounce`가 없었다.

이 말은 다음과 같다.

> 지금 병목은 “다음 공을 칠 수 있는 곳으로 보내는 것”보다, 그 다음 공을 다시 useful strike로 처리하는 contact/face/strike contract 쪽에 더 가깝다.

즉 `easy_next_ball_score`와 `next_intercept_reachable`는 좋아졌는데, 실제 second useful strike로 연결되지 않았다.

## 5. 100-episode 재확인

best variant인 `fixed_inward`만 100 episodes로 다시 확인했다.

결과:

- `mean_useful_bounces=0.44`
- `max_useful_bounces=1`
- `one_or_more_useful_bounce_rate=0.44`
- `two_or_more_useful_bounce_rate=0.00`
- `useful_contact_next_intercept_reachable_rate=1.000`
- `useful_contact_mean_easy_next_ball_score=+0.309`
- `ball_out_of_bounds=77/100`
- `ball_speed_limit=16/100`

핵심 판정:

- 100 episodes에서도 `2+ useful bounce`는 한 번도 없었다.
- 따라서 heuristic gate는 아직 통과하지 못했다.

## 6. PPO 50k~100k 진행 여부

이번 단계에서는 **진행하지 않았다**.

이유:

1. user가 요청한 gate가 `2+ useful bounce 발생 시`였음
2. best heuristic도 100 episodes에서 `max_useful_bounces=1`
3. 이 상태에서 PPO를 더 돌리면, env가 아직 반복 keep-up contract를 안정적으로 제공하지 않는 문제를 RL로 덮을 가능성이 큼

즉 현재는 PPO보다 먼저 아래를 수정해야 한다.

- second strike contact target
- racket face control during follow-up strike
- useful contact 이후 recovery에서 다시 upward-centered strike로 이어지는 contract

## 7. checkpoint 분석 판정

새로운 50k~100k PPO run은 gate 미통과로 생략했기 때문에, 본 단계에서 새 checkpoint 분석은 수행하지 않았다.

다만 기존 smoke run `phase_contract_smoke_v1`에서 checkpoint plumbing은 정상 동작했다.

- best checkpoint: `512` timesteps
- final model보다 best checkpoint가 따로 저장됨

하지만 smoke 자체가 useful bounce `0`이어서, 이 결과를 성능 판단 근거로 쓰면 안 된다.

정리:

- checkpoint infrastructure: 정상
- 성능 checkpoint 분석: gate 미통과로 보류

## 8. bootstrap 필요 여부

이번 단계에서는 **bootstrap도 진행하지 않았다**.

이유:

1. codebase에는 현재 behavior cloning / imitation / expert rollout warm-start 지원이 없다.
2. 더 중요한 점은, best heuristic조차 `2+ useful bounce`를 못 만든다.
3. 이런 rollout을 bootstrap source로 쓰면 “한 번 useful hit 후 실패하는 정책”을 강화할 위험이 크다.

즉 bootstrap은 아래 조건을 만족할 때만 다시 검토하는 것이 맞다.

- heuristic 또는 scripted controller가 최소한 가끔 `2+ useful bounce`를 만든다.
- 그 rollout이 `useful_contact_next_intercept_reachable_rate`뿐 아니라 실제 second useful strike로 이어진다.

## 9. 현재 가장 그럴듯한 병목

이번 결과를 종합하면 가장 그럴듯한 병목은 아래다.

1. first useful contact 이후 next-ball location은 이미 꽤 좋아질 수 있다.
2. 그런데 follow-up strike에서 racket face / timing / upward motion contract가 다시 무너진다.
3. 그래서 많은 contact가 생겨도 second useful strike가 안 나온다.

특히 `fixed_inward`는 50 episodes에서 `293` contacts, 100 episodes에서 `626` contacts를 만들었는데 useful bounce는 각각 `21`, `44`개였다.

즉 문제는 “공을 전혀 못 따라감”이 아니라, “다시 맞히긴 하는데 useful strike로 못 만든다”에 더 가깝다.

## 10. 다음 실제 구현 우선순위

이제 다음 우선순위는 더 명확하다.

### 10.1 strike contact target 재설계

- 목표는 공을 단순 anchor 위로 보내는 것이 아니라
- 다음 strike plane에서 다시 centered upward strike가 가능한 contact geometry를 만들게 하는 것

### 10.2 follow-up strike face control 강화

- fixed inward bias가 timed ramp보다 낫다는 건 “nonzero inward base”가 중요하다는 뜻이다.
- 다음 단계는 neutral->ramp보다, follow-up strike 동안 유지되는 inward face contract를 다시 설계하는 쪽이 더 유망하다.

### 10.3 그 다음에만 PPO 50k~100k

- heuristic에서 `2+ useful bounce`가 한 번이라도 관찰된 뒤
- 그때 `phase_contract_candidate` PPO 50k~100k와 rebound analysis, checkpoint selection으로 넘어간다.

## 11. 지금 단계의 최종 판정

이번 단계의 gate 결과는 아래 한 줄로 정리할 수 있다.

> `easy-next-ball`은 좋아졌지만, second useful strike contract가 아직 없다. 따라서 PPO를 더 돌리기 전에 strike contact target과 face control을 먼저 손봐야 한다.