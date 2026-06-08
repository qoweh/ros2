# `ppo_keepup_v7` 확인 및 `position_tilt` 제어 보고서

## 1. 목적

이 문서는 다음 다섯 가지를 정리한다.

1. 사용자가 실행한 `ppo_keepup_v7`가 실제로 어떤 run이었는지 확인
2. 현재 policy/control/reward 구조를 keep-up 관점에서 다시 정리
3. `limited pitch/roll residual action`을 추가한 이유와 구현 방향 설명
4. 새 `position_tilt` 40k run 결과를 기존 run과 비교
5. curriculum이 무슨 역할을 하는지, 그리고 지금 정말 필요한지 판단

## 2. `ppo_keepup_v7` 확인

사용자가 실행한 아래 명령 결과는 정상적인 fresh 40k run으로 저장되었다.

- `python pingpong_rl/scripts/run_ppo_baseline.py --run-name ppo_keepup_v7 --total-timesteps 40000`

`ppo_keepup_v7_training_summary.json` 기준 확인 사항:

- `training_mode = new`
- `starting_checkpoint = null`
- `action_mode = position`

즉 `v7`은 여전히 기존 `xyz delta` position-only 정책이다.

## 3. 현재 policy/control 구조 정리

현재 코드 기준 학습 stack은 아래처럼 정리할 수 있다.

### 3.1 PPO policy

- Stable-Baselines3 `PPO`
- policy는 `MlpPolicy`
- curriculum callback이 training progress에 따라 env 파라미터를 stage별로 바꾼다.

### 3.2 action 모드

현재는 두 가지 action 모드가 있다.

`position`

- action = `[dx, dy, dz]`
- 기존 baseline과 동일
- 라켓 위치만 직접 제어

`position_tilt`

- action = `[dx, dy, dz, pitch_residual, roll_residual]`
- 새로 추가된 모드
- 위치에 더해 제한된 pitch/roll residual을 직접 제어

중요한 점:

- `position`에서는 rebound direction을 reward로만 간접 유도한다.
- `position_tilt`에서는 policy가 lateral drift correction과 face angle 조절을 직접 배울 수 있다.

### 3.3 controller

현재 controller는 site Jacobian 기반 IK를 사용한다.

- position 오차는 기존처럼 task-space 위치 오차로 joint target을 계산
- `position_tilt` 모드에서는 rotation Jacobian도 같이 사용
- target face normal은 flat racket 기준 small pitch/roll residual로 만든다.

즉 현재 controller는 `position-only`가 아니라 `position + limited tilt`까지 표현 가능해졌다.

### 3.4 heuristic assist

기존 `compute_keepup_target(...)` 기반 tracking assist는 그대로 살아 있다.

- descending ball에 대해 predicted intercept 쪽으로 target position을 당김
- 다만 curriculum에서 초반에는 assist를 더 크게, 후반에는 줄이는 annealing을 적용한다.

## 4. reward 구조 정리

현재 reward는 아래 묶음으로 보는 것이 가장 명확하다.

### 4.1 contact / bounce 형성

- `contact_bonus`
- `stale_contact_penalty`
- `success_bonus`
- `bounce_progress_bonus`

역할:

- contact 자체와 useful bounce 형성을 초기에 배우게 함

### 4.2 높이 / upward strike

- `height_term`
- `lift_term`
- `active_hit_term`
- `passive_contact_penalty`

역할:

- 공을 위로 올리되 너무 약하거나 너무 과하게 치지 않도록 유도
- 단순 contact보다 `위로 올리는 contact`를 더 선호하게 만듦

### 4.3 위치 정렬 / 중앙 적중

- 기본 `xy_alignment_weight`
- `tracking_alignment_reward_weight`
- `contact_centering_reward_weight`

역할:

- 현재 XY 오차를 줄임
- predicted intercept 기준으로 공 아래에 먼저 들어가도록 유도
- contact 순간 공이 라켓 정중앙에 가까울수록 보상

이 항목이 중요한 이유:

- `v6` 분석에서 descending non-contact step의 약 `81.4%`에서 XY 오차가 커졌다.
- 즉 이전 reward는 `미리 공 아래로 들어가기`를 충분히 밀지 못했다.

### 4.4 rebound 안정화

- `lateral_contact_velocity_penalty_weight`
- `rebound_direction_reward_weight`는 현재 기본값 `0.0`

역할:

- contact 이후 옆으로 너무 많이 흐르는 rebound를 억제

판단:

- reward-only rebound shaping은 `v4`, `v5`에서 실패했다.
- 따라서 현재 우선순위는 reward를 더 복잡하게 만드는 것이 아니라 control/action 확장이다.

### 4.5 안정성 / regularization

- `racket_tilt_penalty_weight`
- `joint_velocity_penalty_weight`
- `action_smoothness_penalty_weight`
- `action_filter_alpha`

역할:

- 공을 치는 기본 동작이 생긴 뒤, 자세와 모션 품질을 정리하는 late-stage regularizer

## 5. `position_tilt`를 왜 추가했는가

핵심 이유는 단순하다.

- keep-up은 단순히 공 아래 XY로 들어가는 것만으로 끝나지 않는다.
- 실제로는 `정렬 -> 필요하면 살짝 받쳐 내려가며 타이밍 맞춤 -> 위로 올려침`이 필요하다.
- 그런데 기존 action은 `[dx, dy, dz]`뿐이라 rebound direction을 직접 만들 수 없다.

즉 기존 구조에서는 아래가 모두 간접적이었다.

- face angle correction
- glancing contact 감소
- lateral drift 보정
- 위로 세우는 rebound shaping

그래서 `pitch/roll residual` 2축을 작게 추가했다.

이건 full orientation control보다 탐색 부담이 낮고, keep-up에 필요한 표현력만 늘리는 타협안이다.

## 6. 실험 비교

### 6.1 `v6` vs `v7` vs `v8`

`ppo_keepup_v6_trackassist40k`

- action mode: `position`
- `robot_body_contact`: `0.2986`
- `ball_out_of_bounds`: `0.5683`
- `bounce>=1`: `0.8525`
- `bounce>=3`: `0.1007`
- `bounce>=5`: `0.0204`
- 평균 bounce 수: `1.3058`
- 평균 episode 길이: `48.1247`

`ppo_keepup_v7`

- action mode: `position`
- `robot_body_contact`: `0.3885`
- `ball_out_of_bounds`: `0.4872`
- `bounce>=1`: `0.8946`
- `bounce>=3`: `0.0655`
- `bounce>=5`: `0.0100`
- 평균 bounce 수: `1.2242`
- 평균 episode 길이: `44.6049`

`ppo_keepup_v8_tilt40k`

- action mode: `position_tilt`
- `robot_body_contact`: `0.3773`
- `ball_out_of_bounds`: `0.4931`
- `bounce>=1`: `0.7878`
- `bounce>=3`: `0.0791`
- `bounce>=5`: `0.0046`
- 평균 bounce 수: `1.1399`
- 평균 episode 길이: `46.0837`

### 6.2 해석

`v7`:

- single bounce 쪽은 좋아졌지만
- multi-bounce와 body collision은 악화됐다.
- 특히 후반으로 갈수록 `robot_body_contact`가 크게 증가했다.

`v8 position_tilt`:

- `v7`보다 `robot_body_contact`는 조금 낮아졌지만
- 여전히 `v6`보다 나쁘다.
- `bounce>=3`도 `v6` 수준까지 회복되지 못했다.
- 즉 `pitch/roll residual`을 여는 방향 자체는 구조적으로 맞지만, `40k` budget에서는 아직 충분히 학습되지 않았다.

결론:

- `position_tilt`는 잘못된 방향은 아니다.
- 다만 action 차원이 늘었기 때문에 초기 탐색 부담이 커졌고,
- 현재 `40k`에서는 그 추가 자유도를 아직 제대로 활용하지 못했다.

## 7. descending-ball 정렬 관찰

descending non-contact 구간에서 XY 오차가 다음 step에 증가한 비율:

- `v6`: `0.8141`
- `v7`: `0.7615`
- `v8`: `0.7680`

이 수치만 보면 `v7`, `v8`이 `v6`보다 약간 낫다.

하지만 중요한 것은 최종 성능이다.

- `v7`, `v8` 모두 `robot_body_contact`와 multi-bounce 안정성에서 `v6`를 넘지 못했다.

즉 단순 정렬 개선만으로는 부족하고,

- contact timing
- strike quality
- rebound control

까지 같이 안정되어야 한다.

## 8. curriculum의 역할

현재 curriculum은 단순한 학습 편의 옵션이 아니라, 사실상 exploration shaping 역할을 한다.

stage별 역할은 아래다.

`bootstrap`

- reset randomization 거의 제거
- success threshold 완화
- assist 강하게
- regularization off

목적:

- 공을 최소한 위로 치는 기본 패턴 만들기

`stabilize`

- randomization 일부 복귀
- assist 기본 수준으로 감소
- tilt/joint/smoothness penalty 약하게 켬

목적:

- 만들어진 strike를 여러 번 반복 가능한 쪽으로 옮기기

`refine`

- randomization 강화
- assist 더 줄임
- centering weight와 regularization 강화

목적:

- heuristic 의존을 줄이고, 자세와 모션 품질을 더 실제적인 쪽으로 정리

## 9. curriculum이 필요한가

현재 과제에서는 `필요하다`는 쪽이 맞다.

이유:

1. keep-up은 contact 형성, upward strike, rebound direction, 자세 유지까지 동시에 요구한다.
2. `position_tilt`처럼 action 차원이 늘어나면 exploration 난도가 더 커진다.
3. `v7`, `v8` 모두 후반부에서 `robot_body_contact`가 커지는 패턴이 보인다.

즉 curriculum이 없으면 정책은 초기에 배워야 할 것과 나중에 배워야 할 것을 구분하기 더 어려워진다.

다만 보완은 필요하다.

- 현재 stage 전환은 timestep 비율 기반의 단순 스케줄이다.
- 향후에는 성능 조건 기반 전환이나, `bounce>=3`/body-contact 추세를 기준으로 한 adaptive stage가 더 낫다.

## 10. 현재 판단

현재까지의 판단은 아래처럼 정리된다.

1. reward-only rebound shaping은 우선순위가 아니다.
2. control/action 확장은 맞는 방향이다.
3. 하지만 `position_tilt`를 넣었다고 40k 안에 바로 좋아지지는 않았다.
4. 따라서 다음은 `더 긴 budget` 또는 `더 강한 bootstrap`이 필요하다.

현 시점에서 가장 현실적인 다음 실험은 아래 둘 중 하나다.

### 10.1 `position_tilt` 장기 학습

- `ppo_keepup_v8_tilt40k`를 40k에서 멈추지 말고 더 길게 학습
- 최소 `150k~200k`는 봐야 한다.

### 10.2 `position_tilt` bootstrap 강화

- bootstrap에서 assist를 더 오래 유지
- refine 전환을 늦추거나 condition-based로 변경
- 초기 tilt residual scale을 더 작게 시작하고 점진적으로 늘리는 방식 검토

요약하면,

- `v7`은 reward만으로 정렬을 밀다가 반복 안정성을 잃은 케이스에 가깝고,
- `v8`은 control 표현력은 늘었지만 아직 충분히 학습되지 않은 케이스다.

지금 병목은 여전히 `reward 디자인 미세조정`보다는 `control 표현력 + bootstrap strategy` 쪽이다.