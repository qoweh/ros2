# `ppo_keepup_v3` 후속 실험 및 control assist 보고서

## 1. 목적

이 문서는 사용자가 다시 실행한 아래 학습 결과를 기준으로,

- `python pingpong_rl/scripts/run_ppo_baseline.py --run-name ppo_keepup_v3 --total-timesteps 40000 --reset-model`

현재 keep-up 학습이 실제로 나아지는지 확인하고,

- reward-only rebound shaping
- control-side tracking assist

중 어느 방향이 더 유효한지 비교한 기록이다.

## 2. `ppo_keepup_v3` 실행 확인

`ppo_keepup_v3_training_summary.json` 기준으로 이번 run은 다음 조건을 만족했다.

- `total_timesteps = 40000`
- `training_mode = new`
- `starting_checkpoint = null`
- `target_ball_height_reference = target_height_above_racket`

즉, 예전 checkpoint를 이어받은 run이 아니라 `--reset-model`로 새로 시작된 40k run으로 보는 것이 맞다.

## 3. `ppo_keepup_v3` 내부 progression

`ppo_keepup_v3`의 early/late 구간을 나누어 보면, 학습은 일부 진행됐지만 안정 keep-up으로 수렴했다고 보기는 어렵다.

### 3.1 전체 수치

- episode 수: `903`
- `robot_body_contact` 비율: `0.3145`
- `ball_out_of_bounds` 비율: `0.5482`
- `bounce>=1` 비율: `0.7719`
- `bounce>=3` 비율: `0.1008`
- 평균 bounce 수: `1.2038`
- 평균 episode 길이: `44.4862`

### 3.2 first half vs last half

first half:

- `robot_body_contact` 비율: `0.3038`
- `bounce>=3` 비율: `0.0843`
- 평균 bounce 수: `1.0798`
- 평균 episode 길이: `42.7339`

last half:

- `robot_body_contact` 비율: `0.3252`
- `bounce>=3` 비율: `0.1173`
- 평균 bounce 수: `1.3274`
- 평균 episode 길이: `46.2345`

해석:

- `bounce>=3`와 평균 bounce 수는 올라갔다.
- 하지만 `robot_body_contact`는 줄지 않고 오히려 소폭 증가했다.
- 따라서 현재 `ppo_keepup_v3`는 `더 자주 맞추는 방향`으로는 움직이지만, `body collision 없이 안정적으로 반복`하는 방향으로는 아직 정리되지 않았다.

## 4. Reward-only rebound shaping 실험

orientation/rebound direction 병목을 reward만으로 해결할 수 있는지 보기 위해 두 가지 variant를 시험했다.

### 4.1 `ppo_keepup_v4_rebound40k`

변경:

- contact 시 vertical rebound ratio를 직접 reward/penalty로 반영하는 symmetric shaping 추가

결과:

- `robot_body_contact` 비율: `0.3284`
- `bounce>=3` 비율: `0.0129`
- 평균 bounce 수: `0.8031`
- 평균 episode 길이: `36.9540`
- contact lateral speed p50: `0.4979`

해석:

- vertical ratio 자체는 높게 찍혔지만,
- 실제 repeated keep-up 성능은 크게 악화됐다.
- one-shot vertical spike 쪽으로 reward hacking이 생긴 것으로 보는 편이 맞다.

### 4.2 `ppo_keepup_v5_reboundpen40k`

변경:

- symmetric reward를 버리고 sideways rebound에 대한 penalty-only shaping으로 수정

결과:

- `robot_body_contact` 비율: `0.3732`
- `bounce>=3` 비율: `0.0304`
- 평균 bounce 수: `0.9148`
- 평균 episode 길이: `39.3565`
- contact lateral speed p50: `0.3963`

해석:

- symmetric version보다는 덜 나빴지만,
- 여전히 `ppo_keepup_v3`보다 분명히 못하다.
- 결론적으로 `reward-only rebound shaping`은 현재 제어 구조에서 우선순위가 아니다.

## 5. Control-side tracking assist 실험

reward-only 방식이 실패했기 때문에, 다음 hop은 reward가 아니라 control path로 옮겼다.

이번에 추가한 것은 strike zone 안에서 descending ball에 대해 `compute_keepup_target(...)`를 약하게 blend하는 `tracking_assist_weight=0.2`다.

핵심 의도:

- policy action은 그대로 `EE delta xyz`
- orientation 제어는 아직 없음
- 대신 contact 직전 glancing hit를 줄이기 위해 target을 predicted intercept 쪽으로 약하게 recenter

### 5.1 `ppo_keepup_v6_trackassist40k`

결과:

- episode 수: `834`
- `robot_body_contact` 비율: `0.2986`
- `ball_out_of_bounds` 비율: `0.5683`
- `bounce>=1` 비율: `0.8525`
- `bounce>=3` 비율: `0.1007`
- `bounce>=5` 비율: `0.0204`
- 평균 bounce 수: `1.3058`
- 평균 episode 길이: `48.1247`
- total contacts: `2050`
- contact lateral speed p50: `0.3078`

### 5.2 `ppo_keepup_v3` 대비 비교

`ppo_keepup_v3`:

- `robot_body_contact` 비율: `0.3145`
- `bounce>=1` 비율: `0.7719`
- `bounce>=3` 비율: `0.1008`
- 평균 bounce 수: `1.2038`
- 평균 episode 길이: `44.4862`
- total contacts: `1983`
- contact lateral speed p50: `0.3285`

`ppo_keepup_v6_trackassist40k`:

- `robot_body_contact` 비율: `0.2986`
- `bounce>=1` 비율: `0.8525`
- `bounce>=3` 비율: `0.1007`
- 평균 bounce 수: `1.3058`
- 평균 episode 길이: `48.1247`
- total contacts: `2050`
- contact lateral speed p50: `0.3078`

해석:

- `robot_body_contact`는 줄었다.
- `bounce>=1`는 확실히 늘었다.
- 평균 bounce 수와 평균 episode 길이도 올랐다.
- lateral speed p50도 소폭 줄었다.
- 다만 `bounce>=3`는 사실상 비슷한 수준이다.

즉, control assist는 `초기 repeated contact quality`를 개선하는 데는 도움이 되었지만, 아직 `장기 안정 keep-up`으로 확실히 넘어간 것은 아니다.

### 5.3 `v6` 내부 progression

first half:

- `robot_body_contact` 비율: `0.3237`
- `bounce>=3` 비율: `0.1127`
- 평균 bounce 수: `1.2686`
- 평균 episode 길이: `46.2398`

last half:

- `robot_body_contact` 비율: `0.2734`
- `bounce>=3` 비율: `0.0887`
- 평균 bounce 수: `1.3429`
- 평균 episode 길이: `50.0096`

해석:

- `robot_body_contact`는 학습 후반에 줄어드는 방향이 보였다.
- 평균 bounce 수와 episode 길이도 후반이 더 좋다.
- 하지만 `bounce>=3`는 아직 noisy하며 안정적으로 올라간다고 말하기 어렵다.

## 6. 현재 코드 반영 사항

이번 작업에서 코드 기준으로 반영된 내용은 아래와 같다.

1. keep-up workspace 기본 범위를 더 좁게 유지한다.
2. reward-only rebound shaping은 실험은 남겨 두되, 기본값은 `0.0`으로 꺼 두었다.
3. 기본 training 경로에는 `tracking_assist_weight=0.2` control assist를 넣었다.
4. PPO script에서는 아래 옵션으로 관련 실험을 켜고 끌 수 있게 했다.
   - `--target-rebound-vertical-ratio`
   - `--rebound-direction-reward-weight`
   - `--tracking-assist-weight`
   - `--tracking-assist-preview-time`

## 7. 다음으로 손볼 후보

현재까지의 실험 결과를 기준으로, 다음 우선순위는 아래 순서가 맞다.

### 7.1 제한된 orientation action 추가

가장 유력한 다음 후보다.

- 현재 action은 `xyz delta`뿐이다.
- orientation은 reward로만 암시할 수 있고 직접 제어되지 않는다.
- `pitch/roll residual` 1~2축만 열어도 lateral drift correction이 가능해질 수 있다.

추천 형태:

- 기존 `xyz` 유지
- 추가로 작은 `pitch/roll` residual 1~2축만 허용
- observation에도 racket face normal 또는 orientation residual 포함

### 7.2 impact velocity primitive 추가

현재는 위치 목표만 바꾸므로 impact velocity가 controller response에 간접적으로만 정해진다.

가능한 후보:

- strike amplitude residual
- desired upward racket velocity residual
- contact 직전 몇 step 동안만 쓰는 short strike primitive

이건 stable keep-up에 필요한 `치는 느낌`을 reward보다 직접적으로 줄 수 있다.

### 7.3 tracking assist curriculum/annealing

현재 `tracking_assist_weight=0.2`는 고정값이다.

다음 실험 후보:

- bootstrap 구간에서는 assist를 더 크게
- 후반으로 갈수록 assist를 줄여 policy 자율성 회수

이 방식은 current action space를 유지한 채 bootstrap quality를 더 올릴 수 있다.

### 7.4 robot body contact 예방을 더 직접적으로 넣기

현재 body collision은 still dominant failure 중 하나다.

후보:

- target workspace를 joint/body safe zone 기준으로 더 타이트하게 제한
- 특정 link 근처 접근 시 추가 soft penalty
- `robot_body_contact_body` 분포 기준으로 link6/link5 쪽을 먼저 차단

### 7.5 observation 확장

orientation action을 열기 전후 모두 고려할 수 있다.

후보:

- racket face normal
- predicted intercept XY error
- contact 이후 rebound cadence proxy

단, observation만 늘리고 control이 그대로면 효과는 제한적일 가능성이 높다.

## 8. 결론

이번 후속 실험에서 확인된 것은 명확하다.

- `ppo_keepup_v3`는 내부적으로 `bounce>=3`가 조금 올라가지만 `robot_body_contact`는 줄지 않는다.
- reward-only rebound shaping 둘은 둘 다 실패했다.
- control-side tracking assist는 현재 예산(`40k`)에서 가장 괜찮은 방향이었다.

따라서 다음 우선순위는 reward를 더 복잡하게 만드는 것이 아니라,

- `orientation`
- `impact velocity shaping`
- `tracking assist curriculum`

같이 `rebound direction`을 더 직접적으로 만들 수 있는 control/action 확장 쪽이다.