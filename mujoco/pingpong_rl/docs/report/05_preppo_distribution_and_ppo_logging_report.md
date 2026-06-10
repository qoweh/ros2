# Pre-PPO 분포 분석과 PPO Logging 연결 5차 작업 보고

## 1. 작업 목표

이번 단계의 목표는 세 가지였다.

- multi-episode rollout으로 env signal 분포를 실제로 수집하기
- reward dominance와 종료 패턴을 정리하기
- 같은 logging schema를 PPO 학습 경로까지 연결하기

중요:
- threshold는 자동 수정하지 않았다.
- reward semantics는 바꾸지 않았다.
- physics parameter도 바꾸지 않았다.

## 2. multi-episode rollout 분석 결과

실행 명령:

```bash
conda activate mujoco_env
cd mujoco
python pingpong_rl/scripts/run_ee_rollout_analysis.py \
  --episodes 200 \
  --output-dir docs/etc/rollout_analysis/preppo_200 \
  --output-prefix ee_preppo_200
```

산출물:
- `pingpong_rl/docs/etc/rollout_analysis/preppo_200/ee_preppo_200_episodes.csv`
- `pingpong_rl/docs/etc/rollout_analysis/preppo_200/ee_preppo_200_steps.csv`
- `pingpong_rl/docs/etc/rollout_analysis/preppo_200/ee_preppo_200_contacts.csv`
- `pingpong_rl/docs/etc/rollout_analysis/preppo_200/ee_preppo_200_summary.json`

### 2.1 종료 패턴
- episode 수: `200`
- `terminated = 200`
- `truncated = 0`
- success: `0`
- failure: `200`
- failure reason: `ball_out_of_bounds = 200`

즉 현재 기본 zero-action rollout은 time limit까지 버티는 구조가 아니라, 전부 out-of-bounds로 먼저 끝난다.

### 2.2 contact velocity 분포
summary JSON 기준:

- `ball_velocity_z`
  - `p50 = 0.4687`
  - `p75 = 0.6445`
  - `p90 = 1.7415`
  - `max = 1.7415`

즉 현재 success threshold `0.5`는:
- median보다 약간 높고
- p75보다 낮다.

threshold 후보를 굳이 제안만 하면:
- permissive 후보: 약 `0.45`
- 현 상태 유지 후보: `0.5`
- conservative 후보: 약 `0.64`

하지만 이번 단계에서는 threshold를 바꾸지 않았다.

### 2.3 reward dominance
summary JSON 기준:

- `reward_height_sum`
  - `p50 = 3.3337`
- `reward_contact_sum`
  - `p50 = 0.0`
- `reward_success_sum`
  - `p50 = 0.0`

dominance count:
- `zero_contact_reward_episodes = 200`
- `zero_success_reward_episodes = 200`
- `height_dominant_episodes = 200`
- `survival_without_success_episodes = 200`

즉 현재 zero-action 분포만 보면:
- reward는 거의 전부 height term에서 나온다.
- contact/success 보상은 실질적으로 전혀 누적되지 않는다.
- 현재 신호만 보면 “공을 치는 전략”보다 “height term을 오래 먹는 방향”으로 왜곡될 위험이 크다.

## 3. 중요한 해석

이번 결과에서 더 중요한 점은 success가 0이라는 사실 자체보다, 아래 두 신호가 동시에 보인다는 점이다.

1. contact trace는 많이 잡힌다.
2. success termination은 한 번도 안 난다.

즉 현재는 transient contact는 존재하지만, 현재 success 판정 contract에서는 그 접촉이 success로 이어지지 않는다.

이건 threshold만의 문제라고 단정하면 안 된다.

왜냐하면 현재 success contract는 end-of-step contact 기준이고, rollout analysis는 substep trace 기준도 같이 보고 있기 때문이다.

즉 이번 단계에서 내릴 수 있는 결론은:
- threshold를 바로 바꾸기보다
- current success contract가 실제 transient contact 분포와 얼마나 어긋나는지 먼저 인지해야 한다
이다.

## 4. PPO logging 연결 결과

### 4.1 구현 내용
- Gymnasium wrapper 추가:
  - `pingpong_rl/src/pingpong_rl/envs/ee_delta_gym_env.py`
- PPO logging callback 추가:
  - `pingpong_rl/src/pingpong_rl/training/ppo_logging.py`
- PPO baseline script 추가:
  - `pingpong_rl/scripts/run_ppo_baseline.py`

### 4.2 dependency 추가
- `gymnasium`
- `stable-baselines3`
- `tensorboard`

### 4.3 smoke run 검증
실행 명령:

```bash
conda activate mujoco_env
cd mujoco
python pingpong_rl/scripts/run_ppo_baseline.py \
  --total-timesteps 256 \
  --n-steps 64 \
  --batch-size 32 \
  --run-name ppo_smoke
```

생성 확인:
- model zip
- TensorBoard log directory
- `ppo_smoke_episodes.csv`
- `ppo_smoke_steps.csv`
- `ppo_smoke_contacts.csv`
- `ppo_smoke_training_summary.json`

즉 rollout analysis와 PPO training이 같은 필드 이름으로 logging되는 경로는 연결되었다.

## 5. PPO smoke run에서 드러난 현재 신호

smoke PPO summary 기준:
- episode 수: `5`
- success: `0`
- failure: `5`
- failure reason:
  - `ball_out_of_bounds = 4`
  - `ball_speed_limit = 1`

reward sum stats:
- `reward_contact_sum p50 = 3.0`
- `reward_success_sum p50 = 0.0`
- `reward_height_sum p50 = 0.7782`

해석:
- PPO가 약간 action을 바꾸면 contact reward는 쌓일 수 있다.
- 하지만 success는 여전히 0이다.
- 즉 현재 단계에서 policy가 먼저 배우게 될 가능성이 큰 것은 “성공”이 아니라 “contact를 조금 더 많이 만드는 것”이다.

## 6. 현재 env의 주요 종료 패턴

현재까지 정리하면 주요 종료 패턴은 다음과 같다.

- rollout zero-action 기준:
  - 거의 항상 `ball_out_of_bounds`
- PPO smoke 기준:
  - 주로 `ball_out_of_bounds`
  - 가끔 `ball_speed_limit`
- time limit 종료는 아직 관측되지 않음
- success 종료도 아직 관측되지 않음

즉 현재 pre-PPO 단계의 가장 강한 리스크는:
- success가 너무 드물거나 관측 불가능하고
- height/contact 같은 대체 신호만 먼저 학습되는 구조
라는 점이다.

## 7. 다음 작업 제안

### 7.1 바로 이어서 할 것
- PPO timesteps를 늘리기 전에 같은 logging schema로 1차 baseline 몇 run 더 수집
- `ball_out_of_bounds`가 왜 먼저 뜨는지 rollout CSV와 contact CSV 기준으로 보기
- success threshold는 아직 고치지 말고, 먼저 transient contact와 end-of-step success 판정의 차이를 정리

### 7.2 그 다음
- threshold 후보 재검토
- success contract와 실제 contact trace의 간극을 메울지 판단
- 그 뒤 reward hacking 가능성을 보며 reward 항 재설계 검토