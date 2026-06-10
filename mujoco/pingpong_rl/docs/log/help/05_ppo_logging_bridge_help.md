# PPO Logging Bridge 도움말

이 문서는 이번 단계에서 추가한 PPO-side logging 연결을 설명한다.

핵심 목표는 하나였다.

- rollout analysis에서 보던 필드 이름과 의미를 PPO 학습 로그에서도 그대로 유지하기

즉 이제는 아래 두 경로가 같은 schema를 쓴다.

- `run_ee_rollout_analysis.py`
- `run_ppo_baseline.py`

## 1. 왜 Gymnasium wrapper를 따로 뒀는가

현재 `PingPongEEDeltaEnv`는 이미 RL 직전 contract를 가진 얇은 env다.

하지만 SB3 PPO는 Gymnasium `Env`와 `action_space`, `observation_space`를 기대한다.

그래서 기존 env를 바꾸지 않고, 그 위에 얇은 wrapper 하나만 추가했다.

파일:
- `pingpong_rl/src/pingpong_rl/envs/ee_delta_gym_env.py`

이 wrapper는 아래만 담당한다.

- `observation_space`: `(26,)` float32 Box
- `action_space`: `(3,)` float32 Box
- `reset/step` 반환을 Gymnasium 형식으로 전달

중요:
- underlying env observation 순서/shape는 그대로다.
- `terminated/truncated` 의미도 그대로다.

## 2. PPO logging은 어떤 파일을 내보내는가

실행 스크립트:

```bash
conda activate mujoco_env
cd mujoco
python pingpong_rl/scripts/run_ppo_baseline.py --total-timesteps 2048 --run-name ppo_baseline
```

기본 출력 경로:

- `pingpong_rl/docs/etc/ppo_runs/<run-name>`

현재 생성되는 산출물:

- `*_episodes.csv`
- `*_steps.csv`
- `*_contacts.csv`
- `*_training_summary.json`
- `tensorboard/`
- `*_model.zip`

즉 rollout analysis와 거의 같은 CSV/JSON 구조를 PPO training에도 그대로 붙였다.

## 3. PPO 쪽에서도 유지되는 핵심 logging 필드

reward 계열:
- `reward_total`
- `reward_height`
- `reward_distance`
- `reward_contact`
- `reward_success`
- `reward_failure`

episode 경계:
- `terminated`
- `truncated`
- `success_reason`
- `failure_reason`
- `time_limit_reached`
- `episode_steps`

contact/velocity:
- `contact_observed_during_step`
- `contact_substep`
- `ball_velocity_x`
- `ball_velocity_y`
- `ball_velocity_z`
- `ball_speed_norm`
- `contact_ball_velocity_x`
- `contact_ball_velocity_y`
- `contact_ball_velocity_z`
- `contact_ball_speed_norm`

즉 rollout analysis에서 보던 필드를 PPO 학습에서도 같은 이름으로 읽을 수 있다.

## 4. TensorBoard에는 무엇이 기록되는가

현재는 episode 단위 scalar를 기록한다.

예:
- `reward_total_sum`
- `reward_height_sum`
- `reward_distance_sum`
- `reward_contact_sum`
- `reward_success_sum`
- `reward_failure_sum`
- `episode_steps`
- `contact_count`
- `terminated`
- `truncated`
- `time_limit_reached`

reason은 categorical이라 다음처럼 별도 scalar로 찍는다.

- `success_reason/<name>`
- `failure_reason/<name>`

## 5. smoke run에서 확인된 것

검증 명령:

```bash
conda activate mujoco_env
cd mujoco
python pingpong_rl/scripts/run_ppo_baseline.py \
  --total-timesteps 256 \
  --n-steps 64 \
  --batch-size 32 \
  --run-name ppo_smoke
```

확인된 점:
- model zip 저장됨
- TensorBoard 로그 생성됨
- episode/step/contact CSV 생성됨
- training summary JSON 생성됨

즉 PPO-side logging bridge는 실제로 동작한다.

## 6. 지금 단계에서 아직 하지 않은 것

이번 단계에서는 아래를 일부러 하지 않았다.

- reward 식 변경
- success threshold 수정
- physics parameter 조정
- 성능 좋은 PPO hyperparameter 탐색

이유:
- 현재 목표는 성능 향상이 아니라, PPO가 어떤 신호를 보고 학습하게 될지 먼저 드러내는 것이다.

## 7. 주의할 점

이번 smoke run은 SB3 `device=auto` 기준으로 CPU에서 돌았다.

즉 이후 장기 학습에서 MPS를 쓰려면:

```bash
python pingpong_rl/scripts/run_ppo_baseline.py --device mps
```

처럼 명시하는 편이 낫다.

다만 이번 단계의 목적은 logging 검증이므로, device 최적화는 우선순위를 뒤로 미뤘다.