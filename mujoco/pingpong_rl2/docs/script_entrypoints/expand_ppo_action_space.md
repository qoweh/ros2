# expand_ppo_action_space.py

## 한 줄 역할

`scripts/expand_ppo_action_space.py`는 기존 PPO 모델의 action dimension을 더 큰 action space로 확장한 초기 checkpoint를 만드는 transfer utility다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/expand_ppo_action_space.py \
  --source-model artifacts/ppo_runs/old_run/old_run_model.zip \
  --output-model artifacts/ppo_runs/new_run/new_run_model.zip \
  --target-preset contact_frame_self_rally_v32_17d_v30_transfer
```

## 코드 흐름

1. target env kwargs를 만든다.
   - `target_env_kwargs()`가 `run_ppo_learning.parse_args()`와 `apply_env_preset()`을 재사용한다.
   - 그래서 target action/observation space가 이후 학습 entrypoint와 맞는다.

2. source PPO를 로드한다.
   - `PPO.load(source_path, device=args.device)`

3. target PPO 껍데기를 만든다.
   - target env에 대해 새 `PPO("MlpPolicy", env, ...)`를 생성한다.
   - 여기서 학습이 목적이 아니라 target network shape을 얻는 것이 목적이다.

4. policy parameter를 복사한다.
   - `copy_policy_prefix()`가 observation space가 같은지 확인한다.
   - shape이 같은 weight는 그대로 복사한다.
   - action head와 `log_std`는 기존 action dimension 앞부분만 복사하고 새 action dimension은 0 또는 작은 std로 초기화한다.

5. 새 모델과 summary를 저장한다.
   - output model zip
   - `<output_model stem>_training_summary.json`

## 주요 호출 관계

```text
expand_ppo_action_space.py
  -> run_ppo_learning.py       # target preset/config 재사용
  -> envs/gym_env.py           # target action/observation space 생성
  -> stable_baselines3.PPO
```

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 학습을 수행하는 스크립트가 아니라 transfer용 checkpoint를 만드는 스크립트다. 기존 policy가 학습한 observation 처리부와 기존 action 축의 parameter는 최대한 보존하고, 새 action 축만 0 또는 작은 표준편차로 초기화한다.

```text
expand_ppo_action_space.py
  -> source PPO.load()
  -> target_env_kwargs()
       -> run_ppo_learning.parse_args()
       -> apply_env_preset()
       -> env_kwargs_from_args()
  -> PingPongKeepUpGymEnv(**target_env_kwargs)
  -> target PPO("MlpPolicy", target_env)
  -> copy_policy_prefix()
  -> target_model.save()
  -> transfer summary 저장
```

여기서 `target_env_kwargs()`가 `run_ppo_learning.py`를 import해서 쓰는 이유는 중요하다. target action/observation space를 임의로 만들면 이후 실제 학습 entrypoint와 shape이 달라질 수 있다. 그래서 target preset 또는 config를 학습 스크립트와 같은 parser/preset/env config 경로로 통과시킨다.

## copy_policy_prefix()가 하는 일

먼저 source와 target의 observation space shape이 같은지 확인한다. observation dimension이 다르면 같은 feature extractor와 policy body를 그대로 복사하는 것이 안전하지 않기 때문이다. 이 helper는 action space 확장만 다룬다.

그 다음 source action dimension과 target action dimension을 계산한다. source가 target보다 크거나 같으면 "확장"이 아니므로 예외를 낸다.

policy state dict를 순회하면서 shape이 같은 parameter는 그대로 복사한다. 보통 feature extractor, MLP hidden layer, value head처럼 action dimension과 무관한 부분이 여기에 해당한다.

action dimension에 직접 연결된 parameter는 따로 처리한다.

```text
log_std
  -> 기존 action 축은 source 값을 복사
  -> 새 action 축은 target action limit * new_action_std_ratio 기준으로 초기화

action_net.weight
  -> 기존 action 축 row는 source 값을 복사
  -> 새 action 축 row는 0으로 초기화

action_net.bias
  -> 기존 action 축은 source 값을 복사
  -> 새 action 축은 0으로 초기화
```

이렇게 하면 target policy는 처음부터 새 action 축을 강하게 사용하지 않는다. 기존 policy가 알던 행동은 유지하고, PPO 추가 학습 중 새 residual 축을 필요할 때 배우도록 출발점을 만든다.

## 이 파일이 로봇팔 제어와 연결되는 방식

이 파일 자체는 `env.step()`을 반복하지 않으므로 MuJoCo 물리를 진행하지 않는다. target env를 만드는 이유는 action/observation space와 policy network shape을 알기 위해서다.

만들어진 output model을 이후 `run_ppo_learning.py --resume-from <output_model>` 또는 해당 run directory resume으로 학습하면, 그때부터 새 action mode의 residual action이 `PingPongKeepUpEnv.step()`으로 들어가고 Cartesian controller와 MuJoCo step으로 내려간다.

## 발표 때 설명 포인트

- action space가 커졌을 때 처음부터 학습하지 않고, 기존 policy의 앞 action 축을 보존하는 transfer 방식이다.
- 새 action 축은 처음에는 거의 중립이므로 PPO가 이후 학습에서 필요한 만큼 사용하게 된다.
