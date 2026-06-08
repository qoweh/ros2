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

## 발표 때 설명 포인트

- action space가 커졌을 때 처음부터 학습하지 않고, 기존 policy의 앞 action 축을 보존하는 transfer 방식이다.
- 새 action 축은 처음에는 거의 중립이므로 PPO가 이후 학습에서 필요한 만큼 사용하게 된다.
