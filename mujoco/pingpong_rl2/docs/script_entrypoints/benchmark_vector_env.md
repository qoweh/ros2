# benchmark_vector_env.py

## 한 줄 역할

`scripts/benchmark_vector_env.py`는 Gym vector env의 step 처리량을 측정하는 성능 점검 entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/benchmark_vector_env.py \
  --n-envs 4 \
  --steps 256 \
  --vector-mode async
```

## 코드 흐름

1. `make_gym_vector_env(num_envs=args.n_envs, vector_mode=args.vector_mode)`로 vector env를 만든다.
2. reset 후 `steps`만큼 random action을 샘플링해 env를 step한다.
3. done mask가 있으면 해당 env만 reset한다.
4. elapsed time과 `env_steps_per_second`를 출력한다.

## 주요 호출 관계

```text
benchmark_vector_env.py
  -> training/vector_env.py
```

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 모델을 만들지도 않고 PPO를 학습하지도 않는다. 목적은 `make_gym_vector_env()`가 만든 Gymnasium vector env가 얼마나 빠르게 reset/step 되는지 보는 것이다.

```text
benchmark_vector_env.py
  -> parse_args()
  -> make_gym_vector_env(num_envs, vector_mode)
  -> vector_env.reset(seed)
  -> for steps
       random action을 n_envs개 샘플링
       vector_env.step(actions)
       terminated/truncated env 확인
       끝난 env만 reset_mask로 reset
  -> elapsed time, env_steps_per_second 출력
```

여기서 action은 PPO policy가 낸 action이 아니라 `single_action_space.sample()`로 뽑은 random action이다. 따라서 성능 숫자는 "학습된 정책이 잘한다"는 의미가 전혀 아니다. 순수하게 환경 여러 개를 병렬로 돌릴 때 처리량이 어느 정도인지 보는 값이다.

`vector_mode=async`는 여러 환경을 비동기로 돌려 throughput을 높이려는 모드고, `vector_mode=sync`는 같은 프로세스 흐름에서 동기적으로 돌리는 모드다. 실제 `run_ppo_learning.py`는 SB3용 adapter를 거치지만, 이 벤치마크는 그보다 아래의 Gym vector env 처리량을 먼저 확인한다.

## 결과 해석

`env_steps_per_second`는 `steps * n_envs / elapsed_seconds`다. PPO 학습에서는 rollout 수집 속도가 병목이 될 수 있으므로 `n_envs`를 늘렸을 때 이 값이 얼마나 좋아지는지 본다.

`completed_episodes`는 random action으로 진행하는 동안 episode가 몇 번 끝났는지를 센 값이다. 이 값은 정책 성능 지표가 아니라 reset 처리까지 benchmark에 포함되었는지 확인하는 보조 값이다.

## 발표 때 설명 포인트

- 학습 속도 최적화나 `n_envs` 선택 근거를 설명할 때 사용할 수 있다.
- 최종 성능 자체가 아니라 학습 인프라 점검용이다.
