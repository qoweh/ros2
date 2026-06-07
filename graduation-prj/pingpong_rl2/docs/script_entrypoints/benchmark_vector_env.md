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

## 발표 때 설명 포인트

- 학습 속도 최적화나 `n_envs` 선택 근거를 설명할 때 사용할 수 있다.
- 최종 성능 자체가 아니라 학습 인프라 점검용이다.
