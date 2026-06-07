# run_ppo_learning.py

## 한 줄 역할

`scripts/run_ppo_learning.py`는 `pingpong_rl2`의 PPO 학습을 시작하는 최상위 entrypoint다. 발표에서 “강화학습이 어떻게 진행됐나?”를 설명할 때 가장 먼저 잡아야 하는 파일이다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_ppo_learning.py \
  --run-name keep1 \
  --run-version v39_17d_mid_curriculum_fixed \
  --preset contact_frame_self_rally_v32_17d_v30_transfer
```

실제 v39 summary에서는 새로 시작한 run이 아니라 v36 checkpoint에서 이어 학습한 run으로 기록되어 있다.

- summary: `artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json`
- `training_mode`: `resume`
- `starting_model_path`: `.../keep1_v36_17d_balanced_xyz012_model.zip`
- `bootstrap_heuristic_episodes`: `0`
- `bootstrap_epochs`: `0`
- `bootstrap`: `null`

## 코드 흐름

1. CLI와 설정을 확정한다.
   - `parse_args()`가 학습 관련 인자를 모두 만든다.
   - `apply_env_preset(args)`가 preset 기본값을 덮어쓴다.
   - `apply_config_overrides(args, args.config_overrides)`가 `--set key=value` 값을 최종 적용한다.
   - 관련 코드: `src/pingpong_rl2/training/cli_config.py`, `src/pingpong_rl2/training/env_config.py`

2. smoke mode와 run 이름을 정리한다.
   - `--smoke`가 켜지면 timestep, n_steps, batch_size, bootstrap episode 수를 줄여 빠른 파이프라인 검증만 한다.
   - `resolve_requested_run_name()`은 `--run-name`, `--run-version`, action mode, smoke 여부를 합쳐 artifact 이름을 만든다.
   - batch size가 rollout size보다 크면 SB3 PPO가 불가능하므로 미리 예외를 낸다.

3. run directory와 환경 설정을 만든다.
   - `build_run_dir()`가 `artifacts/ppo_runs/<run_name>` 경로를 만든다.
   - `resolve_starting_model()`이 새 학습인지 resume인지 결정한다.
   - `env_kwargs_from_args(args)`가 `PingPongKeepUpGymEnv(**env_kwargs)`에 들어갈 설정 dict를 만든다.
   - 실제 env를 한 번 열어 `training_config()`를 읽고 summary에 저장할 resolved env config를 확보한다.

4. 병렬 학습 환경을 만든다.
   - `make_sb3_async_vector_env(num_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed)`를 호출한다.
   - 반환된 vector env는 `VecMonitor`로 감싸져 monitor CSV를 남긴다.
   - `build_reset_xy_curriculum_callback(args)`가 켜져 있으면 reset distribution curriculum이 학습 중 업데이트된다.

5. PPO 모델을 만들거나 불러온다.
   - 새 학습이면 `PPO("MlpPolicy", monitored_env, ...)`를 생성한다.
   - `scale_log_std_by_action_limit`가 켜져 있으면 action limit 비율에 맞춰 policy log_std를 초기화한다.
   - `zero_init_action_mean`이 켜져 있으면 actor mean head를 0으로 초기화한다.
   - resume이면 `PPO.load(starting_model_path, env=monitored_env, device=args.device)`로 모델을 불러오고 새 env를 붙인다.

6. optional heuristic bootstrap을 수행한다.
   - 조건은 `starting_model_path is None and bootstrap_heuristic_episodes > 0 and bootstrap_epochs > 0`이다.
   - `collect_heuristic_bootstrap_dataset()`가 `HeuristicKeepUpPolicy` 행동을 모은다.
   - `bootstrap_actor_from_dataset()`가 PPO actor를 supervised learning 방식으로 먼저 맞춘다.
   - 이 단계는 PPO 알고리즘 자체가 아니라 초기 policy warm start다.
   - v39는 resume run이고 bootstrap 값이 0이라 이 블록을 타지 않았다.

7. PPO 학습, 저장, 평가를 실행한다.
   - `learn_model()`이 `model.learn()`을 감싼다.
   - 완료 후 `<run_name>_model.zip`을 저장한다.
   - `evaluate_model()`로 deterministic evaluation을 실행한다.

8. training summary JSON을 저장한다.
   - config, env_config, bootstrap summary, evaluation summary, monitor path, model path가 들어간다.
   - 이후 `run_ppo_evaluation.py`, `run_ppo_rebound_analysis.py`, `run_viewer.py`가 이 summary에서 env kwargs를 복원한다.

## 주요 호출 관계

```text
run_ppo_learning.py
  -> training/cli_config.py          # CLI, config file, --set override
  -> training/env_config.py          # preset -> env kwargs
  -> training/run_paths.py           # run dir, resume model 결정
  -> envs/gym_env.py                 # Gymnasium wrapper
  -> envs/keepup_env.py              # 실제 keep-up task/env
  -> training/vector_env.py          # SB3 병렬 env
  -> training/curriculum.py          # reset curriculum callback
  -> stable_baselines3.PPO           # policy 학습
  -> training/bootstrap.py           # optional heuristic bootstrap
  -> training/evaluation.py          # 학습 후 평가
```

## 발표 때 설명 포인트

- PPO가 직접 보는 것은 `PingPongKeepUpGymEnv`의 observation이고, 출력은 action space에 맞는 residual/action vector다.
- reward, termination, success count 같은 task logic은 `envs/keepup_env.py`에 있다.
- 학습 설정은 CLI에 흩어져 보이지만 최종 truth는 training summary JSON이다.
- v39는 heuristic으로 직접 학습한 것이 아니라 v36 PPO checkpoint를 이어서 학습했다.
