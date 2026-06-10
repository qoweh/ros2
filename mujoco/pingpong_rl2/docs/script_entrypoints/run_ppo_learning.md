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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일을 읽을 때 가장 중요한 점은 `run_ppo_learning.py`가 로봇팔을 직접 움직이는 파일이 아니라는 점이다. 이 파일은 학습 설정을 확정하고 SB3 PPO에게 환경을 넘긴다. 실제 물리 step은 PPO 내부 rollout 중 `env.step(action)`이 호출될 때 아래로 내려간다.

```text
run_ppo_learning.py
  -> model.learn()
    -> SB3 rollout 수집
      -> VecMonitor.step()
        -> SB3AsyncVectorEnvAdapter.step_wait()
          -> PingPongKeepUpGymEnv.step(action)
            -> PingPongKeepUpEnv.step(action)
              -> action_mode별 residual action 해석
              -> RacketCartesianController.compute_joint_targets()
              -> PingPongSim.step_with_contact_trace(joint_targets)
                -> data.ctrl[:7] = joint_targets
                -> mujoco.mj_step()
```

즉 터미널에서 이 파일을 실행하면 겉으로는 학습 스크립트 하나가 도는 것처럼 보이지만, 실제로는 `training/` 모듈이 설정과 vector env를 만들고, `envs/keepup_env.py`가 MDP의 step/reward/termination을 정의하고, `controllers/ee_pose_controller.py`가 라켓 목표를 7개 관절 목표로 바꾸며, `envs/pingpong_sim.py`가 MuJoCo에 actuator 값을 넣고 물리를 진행한다.

## main()을 코드 순서대로 풀어보기

첫 부분의 `parse_args()`는 argparse 설정을 읽는 단계다. 다만 이 프로젝트에서는 CLI 값만 쓰지 않는다. `apply_env_preset(args)`가 preset을 먼저 적용하고, 그 다음 `apply_config_overrides(args, args.config_overrides)`가 `--set key=value` 같은 최종 덮어쓰기를 반영한다. 그래서 같은 명령이라도 `--preset`, `--config-file`, `--set`이 어떤 순서로 섞였는지에 따라 최종 환경이 달라질 수 있다. 최종적으로 믿어야 하는 값은 콘솔에 찍힌 일부 값이 아니라 `<run_name>_training_summary.json` 안의 `config`와 `env_config`다.

그 다음 smoke mode가 처리된다. smoke mode는 알고리즘을 바꾸는 옵션이 아니라 전체 파이프라인이 깨지지 않는지 확인하기 위해 timestep, rollout 크기, bootstrap episode 수, 평가 episode 수를 작게 줄이는 실행 모드다. 발표나 결과 분석용 모델을 만들기 위한 모드가 아니라 빠른 검증용이다.

`resolve_requested_run_name()`은 `--run-name`, `--run-version`, `action_mode`, smoke 여부를 묶어서 artifact 이름을 만든다. 이 이름은 단순 표시용이 아니라 모델 zip, monitor CSV, summary JSON 경로에 직접 연결된다. 이어서 `resolve_tilt_profile(args)`는 tilt profile 이름을 실제 pitch/roll 제한값으로 해석한다. `batch_size <= n_steps * n_envs` 검사는 SB3 PPO가 rollout buffer에서 batch를 잘라 학습할 수 있는지 확인하는 방어 코드다.

환경 설정은 `env_kwargs_from_args(args)`에서 만들어진다. 이 dict가 결국 `PingPongKeepUpGymEnv(**env_kwargs)`로 들어간다. 여기서 한 번 `config_env`를 만들어 `training_config()`를 읽는 이유는, CLI와 preset으로부터 나온 값을 환경 생성자가 내부 기본값/검증/파생값까지 반영한 뒤의 최종 환경 설정을 저장하기 위해서다.

PPO 학습용 환경은 단일 env가 아니라 `make_sb3_async_vector_env()`로 만든 vector env다. `n_envs`가 여러 개면 PPO는 동시에 여러 episode 조각을 모은다. 그 위에 `VecMonitor`를 씌워 episode return과 length를 monitor CSV로 남긴다. reset curriculum이 켜져 있으면 `build_reset_xy_curriculum_callback(args)`가 만든 callback이 학습 진행률에 따라 reset 분포를 바꾼다.

모델 생성 분기는 `starting_model_path is None`인지로 갈린다. 새 학습이면 `PPO("MlpPolicy", monitored_env, ...)`를 만든다. resume이면 `PPO.load(..., env=monitored_env)`로 기존 policy parameter를 불러오고 현재 실행에서 새로 만든 env를 붙인다. 그래서 resume은 이전 모델의 network를 이어 쓰지만, rollout은 현재 `env_kwargs`로 만든 환경에서 다시 수집된다.

새 학습에서만 policy 초기화 옵션이 적용된다. `initialize_scaled_policy_log_std()`는 action limit에 맞춰 Gaussian policy의 초기 표준편차를 조절한다. `zero_init_action_mean`은 actor의 action mean head를 0으로 초기화한다. 둘 다 로봇팔을 직접 움직이는 로직이 아니라 PPO policy의 초기 action 분포를 안정적으로 잡는 장치다.

## heuristic bootstrap이 정확히 하는 일

heuristic bootstrap은 PPO 알고리즘 자체가 아니다. 조건은 다음과 같다.

```text
starting_model_path is None
and bootstrap_heuristic_episodes > 0
and bootstrap_epochs > 0
```

이 조건 때문에 resume run에서는 bootstrap을 타지 않는다. 최종 발표 기준 v39는 summary상 `training_mode=resume`, `bootstrap=null`이라 이 블록을 직접 실행하지 않았다.

bootstrap이 켜진 새 run에서는 먼저 `collect_heuristic_bootstrap_dataset()`가 별도의 `PingPongKeepUpGymEnv`를 만들고 `HeuristicKeepUpPolicy`로 episode를 실행한다. 이때 저장되는 것은 `observation`과 heuristic이 낸 `action` 쌍이다. sample mode에 따라 전체 episode sample을 쓰거나, 유효 타격 이후 sample만 쓰거나, 다음 intercept가 reachable인 sample만 고른다.

그 다음 `bootstrap_actor_from_dataset()`가 PPO policy network를 supervised learning으로 맞춘다. 코드적으로는 observation batch를 policy에 넣고, deterministic action을 꺼낸 뒤, heuristic action과의 MSE loss를 줄인다.

```text
observation batch
  -> model.policy.get_distribution(obs)
  -> deterministic predicted_action
  -> MSE(predicted_action, heuristic_action)
  -> actor parameter update
```

이 과정은 reward를 보고 배우는 PPO update가 아니다. 랜덤 policy가 공을 거의 맞히지 못해 학습 초반 signal이 희박해지는 문제를 줄이기 위한 actor warm start다. 쉽게 말하면 "처음부터 말도 안 되는 residual action을 내지 말고, hand-coded policy가 하던 근처에서 시작하라"는 초기화다.

## PPO 학습 중 로봇팔이 움직이는 순간

`learn_model()`은 얇은 wrapper이고, 실질적인 학습은 `model.learn()` 안에서 일어난다. SB3는 rollout을 모으기 위해 계속 `env.step(action)`을 호출한다. 여기서 action은 관절각이나 torque가 아니다. 최종 17D action mode에서는 위치, tilt, velocity, apex, strike plane, tracking residual처럼 "기본 타격 계획을 보정하는 값"이다.

`PingPongKeepUpEnv.step()`은 이 residual action을 읽어 목표 라켓 위치와 속도, 라켓 기울기를 계산한다. 그 다음 `RacketCartesianController.compute_joint_targets()`가 Cartesian 목표를 7개 joint target으로 바꾼다. 이 controller는 MuJoCo site Jacobian을 사용한다.

```text
라켓 목표 위치/방향 오차 e
  -> Jacobian J(q)
  -> dq = J^T (J J^T + lambda I)^-1 e
  -> 현재 관절값 + dq
  -> joint_targets
```

마지막으로 `PingPongSim.step_with_contact_trace()`가 `data.ctrl[:7]`에 joint target을 넣고 `mujoco.mj_step()`을 여러 substep 실행한다. 그러면 물리엔진이 관절, 라켓, 공, 접촉, 속도, 위치의 다음 상태를 적분해서 만든다. 그 결과로 reward, terminated/truncated, info가 만들어지고 PPO가 그 경험을 학습에 사용한다.

## summary JSON이 중요한 이유

학습이 끝나면 이 파일은 모델만 저장하지 않는다. `summary` dict 안에 실행 당시 config, resolved env config, bootstrap summary, evaluation summary, model path, monitor path를 같이 저장한다. 이후 `run_ppo_evaluation.py`, `run_ppo_rebound_analysis.py`, `run_viewer.py`는 이 summary를 읽어 당시 환경 설정을 복원한다.

따라서 "이 모델은 어떤 action mode로 학습했나?", "bootstrap을 썼나?", "reset 분포가 뭐였나?", "학습 직후 평가 결과가 어땠나?" 같은 질문은 코드의 기본값만 보면 안 되고, 해당 run의 training summary JSON을 확인해야 한다.

## 발표 때 설명 포인트

- PPO가 직접 보는 것은 `PingPongKeepUpGymEnv`의 observation이고, 출력은 action space에 맞는 residual/action vector다.
- reward, termination, success count 같은 task logic은 `envs/keepup_env.py`에 있다.
- 학습 설정은 CLI에 흩어져 보이지만 최종 truth는 training summary JSON이다.
- v39는 heuristic으로 직접 학습한 것이 아니라 v36 PPO checkpoint를 이어서 학습했다.
