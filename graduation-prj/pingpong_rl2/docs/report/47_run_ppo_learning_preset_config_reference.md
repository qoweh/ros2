# run_ppo_learning Preset And Config Reference

## 한줄 결론

`run_ppo_learning.py`에서 `--preset`은 단순 별칭이 아니라 self-rally 환경, action space, reward, PPO hyperparameter, checkpoint/evaluation 기준을 한 번에 고정하는 실험 설정이다. preset을 넣지 않으면 기본값은 아주 초기 `position` baseline이라 현재 목표인 탁구공 self-rally와 다르다. 앞으로는 긴 CLI 대신 `--config-file`로 실행 인자를 묶고, 드문 일회성 변경은 `--set KEY=VALUE`로 처리한다.

## 실행 순서

1. argparse 기본값을 만든다.

아무 인자도 넣지 않으면 `action_mode=position`, `max_episode_steps=600`, `total_timesteps=20000`, `n_steps=256`, `batch_size=256`, `learning_rate=3e-4` 같은 초기 baseline 값이 들어간다.

2. `--config-file`을 읽는다.

JSON 파일의 `args` 객체 또는 top-level key를 argparse destination 이름으로 읽는다. 예를 들어 `run_version` 또는 `run-version` 둘 다 가능하다. CLI로 직접 넣은 값은 config file 값보다 우선한다.

3. `--preset`을 적용한다.

preset은 `_ENV_PRESETS`에 저장된 dict다. preset-managed 인자가 아직 기본값이면 preset 값으로 바꾼다. 이미 다른 값으로 바뀌어 있으면 충돌로 보고 에러를 낸다. 이 동작은 "의도치 않은 혼합 실험"을 막기 위한 것이다.

4. `--set KEY=VALUE` override를 적용한다.

`--set`은 preset 적용 후 실행된다. 따라서 `--set bootstrap_heuristic_episodes=0`처럼 preset 값을 확실히 덮어쓸 수 있다. 값은 JSON으로 파싱되므로 숫자, bool, list를 직접 넣을 수 있다.

5. `--smoke`가 있으면 학습량을 강제로 줄인다.

`total_timesteps=1024`, `n_steps=64`, `batch_size=64`, `n_envs<=2`, eval/checkpoint episode는 최대 2로 줄인다. smoke는 성능 평가가 아니라 preset/env/model load가 깨지지 않는지 확인하는 용도다.

6. run directory와 시작 모델을 결정한다.

`--resume-from`이 있으면 그 checkpoint에서 시작한다. 없고 run directory 안에 `<run_name>_model.zip`이 있으면 자동 resume한다. `--reset-model`은 기존 checkpoint가 있어도 새 모델로 시작한다.

7. env kwargs를 만든 뒤 PPO를 학습한다.

`env_kwargs_from_args()`가 argparse 값을 `PingPongKeepUpGymEnv` 생성자 인자로 바꾼다. 학습 후에는 model, checkpoint history, training summary JSON을 저장한다.

## preset의 역할

`--preset`은 아래를 한 번에 고정한다.

| 범위 | 역할 |
| --- | --- |
| action mode | RL이 어떤 action residual을 직접 학습하는지 결정 |
| observation | phase, contact context, next intercept, desired outgoing velocity를 볼지 결정 |
| reset distribution | 공 시작 xy, z, 초기 속도 범위 결정 |
| controller/feedforward | contact-frame planner, strike plane, follow-through, brake, body clearance 결정 |
| action bounds | radial/tangent/z, tilt, velocity, apex residual의 상한/하한 결정 |
| reward/success | useful contact 조건, apex/easy-next-ball/stable-cycle 보상 결정 |
| PPO hyperparameter | rollout 크기, batch, learning rate, epochs, clip range 결정 |
| evaluation | checkpoint 간격, checkpoint eval episode 수, final eval episode 수 결정 |

즉 현재 self-rally 학습에서는 `--preset` 없이 실행하면 거의 다른 과제를 학습하게 된다.

## config file의 역할

추가된 설정파일:

```text
configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json
```

내용:

```json
{
  "args": {
    "preset": "contact_frame_self_rally_v25_long_horizon_30_bounce",
    "run_name": "pmk_cf_self_rally",
    "run_version": "v25",
    "resume_from": "artifacts/ppo_runs/pmk_cf_self_rally_v23/pmk_cf_self_rally_v23_model.zip",
    "total_timesteps": 500000
  }
}
```

권장 실행 형태:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json
```

새 실험을 할 때는 `run_version`만 CLI로 바꾸면 된다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json \
  --run-version v26
```

주의: 위 v25 config를 그대로 다시 실행하면 `pmk_cf_self_rally_v25` run directory에 다시 저장될 수 있다.

드문 override 예시:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json \
  --run-version smoke_check \
  --smoke \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

## 주요 기본값

아무 preset/config도 넣지 않았을 때의 argparse 기본값이다.

| 인자 | 기본값 | 역할 |
| --- | ---: | --- |
| `config_file` | `None` | JSON 설정파일 경로 |
| `set` | `[]` | preset/config 값을 덮는 generic override |
| `preset` | `None` | 실험 preset 선택. 없으면 manual baseline |
| `run_name` | `None` | run base name. 없으면 action mode별 기본 이름 사용 |
| `run_version` | `None` | `<run_name>_<version>` suffix |
| `output_dir` | `None` | 출력 directory. 없으면 `artifacts/ppo_runs/<run>` |
| `resume_from` | `None` | 명시 checkpoint에서 resume |
| `reset_model` | `False` | 기존 모델 무시하고 새 PPO 생성 |
| `total_timesteps` | `20000` | 이번 실행에서 학습할 PPO step 수 |
| `n_envs` | `4` | 병렬 환경 수 |
| `n_steps` | `256` | PPO rollout 길이 |
| `batch_size` | `256` | PPO minibatch 크기 |
| `learning_rate` | `0.0003` | PPO optimizer learning rate |
| `gamma` | `0.99` | discount factor |
| `n_epochs` | `10` | rollout 하나당 PPO update epoch |
| `clip_range` | `0.2` | PPO clipped objective 범위 |
| `ent_coef` | `0.0` | entropy bonus |
| `vf_coef` | `0.5` | value loss coefficient |
| `seed` | `7` | env/model seed |
| `device` | `auto` | CPU/GPU 자동 선택 |
| `scene_path` | `None` | MuJoCo XML. preset이 보통 `assets/scene.xml` 지정 |
| `action_mode` | `position` | 기본은 단순 위치 action |
| `ball_height` | `0.50` | 기본 공 시작 높이 |
| `target_ball_height` | `None` | 목표 apex. 없으면 `ball_height`를 사용 |
| `max_episode_steps` | `600` | episode step 제한 |
| `reset_xy_range` | `0.06` | 시작 xy randomization 범위 |
| `reset_ball_height_range` | `0.0` | 시작 z randomization 범위 |
| `reset_velocity_xy_range` | `0.01` | 시작 xy 속도 randomization |
| `reset_velocity_z_range` | `[-0.02, 0.01]` | 시작 z 속도 randomization |
| `success_velocity_threshold` | `0.5` | upward useful contact 속도 기준 |
| `eval_episodes` | `5` | 학습 후 deterministic final eval episode 수 |
| `checkpoint_interval` | `10000` | checkpoint 저장/eval 간격 |
| `checkpoint_eval_episodes` | `10` | checkpoint마다 평가할 episode 수 |
| `early_stop_patience_evals` | `0` | non-improving eval 조기 종료. 0이면 비활성 |
| `smoke` | `False` | 빠른 연결 확인 모드 |

## 기본값 그룹별 의미

아래 인자들은 기본값이 대부분 `None` 또는 `False`다. 이 상태에서는 env에 전달되지 않거나 기능이 꺼져 있고, preset이 켤 때만 활성화된다.

| 그룹 | 기본값 | 역할 |
| --- | --- | --- |
| `lateral_action_limit`, `vertical_action_limit`, `tilt_action_limit` | `None` | action bound 직접 지정 |
| `target_tilt_limit`, `target_pitch_range`, `initial_target_tilt` | `None` | tilt clamp/초기 tilt |
| `strike_tilt_*`, `followup_strike_*` | `None` | contact 전후 scripted tilt/contact bias |
| `contact_frame_*_action_limit` | `None` | contact-frame residual action bound |
| `contact_frame_planner_enabled` | `False` | self-rally contact-frame planner 사용 여부 |
| `contact_frame_planner_hold_during_descent` | `True` | 하강 중 planner target 유지 |
| `contact_frame_*gain`, `*_max`, `*_time` | `None` | velocity lead, follow-through, lateral brake, recovery lift 같은 feedforward 보정 |
| `controller_*` | `None` | IK/orientation/velocity/nullspace/body-clearance gain |
| `include_*_observation` | `False` | observation 확장 기능 |
| `desired_outgoing_xy_mode` | `next_intercept` | desired outgoing xy 계산 방식 |
| `require_*_for_success` | `False` | useful contact 판정 강화 |
| `terminate_on_*` | `False` | non-useful/low-apex contact 종료 조건 |
| `low_apex_contact_height_threshold` | `None` | low-apex 종료 threshold |
| `low_apex_contact_grace_count` | `0` | 연속 low-apex 허용 횟수 |
| `*_reward_weight`, `*_penalty_weight` | `None` | 추가 reward/penalty shaping |
| `stable_cycle_reward_cap` | `4` | stable streak reward cap |
| `log_std_init` | `None` | PPO Gaussian log std 초기값 |
| `scale_log_std_by_action_limit` | `False` | action bound 기준 per-axis std 초기화 |
| `action_std_limit_ratio`, `action_std_min`, `action_std_max` | `None` | scaled std 세부값 |
| `zero_init_action_mean` | `False` | residual policy mean을 0으로 초기화 |
| `bootstrap_*` | 기본 비활성 | heuristic rollout supervised warm-start |

## v25 preset 핵심값

현재 성능이 나온 `contact_frame_self_rally_v25_long_horizon_30_bounce`의 핵심 설정이다.

| 항목 | 값 | 의미 |
| --- | ---: | --- |
| `action_mode` | `position_contact_frame_velocity_tilt_lateral_apex_residual` | 15D residual policy |
| `max_episode_steps` | `1800` | 30회 이상 지속을 기록할 수 있게 horizon 확장 |
| `ball_height` | `0.34` | 시작 높이 |
| `target_ball_height` | `0.30` | 목표 post-contact apex |
| `reset_xy_range` | `0.028` | sweet spot 근처 시작 |
| `reset_ball_height_range` | `0.02` | 시작 높이 약간 다양화 |
| `reset_velocity_xy_range` | `0.0` | 시작 lateral velocity 없음 |
| `reset_velocity_z_range` | `[-0.01, 0.01]` | 시작 z velocity 약간 다양화 |
| `n_steps` | `512` | rollout 길이 |
| `batch_size` | `512` | PPO batch |
| `learning_rate` | `0.00002` | resume fine-tuning용 낮은 lr |
| `n_epochs` | `2` | PPO update 과격함 제한 |
| `clip_range` | `0.08` | 정책 변화폭 제한 |
| `checkpoint_interval` | `100000` | 100k마다 checkpoint/eval |
| `checkpoint_eval_episodes` | `30` | checkpoint ranking 신뢰도 증가 |
| `eval_episodes` | `80` | 학습 직후 eval 수 |
| `stable_cycle_reward_cap` | `12` | 긴 안정 루프 보상 유지 |
| `low_apex_contact_height_threshold` | `0.14` | 진짜 낮은 통통 루프만 종료 |
| `low_apex_contact_grace_count` | `3` | 낮은 contact 3회까지 회복 허용 |
| `scale_log_std_by_action_limit` | `True` | 작은 action bound에 맞춰 PPO std 초기화 |
| `action_std_limit_ratio` | `0.35` | action limit의 35%를 초기 std로 사용 |
| `zero_init_action_mean` | `True` | residual zero를 시작 정책 중심으로 둠 |

v25 15D action bound:

| action 축 | bound |
| --- | ---: |
| radial/tangent 위치 residual | `+-0.02m` |
| z 위치 residual | `+-0.03m` |
| pitch/roll tilt residual | `+-0.008rad` |
| `vz_scale` | `+-0.35` |
| outgoing x/y residual | `+-0.35m/s` |
| racket vz residual | `+-0.45m/s` |
| trajectory/centering tilt scale | `+-0.75` |
| racket vx/vy residual | `+-0.35m/s` |
| target apex z residual | `+-0.08m` |
| strike plane z residual | `+-0.025m` |

## 정리/리팩토링 결과

- `run_ppo_learning.py`에 `--config-file`을 추가했다.
- `run_ppo_learning.py`에 `--set KEY=VALUE`를 추가했다.
  - 개별 low-use CLI를 더 늘리지 않고 일회성 override를 처리한다.
  - `--set`은 preset 뒤에 적용되어 preset 값을 확실히 덮어쓴다.
  - JSON list와 Python tuple이 같은 값인데 충돌하는 문제를 `values_equal()`로 보정했다.
- `configs/` directory를 만들고 v25 재현 config를 추가했다.
- 학습 summary의 `config`에 `config_file`, `config_overrides`, `eval_episodes`를 기록하도록 보강했다.
- 생성물인 `__pycache__`/`.pyc`는 정리했다.
- tracked script/source 파일은 삭제하지 않았다. `run_material_sanity.py`, `run_contact_feasibility_map.py`, `run_heuristic_keepup_diagnostic.py` 같은 파일은 자주 실행하지 않더라도 물성 검증, contact upper-bound, heuristic baseline 재현에 쓰이는 실험 도구라서 졸작 발표 전 삭제 리스크가 더 크다.
- 기존 low-use CLI 인자는 과거 report와 모델 재현 명령 때문에 완전 삭제하지 않았다. 대신 새 실험 경로는 config file + preset + `--set`으로 고정해서 실제로 만지는 표면을 줄였다.

## 발표 때 설명 포인트

- "preset 없이 학습"은 현재 목표와 다른 baseline이다.
- "v25 성능 향상"은 긴 horizon, 30회 기준 checkpoint, v23 resume 안정성의 조합이다.
- PPO hyperparameter는 큰 탐색이 아니라 resume fine-tuning용으로 보수적이다.
- 환경/reward/action 설정은 preset이 고정하고, config file은 실행 편의와 재현성을 담당한다.
- 드문 값 변경은 `--set`으로 남겨서 CLI 인자 추가를 멈춘다.
